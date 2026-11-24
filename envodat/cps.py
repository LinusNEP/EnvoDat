"""Raw data from the EnvoDat share on the CPS server.
"""

import base64
import fnmatch
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from . import _config

_PROPFIND = (b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
             b"<d:resourcetype/><d:getcontentlength/><d:getlastmodified/>"
             b"</d:prop></d:propfind>")
_NS = {"d": "DAV:"}
TIMEOUT = 120   # seconds per request
RETRIES = 4     # extra attempts after a timeout or 5xx/429 answer


@dataclass
class RemoteFile:
    path: str   # relative to the share root, '/'-separated
    size: int


class CPSShare:

    def __init__(self, share_url=None, password=""):
        url = urlparse(share_url or _config.CPS_SHARE_URL)
        self.host = f"{url.scheme}://{url.netloc}"
        self.token = url.path.rstrip("/").rsplit("/", 1)[-1]
        self.password = password or os.environ.get("ENVODAT_CPS_PASSWORD", "")
        self._base = None  # (dav_root_url, auth_header) once detected

    # -- protocol ---------------------------------------------------------
    def _candidates(self):
        # Nextcloud >= 29 serves unprotected shares without auth; protected ones use
        # "anonymous:<password>". Older releases use "<token>:<password>".
        new_auth = ("anonymous:" + self.password) if self.password else None
        old_auth = self.token + ":" + self.password
        return [
            (f"{self.host}/public.php/dav/files/{self.token}", new_auth),  # Nextcloud >= 29
            (f"{self.host}/public.php/webdav", old_auth),                  # older releases
        ]

    def _request(self, url, auth, method="GET", headers=None, data=None):
        # Nextcloud rejects public-share WebDAV requests without this header (401).
        h = {"User-Agent": "envodat", "X-Requested-With": "XMLHttpRequest"}
        if auth:
            h["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        # Retry timeouts, dropped connections and server-side errors (busy server);
        # 4xx answers such as 401/404 are final and raised at once.
        for attempt in range(RETRIES + 1):
            try:
                return urllib.request.urlopen(req, timeout=TIMEOUT)
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429 or attempt == RETRIES:
                    raise
            except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
                if attempt == RETRIES:
                    raise urllib.error.URLError(getattr(exc, "reason", exc)) from exc
            wait = 10 * (attempt + 1)
            print(f"  server busy or unreachable, retrying in {wait} s ...", file=sys.stderr)
            time.sleep(wait)

    def _propfind(self, rel):
        bases = [self._base] if self._base else self._candidates()
        last = None
        for base, auth in bases:
            url = base + "/" + quote(rel.strip("/"))
            try:
                with self._request(url, auth, "PROPFIND",
                                   {"Depth": "1", "Content-Type": "application/xml"},
                                   _PROPFIND) as r:
                    body = r.read()
                self._base = (base, auth)
                return base, body
            except urllib.error.HTTPError as exc:
                last = exc  # try the next endpoint flavour
            except urllib.error.URLError as exc:
                raise ConnectionError(f"Cannot reach {self.host}: {exc.reason}") from exc
        hint = ""
        if getattr(last, "code", None) == 401:
            hint = (" The share may be password-protected: pass --password (or set "
                    "ENVODAT_CPS_PASSWORD)." if not self.password else
                    " Check the share password.")
        raise ConnectionError(f"WebDAV listing of '{rel or '/'}' failed: {last}.{hint}")

    # -- public API -------------------------------------------------------
    def listdir(self, path=""):
        """List one folder. Returns (subfolders, files) with share-relative paths."""
        base, body = self._propfind(path)
        prefix = unquote(urlparse(base).path).rstrip("/") + "/"
        dirs, files = [], []
        for resp in ET.fromstring(body).findall("d:response", _NS):
            href = unquote(urlparse(resp.findtext("d:href", "", _NS)).path)
            rel = href[len(prefix):].strip("/") if href.startswith(prefix) else href.strip("/")
            if rel == path.strip("/"):
                continue  # the folder itself
            prop = resp.find("d:propstat/d:prop", _NS)
            if prop is not None and prop.find("d:resourcetype/d:collection", _NS) is not None:
                dirs.append(rel)
            else:
                size = prop.findtext("d:getcontentlength", "0", _NS) if prop is not None else "0"
                files.append(RemoteFile(rel, int(size or 0)))
        return dirs, files

    def walk(self, path=""):
        """Yield every file below ``path`` (recursive, one listing per folder)."""
        dirs, files = self.listdir(path)
        yield from files
        for d in dirs:
            yield from self.walk(d)

    def download_file(self, remote, dest, quiet=False):
        """Download one :class:`RemoteFile` to ``dest``, resuming a partial ``.part`` file."""
        if self._base is None:
            self._propfind("")
        base, auth = self._base
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == remote.size:
            return dest
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        url = base + "/" + quote(remote.path)
        with self._request(url, auth, headers=headers) as r, open(part, "ab" if have else "wb") as f:
            if have and r.status != 206:  # server ignored the range: start over
                f.seek(0), f.truncate()
                have = 0
            done = have
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not quiet and remote.size:
                    print(f"\r  {100 * done // remote.size:3d}%  {remote.path}", end="",
                          file=sys.stderr, flush=True)
        if not quiet:
            print(file=sys.stderr)
        if remote.size and part.stat().st_size != remote.size:
            raise IOError(f"Incomplete download of {remote.path}; rerun to resume.")
        part.replace(dest)
        return dest


def _matches(path, include, exclude):
    p = path.lower()
    ok = any(fnmatch.fnmatch(p, pat.lower()) for pat in include) if include else True
    return ok and not any(fnmatch.fnmatch(p, pat.lower()) for pat in exclude or [])


def cps_list(include=None, exclude=None, path="", share_url=None, password=""):
    share = CPSShare(share_url, password)
    return [f for f in share.walk(path) if _matches(f.path, _as_list(include), _as_list(exclude))]


def download_raw(include, exclude=None, out=None, dry_run=False, share_url=None, password=""):
    share = CPSShare(share_url, password)
    files = [f for f in share.walk() if _matches(f.path, _as_list(include), _as_list(exclude))]
    total = sum(f.size for f in files)
    print(f"{len(files)} file(s), {total / 1e9:.2f} GB", file=sys.stderr)
    if dry_run:
        return files
    out = Path(out) if out else _config.root() / "raw"
    probe = out
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if total > free:
        raise OSError(f"Need {total / 1e9:.1f} GB but only {free / 1e9:.1f} GB free at {out}")
    return [share.download_file(f, out / f.path) for f in files]


def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)
