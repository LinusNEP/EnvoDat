#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from envodat import _config  # noqa: E402
from envodat.catalog import SCENES, match_scene  # noqa: E402
from envodat.cps import CPSShare, _matches  # noqa: E402

GB = 1e9
HF_FILE_LIMIT_GB = 50    # hard per-file limit on the Hub
HF_REPO_HINT_GB = 300    # above this, Hugging Face asks you to get in touch first


def _share(args):
    return CPSShare(args.share, args.password)


def _dup_key(path):
    parts = path.split("/")
    for i, part in enumerate(parts[:-1]):
        sid = match_scene(part)
        if sid:
            return sid, "/".join(parts[i + 1:]).lower()
    return None


def _preferred(path, sid):
    official = SCENES.get(sid, {}).get("name", "")
    return (0 if official in path.split("/") else 1, path)


def dedupe(files, quiet=False):
    groups, kept, skipped = {}, [], []
    for f in files:
        key = _dup_key(f.path)
        if key is None:
            kept.append(f)
        else:
            groups.setdefault(key, []).append(f)
    for (sid, _), copies in groups.items():
        copies.sort(key=lambda f: _preferred(f.path, sid))
        kept.append(copies[0])
        skipped += copies[1:]
        if not quiet and len({c.size for c in copies}) > 1:
            print(f"  NOTE: copies of {copies[0].path} differ in size; keeping it, check these:")
            for c in copies[1:]:
                print(f"        {c.size / GB:8.3f} GB  {c.path}")
    kept.sort(key=lambda f: f.path)
    return kept, skipped


def _listing(args, share):
    if args.tree:
        from envodat.cps import RemoteFile
        files = []
        with open(args.tree) as f:
            next(f, None)  # header
            for line in f:
                size, _, path = line.rstrip("\n").split("\t", 2)
                if path.startswith(args.path.strip("/")):
                    files.append(RemoteFile(path, int(size)))
        print(f"Using the saved listing {args.tree} ({len(files)} files)")
        return files
    return list(share.walk(args.path))


def _select(args):
    share = _share(args)
    files = [f for f in _listing(args, share) if _matches(f.path, args.include, args.exclude)]
    skipped = []
    if not args.no_dedupe:
        files, skipped = dedupe(files)
        if skipped:
            print(f"Skipping {len(skipped)} duplicate copies "
                  f"({sum(f.size for f in skipped) / GB:.2f} GB) under other spellings of a scene "
                  "folder.")
    return share, files


def cmd_tree(args):
    files = list(_share(args).walk(args.path))
    by_top, by_ext = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    with open(args.out, "w") as f:
        f.write("size_bytes\tscene\tpath\n")
        for r in files:
            f.write(f"{r.size}\t{match_scene(r.path) or ''}\t{r.path}\n")
            top = r.path.split("/", 1)[0] if "/" in r.path else "(root)"
            ext = Path(r.path).suffix.lower() or "(none)"
            for d, k in ((by_top, top), (by_ext, ext)):
                d[k][0] += 1
                d[k][1] += r.size
    total = sum(r.size for r in files)
    print(f"{len(files)} files, {total / GB:.1f} GB  ->  full listing in {args.out}\n")
    for title, d in (("By top-level folder", by_top), ("By extension", by_ext)):
        print(title)
        for k, (n, s) in sorted(d.items(), key=lambda kv: -kv[1][1]):
            print(f"  {k:<40} {n:>8} files {s / GB:>10.2f} GB")
        print()
    _, dups = dedupe(files, quiet=True)
    if dups:
        print(f"{len(dups)} files ({sum(r.size for r in dups) / GB:.1f} GB) are duplicate copies under "
              "other spellings of a scene folder.\n")
    big = [r for r in files if r.size > HF_FILE_LIMIT_GB * GB]
    if big:
        print(f"{len(big)} file(s) exceed the Hub's {HF_FILE_LIMIT_GB} GB per-file limit "
              f"(split them before a raw upload):")
        for r in big:
            print(f"  {r.size / GB:8.1f} GB  {r.path}")


def _extract(path):
    name = path.name.lower()
    target = path.parent / path.name.split(".")[0]
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(target)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(path) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(target, filter="data")
            else:
                t.extractall(target)
    else:
        return False
    return True


def cmd_annotated(args):
    import hf_upload

    share, files = _select(args)
    total = sum(f.size for f in files)
    print(f"{len(files)} file(s) selected, {total / GB:.2f} GB")
    if not files:
        sys.exit("Nothing matched. Run `tree` and adjust --include.")
    if args.dry_run:
        for f in files:
            print(f"  {f.size / GB:8.3f} GB  {match_scene(f.path) or '(no scene)':<12} {f.path}")
        return
    work = Path(args.work)
    for f in files:
        local = share.download_file(f, work / f.path)
        if _extract(local):
            print(f"  extracted {f.path}")
            if not args.keep_archives:
                local.unlink()
    hf_upload.publish(work, args.stage_dir, args.repo_id, copy=False, public=args.public,
                      stage_only=args.stage_only)
    if not args.stage_only and not args.keep:  # staging symlinks into work/, so clean both
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(args.stage_dir, ignore_errors=True)


def cmd_raw(args):
    from huggingface_hub import HfApi

    share, files = _select(args)
    too_big = [f for f in files if f.size > HF_FILE_LIMIT_GB * GB]
    files = [f for f in files if f.size <= HF_FILE_LIMIT_GB * GB]
    total = sum(f.size for f in files)
    print(f"{len(files)} file(s) to transfer, {total / GB:.1f} GB")
    for f in too_big:
        print(f"  SKIP (> {HF_FILE_LIMIT_GB} GB per-file limit): {f.path} ({f.size / GB:.1f} GB)")
    if args.dry_run or not files:
        for f in files:
            print(f"  {f.size / GB:8.2f} GB  {f.path}")
        return
    if total > HF_REPO_HINT_GB * GB and not args.yes:
        sys.exit(f"This is {total / GB:.0f} GB. Hugging Face asks for repos above ~{HF_REPO_HINT_GB} GB "
                 "to be agreed with them first (datasets@huggingface.co). Rerun with --yes once "
                 "they have confirmed the storage.")

    api = HfApi()
    import hf_upload
    hf_upload.ensure_repo(api, args.repo_id, args.public)
    on_hub = {}
    for item in api.list_repo_tree(args.repo_id, repo_type="dataset", path_in_repo=args.prefix,
                                   recursive=True):
        if getattr(item, "size", None) is not None:
            on_hub[item.path] = item.size
    work = Path(args.work)
    for i, f in enumerate(files, 1):
        dest = f"{args.prefix}/{f.path}"
        if on_hub.get(dest) == f.size:
            print(f"[{i}/{len(files)}] already on Hub: {dest}")
            continue
        print(f"[{i}/{len(files)}] {f.path} ({f.size / GB:.2f} GB)")
        local = share.download_file(f, work / f.path)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=dest, repo_id=args.repo_id,
                        repo_type="dataset", commit_message=f"Add {dest}")
        local.unlink()
    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.prefix}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", default=_config.CPS_SHARE_URL, help="Nextcloud public share URL")
    ap.add_argument("--password", default=os.environ.get("ENVODAT_CPS_PASSWORD", ""),
                    help="Share password, if any (or $ENVODAT_CPS_PASSWORD)")
    ap.add_argument("--path", default="", help="Only consider this folder of the share")
    ap.add_argument("--tree", metavar="TSV",
                    help="Use the file list saved by `tree` (cps_tree.tsv) instead of listing "
                         "the share again; much faster and gentler on the server")
    ap.add_argument("--repo-id", default=_config.HF_REPO_ID)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tree", help="List the share and summarise sizes")
    p.add_argument("--out", default="cps_tree.tsv")
    p.set_defaults(fn=cmd_tree)

    p = sub.add_parser("annotated", help="Download annotation exports and publish them")
    p.add_argument("--include", nargs="+", default=["*yolo*"],
                   help="Glob(s) selecting the YOLO exports (default: '*yolo*')")
    p.add_argument("--exclude", nargs="*")
    p.add_argument("--work", default="./cps_work", help="Download/extract folder")
    p.add_argument("--stage-dir", default="./hf_staging")
    p.add_argument("--stage-only", action="store_true", help="Stage locally, do not upload")
    p.add_argument("--keep", action="store_true", help="Keep the download folder afterwards")
    p.add_argument("--keep-archives", action="store_true", help="Keep archives after extraction")
    p.add_argument("--dry-run", action="store_true", help="Only list the files that would be used")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Keep duplicate copies under other spellings of a scene folder")
    p.add_argument("--public", action="store_true",
                   help="Upload to a PUBLIC repo (default: private, only you can see it)")
    p.set_defaults(fn=cmd_annotated)

    p = sub.add_parser("raw", help="Transfer raw files one by one (resumable)")
    p.add_argument("--include", nargs="+", required=True, help="Glob(s), e.g. '*.bag'")
    p.add_argument("--exclude", nargs="*")
    p.add_argument("--prefix", default="raw", help="Folder on the Hub (default raw/)")
    p.add_argument("--work", default="./cps_work")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Keep duplicate copies under other spellings of a scene folder")
    p.add_argument("--yes", action="store_true", help="Confirm a transfer above ~300 GB")
    p.add_argument("--public", action="store_true",
                   help="Upload to a PUBLIC repo (default: private, only you can see it)")
    p.set_defaults(fn=cmd_raw)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
