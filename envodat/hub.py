"""Annotated, training-ready scenes on the Hugging Face Hub."""

import sys

import yaml

from . import _config
from .catalog import SCENES, match_scene

SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}
COMPLETE_MARKER = ".envodat-complete"   # written only after a scene has been fully unpacked


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError("Install the Hub client:  pip install huggingface_hub") from exc
    return huggingface_hub


def _as_list(scene):
    if scene is None:
        return []
    return [scene] if isinstance(scene, str) else list(scene)


def _normalise(scenes):
    out = []
    for s in scenes:
        sid = s if s == "all" else (match_scene(s) or s.lower())
        out.append(sid)
    return out


def _gated_message(repo_id):
    return (f"EnvoDat on Hugging Face is gated. Open https://huggingface.co/datasets/{repo_id}, "
            "agree to the conditions (one click), then log in once with `hf auth login`.")


def _not_found():
    """Exception types for 'file not in repo', across huggingface_hub versions."""
    from huggingface_hub import errors
    return tuple(getattr(errors, n) for n in ("EntryNotFoundError", "RemoteEntryNotFoundError")
                 if hasattr(errors, n))


def list_scenes(format="yolo", revision=None, repo_id=None):
    """Lists the scenes published on the Hugging Face Hub.

    Args:
        format: Annotation format (default ``"yolo"``).
        revision: Branch, tag or commit.
        repo_id: Dataset repository; defaults to ``Linpen/EnvoDat``.

    Returns:
        Sorted scene ids, e.g. ``["leo-for", "mu-hall", ...]``.
    """
    hub = _hub()
    repo_id = repo_id or _config.HF_REPO_ID
    try:
        files = hub.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    except hub.errors.GatedRepoError as exc:
        raise PermissionError(_gated_message(repo_id)) from exc
    prefix = format.rstrip("/") + "/"
    return sorted(f[len(prefix):-4] for f in files
                  if f.startswith(prefix) and f.endswith(".zip") and f.count("/") == 1)


def scene_dir(scene, format="yolo"):
    """Local folder of a downloaded scene (it may not exist yet)."""
    return _config.root() / format / _normalise([scene])[0]


def data_yaml(scene):
    """Returns the path of a downloaded scene's Ultralytics ``data.yaml``.

    Raises:
        FileNotFoundError: If the scene has not been downloaded.
    """
    path = scene_dir(scene) / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run envodat.download('{scene}') first.")
    return path


def _localise_yaml(sdir):
    """Give the scene's data.yaml an absolute ``path`` so it works from any directory."""
    src = sdir / "data.yaml"
    if not src.exists():
        return None
    cfg = yaml.safe_load(src.read_text()) or {}
    cfg.pop("path", None)
    cfg = {"path": str(sdir.resolve()), **cfg}
    for split in ("train", "val", "test"):
        if split in cfg and not (sdir / cfg[split]).is_dir():
            cfg.pop(split)
    src.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return src


def _load_config(config):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}
    if "root" in cfg:
        _config.init(cfg.pop("root"))
    if "scenes" in cfg:
        cfg["scene"] = cfg.pop("scenes")
    return cfg


def download(scene=None, format="yolo", revision=None, config=None, repo_id=None, force=False):
    """Downloads annotated scenes from the Hugging Face Hub into the data root.

    Each scene is a single zip, unpacked into ``<root>/yolo/<scene>/`` with a ``data.yaml``
    that uses absolute paths. Complete scenes are skipped; incomplete ones are downloaded again.

    Args:
        scene: Scene id (``"mu-hall"``), list of ids, or ``"all"``.
        format: Annotation format (default ``"yolo"``).
        revision: Branch, tag or commit to pin.
        config: YAML file with ``root``, ``scenes``, ``format`` and ``revision``; replaces
            the other arguments.
        repo_id: Dataset repository; defaults to ``Linpen/EnvoDat``.
        force: Downloads and unpacks again even if the scene is complete.

    Returns:
        A dict mapping each scene id to its local folder.
    """
    if config is not None:
        return download(**_load_config(config), repo_id=repo_id, force=force)
    scenes = _normalise(_as_list(scene))
    if not scenes:
        raise ValueError("Pass scene='mu-hall', a list of scenes, or scene='all'.")
    repo_id = repo_id or _config.HF_REPO_ID
    if scenes == ["all"]:
        scenes = list_scenes(format, revision, repo_id)
    root = _config.root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {', '.join(scenes)} ({format}) from huggingface.co/datasets/{repo_id} "
          f"to {root}", file=sys.stderr)
    import shutil
    import zipfile
    hub = _hub()
    result = {}
    for s in scenes:
        sdir = root / format / s
        if (sdir / COMPLETE_MARKER).exists() and not force:
            result[s] = sdir
            continue
        try:
            zpath = hub.hf_hub_download(repo_id, f"{format}/{s}.zip", repo_type="dataset",
                                        revision=revision)
        except hub.errors.GatedRepoError as exc:
            raise PermissionError(_gated_message(repo_id)) from exc
        except _not_found():
            print(f"WARNING: scene '{s}' is not on the Hub in format '{format}'. "
                  f"See envodat.list_scenes().", file=sys.stderr)
            continue
        if sdir.exists():
            shutil.rmtree(sdir)
        tmp = sdir.with_name(sdir.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        tmp.rename(sdir)
        if format == "yolo":
            _localise_yaml(sdir)
        (sdir / COMPLETE_MARKER).write_text(f"{repo_id} {format}/{s}.zip {revision or 'main'}\n")
        result[s] = sdir
    return result


def load(scene, split=None, repo_id=None, revision=None, **kwargs):
    """Loads a scene as a 🤗 ``datasets`` object.

    The local copy is used when the scene has been downloaded; otherwise the scene is read
    from the Hub. Each example holds ``image``, ``width``, ``height`` and ``objects``
    (COCO-style pixel boxes, class ids and class names).

    Args:
        scene: Scene id, e.g. ``"mu-hall"``.
        split: ``"train"``, ``"validation"`` or ``"test"``; ``None`` gives all splits.
        **kwargs: Passed to :func:`datasets.load_dataset`.

    Returns:
        A ``DatasetDict``, or a ``Dataset`` when ``split`` is given.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install 🤗 datasets:  pip install 'envodat[hf]'") from exc
    sid = _normalise([scene])[0]
    sdir = scene_dir(sid)
    if sdir.is_dir():
        files = {name: str(sdir / d / "images" / "*") for name, d in SPLIT_DIRS.items()
                 if (sdir / d / "images").is_dir()}
        return load_dataset("imagefolder", data_files=files, split=split, **kwargs)
    return load_dataset(repo_id or _config.HF_REPO_ID, sid, split=split, revision=revision,
                        **kwargs)


def class_names(scene):
    """Class names of a downloaded scene, in label-id order."""
    names = yaml.safe_load(data_yaml(scene).read_text())["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def info(scene):
    """Catalog entry of a scene (name, environment, sequences, description)."""
    sid = _normalise([scene])[0]
    if sid not in SCENES:
        raise KeyError(f"Unknown scene '{scene}'. Known: {', '.join(SCENES)}")
    entry = dict(SCENES[sid], id=sid)
    entry["downloaded"] = scene_dir(sid).is_dir()
    return entry


__all__ = ["list_scenes", "download", "load", "data_yaml", "scene_dir", "class_names", "info"]
