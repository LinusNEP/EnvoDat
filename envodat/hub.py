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


def _not_found():
    """Exception types for 'file not in repo', across huggingface_hub versions."""
    from huggingface_hub import errors
    return tuple(getattr(errors, n) for n in ("EntryNotFoundError", "RemoteEntryNotFoundError")
                 if hasattr(errors, n))


def list_scenes(format="yolo", revision=None, repo_id=None):
    """List the scenes published on the Hugging Face Hub.

    Args:
        format: Annotation format folder on the Hub (default ``"yolo"``).
        revision: Branch, tag or commit to query.
        repo_id: Override the dataset repo (default ``Linpen/EnvoDat``).

    Returns:
        Sorted list of scene ids, e.g. ``["leo-for", "mu-hall", ...]``.
    """
    files = _hub().list_repo_files(repo_id or _config.HF_REPO_ID, repo_type="dataset",
                                   revision=revision)
    prefix = format.rstrip("/") + "/"
    return sorted(f[len(prefix):-4] for f in files
                  if f.startswith(prefix) and f.endswith(".zip") and f.count("/") == 1)


def scene_dir(scene, format="yolo"):
    """Local folder of a downloaded scene (it may not exist yet)."""
    return _config.root() / format / _normalise([scene])[0]


def data_yaml(scene):
    """Path to the ready-to-use Ultralytics ``data.yaml`` of a downloaded scene.

    Raises:
        FileNotFoundError: if the scene has not been downloaded with :func:`download`.
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
    """Download annotated scenes from the Hugging Face Hub into the data root.

    Each scene is one zip on the Hub (``yolo/<scene>.zip``), so a download is a single
    request. It is cached and unpacked into ``<root>/yolo/<scene>/``, and its YOLO
    ``data.yaml`` is rewritten with an absolute path so it can be passed straight to
    Ultralytics. Scenes that are already completely unpacked are skipped unless ``force``;
    a folder left by an interrupted download is replaced.

    Args:
        scene: A scene id (``"mu-hall"``), a list of ids, or ``"all"``. Names such as
            ``"MU-Hall"`` are accepted too.
        format: Annotation format folder on the Hub (default ``"yolo"``).
        revision: Branch, tag or commit to pin (recommended for papers).
        config: Path to a YAML file with any of the keys ``root``, ``scenes``,
            ``format`` and ``revision``. Keyword arguments are ignored when given.
        repo_id: Override the dataset repo (default ``Linpen/EnvoDat``).
        force: Unpack again even if the scene folder already exists.

    Returns:
        Dict mapping each downloaded scene id to its local folder.
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
    """Load a scene as a 🤗 ``datasets`` object.

    Uses the local copy when the scene has been downloaded, and otherwise streams
    it from the Hub. Each example has ``image``, ``width``, ``height`` and
    ``objects`` with COCO-style pixel boxes ``[x_min, y_min, w, h]``, class ids
    (``category``) and class names (``category_name``).

    Args:
        scene: Scene id, e.g. ``"mu-hall"``.
        split: ``"train"``, ``"validation"`` or ``"test"``; ``None`` returns all splits.
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
