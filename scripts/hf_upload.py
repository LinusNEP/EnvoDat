#!/usr/bin/env python
"""
[Maintainers] Publish the annotated EnvoDat scenes to the Hugging Face Hub.

The script discovers every scene under --src that has a YOLO export, i.e. a folder
containing train/images (plus labels, and valid/ and test/ if present) and a YAML
with the class names. It works with both a flat layout and the hierarchical
layout of the release:

    src/                             src/
    ├── mu-hall/                     ├── Indoors/
    │   ├── train/images, labels     │   └── mu-hall/
    │   ├── valid/...                │       ├── train/images, labels
    │   ├── test/...                 │       └── envodata-mu-hall.yaml
    │   └── envodata-mu-hall.yaml    └── SubTunnels/ ...
    └── ...

It then stages a Hub-ready tree with a few large files per scene (the Hub limits
free accounts to 1000 requests per 5 minutes, so one file per image would lock
users out):

    yolo/<scene>.zip                            -> Ultralytics layout + data.yaml (envodat.download)
    data/<scene>/<split>-0000N-of-0000M.parquet -> 🤗 datasets / data viewer: load_dataset(repo, "<scene>")
    README.md                                   -> dataset card with one config per scene

The unpacked YOLO folders stay in the staging folder (yolo/<scene>/) for checking,
but are not uploaded.

Usage:
    hf auth login                               # once, with a write token
    python scripts/hf_upload.py --src /path/to/EnvoDat-yolo --stage-only   # inspect first
    python scripts/hf_upload.py --src /path/to/EnvoDat-yolo                # stage + upload (private)
    python scripts/hf_upload.py --src ... --extra coco=/path/to/coco-export # other formats as-is
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install pyyaml pillow huggingface_hub")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envodat.catalog import SCENES, match_scene  # noqa: E402

HF_REPO_ID = os.environ.get("ENVODAT_HF_REPO", "Linpen/EnvoDat")
CARD_TEMPLATE = Path(__file__).resolve().parent.parent / "hf" / "dataset_card_template.md"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# source split folder -> (staged folder, data.yaml key, 🤗 split name)
SPLITS = {"train": ("train", "train", "train"),
          "valid": ("valid", "val", "validation"),
          "val": ("valid", "val", "validation"),
          "test": ("test", "test", "test")}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def find_scenes(src):
    """Yield (scene_id, environment, scene_dir) for every folder containing train/images.

    The scene id comes from the closest path component that names a known EnvoDat
    scene (e.g. ``Indoors/MU-Hall/annotations/yolov8`` -> ``mu-hall``); otherwise the
    folder name is used. Duplicates get a numeric suffix and a warning.
    """
    seen = set()
    for train_imgs in sorted(Path(src).rglob("train/images")):
        scene_dir = train_imgs.parent.parent
        rel = scene_dir.relative_to(src).parts or (Path(src).resolve().name,)
        scene = next((m for part in reversed(rel) for m in [match_scene(part)] if m),
                     slug(scene_dir.name))
        if scene in seen:
            n = 2
            while f"{scene}-{n}" in seen:
                n += 1
            print(f"  WARNING: second YOLO export for '{scene}' at {scene_dir}; "
                  f"staging it as '{scene}-{n}'. Check before uploading.")
            scene = f"{scene}-{n}"
        seen.add(scene)
        env = SCENES.get(scene, {}).get("environment", "")
        yield scene, env, scene_dir


def read_names(scene_dir):
    """Class names from the scene's YOLO YAML (list or {id: name} dict)."""
    candidates = sorted(scene_dir.glob("*.yaml")) + sorted(scene_dir.glob("*.yml"))
    for y in candidates:
        cfg = yaml.safe_load(y.read_text()) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list):
            return names
    sys.exit(f"No YAML with 'names' found in {scene_dir}")


def yolo_to_coco_boxes(label_file, w, h):
    """YOLO txt (boxes or polygons, normalised) -> COCO-style pixel [x, y, w, h]."""
    bboxes, cats = [], []
    if not label_file.exists():
        return bboxes, cats
    for line in label_file.read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        c, coords = int(float(vals[0])), [float(v) for v in vals[1:]]
        if len(coords) == 4:                       # cx cy bw bh
            cx, cy, bw, bh = coords
            x0, y0 = cx - bw / 2, cy - bh / 2
        else:                                      # polygon x1 y1 x2 y2 ...
            xs, ys = coords[0::2], coords[1::2]
            x0, y0, bw, bh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        bboxes.append([round(x0 * w, 2), round(y0 * h, 2), round(bw * w, 2), round(bh * h, 2)])
        cats.append(c)
    return bboxes, cats


def place(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(Path(src).resolve())


def normalise_label(text):
    """Return label text with box rows turned into 4-corner polygons if the file mixes
    boxes and polygons, else None. Ultralytics drops any image whose label file mixes
    the two ("labels mix segment and detection rows"); the polygon form keeps every object
    and still trains detectors (boxes are taken from the polygons)."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    kinds = {len(r) == 5 for r in rows if len(r) >= 5}
    if kinds != {True, False}:
        return None
    out = []
    for r in rows:
        if len(r) == 5:
            c, (cx, cy, w, h) = r[0], map(float, r[1:])
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            r = [c] + [f"{v:.6f}" for v in (x0, y0, x1, y0, x1, y1, x0, y1)]
        out.append(" ".join(r))
    return "\n".join(out) + "\n"


def stage_scene(scene, env, scene_dir, out, copy):
    names = read_names(scene_dir)
    dst_root = out / "yolo" / scene
    counts, cfg = {}, {"train": None, "val": None, "test": None}
    for src_split, (dst_split, yaml_key, hf_split) in SPLITS.items():
        img_dir = scene_dir / src_split / "images"
        if not img_dir.is_dir() or hf_split in counts:
            continue
        lbl_dir = scene_dir / src_split / "labels"
        dst_img, dst_lbl = dst_root / dst_split / "images", dst_root / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        n = fixed = 0
        with open(dst_img / "metadata.jsonl", "w") as meta:
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                lbl = lbl_dir / (img.stem + ".txt")
                place(img, dst_img / img.name, copy)
                if lbl.exists():
                    norm = normalise_label(lbl.read_text())
                    if norm is None:
                        place(lbl, dst_lbl / lbl.name, copy)
                    else:
                        target = dst_lbl / lbl.name
                        if target.exists() or target.is_symlink():
                            target.unlink()
                        target.write_text(norm)
                        fixed += 1
                with Image.open(img) as im:
                    w, h = im.size
                bboxes, cats = yolo_to_coco_boxes(lbl, w, h)
                meta.write(json.dumps({
                    "file_name": img.name, "scene": scene, "width": w, "height": h,
                    "objects": {"bbox": bboxes, "category": cats,
                                "category_name": [names[c] if c < len(names) else str(c)
                                                  for c in cats]},
                }) + "\n")
                n += 1
        counts[hf_split] = n
        cfg[yaml_key] = f"{dst_split}/images"
        if fixed:
            print(f"  {scene}/{dst_split}: {fixed} label file(s) mixed boxes and polygons; "
                  "boxes written as polygons so no image is skipped")
    data_yaml = {**{k: v for k, v in cfg.items() if v}, "nc": len(names), "names": names}
    (dst_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return {"scene": scene, "env": env, "classes": len(names), "names": names, "counts": counts}


PARQUET_SHARD_BYTES = 400 * 1024 ** 2   # ~400 MB of images per Parquet file
HF_SPLITS = (("train", "train"), ("validation", "valid"), ("test", "test"))


def pack_scene(info, out):
    """Write yolo/<scene>.zip and data/<scene>/<split>-*.parquet from a staged scene."""
    import zipfile
    try:
        from datasets import ClassLabel, Dataset, Features, Image as HFImage, Sequence, Value
    except ImportError:
        sys.exit("Missing dependency. Install with:  pip install datasets")
    scene, names = info["scene"], info["names"]
    sdir = out / "yolo" / scene

    zpath = out / "yolo" / f"{scene}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # jpg/png are already compressed
        for f in sorted(sdir.rglob("*")):
            if f.is_file() or f.is_symlink():
                z.write(f, f.relative_to(sdir).as_posix())

    features = Features({
        "image": HFImage(),
        "file_name": Value("string"),
        "scene": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence({
            "bbox": Sequence(Value("float32"), length=4),
            "category": ClassLabel(names=[str(n) for n in names]),
            "category_name": Value("string"),
        }),
    })
    ddir = out / "data" / scene
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True)
    for hf_split, folder in HF_SPLITS:
        meta = sdir / folder / "images" / "metadata.jsonl"
        if not meta.exists():
            continue
        rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
        img_dir = meta.parent

        def gen(rows=rows, img_dir=img_dir):
            for r in rows:
                o = r["objects"]
                yield {"image": {"bytes": (img_dir / r["file_name"]).read_bytes(),
                                 "path": r["file_name"]},
                       "file_name": r["file_name"], "scene": r["scene"],
                       "width": r["width"], "height": r["height"],
                       "objects": {"bbox": o["bbox"], "category": o["category"],
                                   "category_name": o["category_name"]}}

        ds = Dataset.from_generator(gen, features=features)
        size = sum((img_dir / r["file_name"]).stat().st_size for r in rows)
        n = max(1, -(-size // PARQUET_SHARD_BYTES))
        for i in range(n):
            ds.shard(n, i, contiguous=True).to_parquet(
                str(ddir / f"{hf_split}-{i:05d}-of-{n:05d}.parquet"))
    return zpath


def render_card(scenes, repo_id):
    configs = ["configs:"]
    rows = ["| Scene (config) | Environment | Train | Validation | Test | Classes |",
            "|---|---|---:|---:|---:|---:|"]
    for s in scenes:
        configs += [f"- config_name: {s['scene']}", "  data_files:"]
        for hf_split, dst in (("train", "train"), ("validation", "valid"), ("test", "test")):
            if hf_split in s["counts"]:
                configs += [f"  - split: {hf_split}", f"    path: data/{s['scene']}/{hf_split}-*.parquet"]
        c = s["counts"]
        rows.append(f"| `{s['scene']}` | {s['env'] or '–'} | {c.get('train', 0)} | "
                    f"{c.get('validation', 0)} | {c.get('test', 0)} | {s['classes']} |")
    card = CARD_TEMPLATE.read_text()
    return (card.replace("__CONFIGS__", "\n".join(configs))
                .replace("__SCENES_TABLE__", "\n".join(rows))
                .replace("__REPO_ID__", repo_id)
                .replace("__FIRST_SCENE__", scenes[0]["scene"]))


def ensure_repo(api, repo_id, public=False):
    """Create the dataset repo (private unless ``public``) and refuse to upload into an
    existing public repo unless ``public`` was asked for explicitly."""
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    is_private = api.repo_info(repo_id, repo_type="dataset").private
    if not is_private and not public:
        sys.exit(f"{repo_id} already exists and is PUBLIC. Nothing was uploaded. Rerun with "
                 "--public to upload into it anyway, or make it private in its settings first.")
    print(f"Uploading to {'PUBLIC' if not is_private else 'private'} repo "
          f"https://huggingface.co/datasets/{repo_id}")


def publish(src, out="./hf_staging", repo_id=HF_REPO_ID, extra=(), copy=False, public=False,
            stage_only=False):
    """Stage every YOLO scene under ``src`` (plus ``extra`` format trees) and upload them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = []
    for scene, env, scene_dir in find_scenes(src):
        info = stage_scene(scene, env, scene_dir, out, copy)
        pack_scene(info, out)
        print(f"  staged {scene:<20} {info['counts']}  classes={info['classes']}")
        scenes.append(info)
    if not scenes:
        sys.exit(f"No scenes with train/images found under {src}")

    import zipfile
    for spec in extra:
        fmt, _, path = spec.partition("=")
        (out / "extra").mkdir(exist_ok=True)
        with zipfile.ZipFile(out / "extra" / f"{fmt}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(path).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(path).as_posix())
        print(f"  packed extra format '{fmt}' from {path} into extra/{fmt}.zip")

    (out / "README.md").write_text(render_card(scenes, repo_id))
    print(f"\nStaged {len(scenes)} scene(s) in {out.resolve()}")
    if stage_only:
        print("Stage-only: nothing uploaded. Inspect the folder, then rerun without --stage-only.")
        return scenes

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Missing dependency. Install with:  pip install 'huggingface_hub>=0.25'")
    api = HfApi()
    ensure_repo(api, repo_id, public)
    # Upload only the packed files; delete any loose per-image files from an earlier layout.
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(out),
        allow_patterns=["README.md", "yolo/*.zip", "data/*/*.parquet", "extra/*.zip"],
        delete_patterns=["yolo/*/*"],
        commit_message="Upload packed EnvoDat scenes (zip + Parquet)",
    )
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")
    return scenes


def main():
    ap = argparse.ArgumentParser(description="Stage and upload EnvoDat to the Hugging Face Hub.")
    ap.add_argument("--src", required=True, help="Root folder of the YOLO exports")
    ap.add_argument("--out", default="./hf_staging", help="Staging folder")
    ap.add_argument("--repo-id", default=HF_REPO_ID)
    ap.add_argument("--extra", action="append", default=[], metavar="FORMAT=PATH",
                    help="Upload another format tree as-is under FORMAT/, e.g. coco=/data/coco")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument("--public", action="store_true",
                    help="Create/upload to a PUBLIC repo (default: private, only you can see it)")
    ap.add_argument("--stage-only", action="store_true", help="Stage locally, do not upload")
    args = ap.parse_args()
    publish(args.src, args.out, args.repo_id, args.extra, args.copy, args.public,
            args.stage_only)


if __name__ == "__main__":
    main()
