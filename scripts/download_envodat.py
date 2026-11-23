#!/usr/bin/env python
"""
Download EnvoDat (thin wrapper around the `envodat` toolkit; `envodat download` is equivalent).

Sources:
  hf   (default)  Training-ready annotated scenes from the Hugging Face Hub
                  (https://huggingface.co/datasets/LinusNEP/EnvoDat), YOLO layout + data.yaml.
  cps             Raw data (ROS bags, LiDAR, RGB-D, IMU, ground-truth poses) from the
                  CPS server, selected with --include glob patterns.

Usage:
    python download_envodat.py --list                              # scenes on the Hub
    python download_envodat.py --scene mu-hall --out ./data        # one scene (YOLO)
    python download_envodat.py --scene mu-hall --scene leo-str     # several scenes
    python download_envodat.py --all --out ./data                  # every scene
    python download_envodat.py --source cps --include "*mu-hall*seq01*.bag" --dry-run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import envodat as ed  # noqa: E402
from envodat import _config  # noqa: E402


def fetch_hf(scenes, out="./data", fmt="yolo", repo_id=None, revision=None):
    """Download scenes (list of ids or ["all"]) and return {scene: data.yaml or folder}."""
    ed.init(out)
    res = ed.download(scenes[0] if scenes == ["all"] else scenes, fmt, revision, repo_id=repo_id)
    return {s: (d / "data.yaml" if (d / "data.yaml").exists() else d) for s, d in res.items()}


def main():
    ap = argparse.ArgumentParser(description="Download EnvoDat from Hugging Face or the CPS server.")
    ap.add_argument("--source", choices=["hf", "cps"], default="hf")
    ap.add_argument("--scene", action="append", help="Scene id, e.g. mu-hall (repeatable)")
    ap.add_argument("--format", dest="fmt", default="yolo", help="Annotation format on the Hub")
    ap.add_argument("--all", action="store_true", help="Download every scene")
    ap.add_argument("--out", default="./data", help="Output directory")
    ap.add_argument("--list", action="store_true", help="List available scenes and exit")
    ap.add_argument("--repo-id", default=_config.HF_REPO_ID)
    ap.add_argument("--revision", help="Branch, tag or commit on the Hub")
    ap.add_argument("--include", nargs="+", help="[cps] glob(s) on the share path")
    ap.add_argument("--exclude", nargs="*", help="[cps] glob(s) to skip")
    ap.add_argument("--dry-run", action="store_true", help="[cps] only list matches and sizes")
    args = ap.parse_args()

    if args.source == "cps":
        if args.list:
            args.include, args.dry_run = args.include or ["*"], True
        if not args.include:
            ap.error("--source cps needs --include patterns (or --list).")
        files = ed.download_raw(args.include, args.exclude, os.path.join(args.out, "raw"),
                                dry_run=args.dry_run)
        if args.dry_run:
            for f in files:
                print(f"{f.size / 1e9:10.3f} GB  {f.path}")
        return

    if args.list:
        for s in ed.list_scenes(args.fmt, args.revision, args.repo_id):
            print(s)
        return
    if not (args.all or args.scene):
        ap.error("specify --scene (repeatable), --all, or --list.")
    res = fetch_hf(["all"] if args.all else args.scene, args.out, args.fmt, args.repo_id,
                   args.revision)
    for s, p in res.items():
        print(f"  {s}: {p}")
    if args.fmt == "yolo" and res:
        print("\nTrain with e.g.:\n  python scripts/yolov8_training.py --data "
              f"{next(iter(res.values()))}")


if __name__ == "__main__":
    main()
