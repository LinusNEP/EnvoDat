"""Command-line interface: ``envodat <command>``."""

import argparse
import sys

from . import _config, catalog, cps, hub, slam, train


def _scenes(args):
    print(f"{'id':<13} {'name':<12} {'environment':<13} sequences  description")
    for sid, s in catalog.SCENES.items():
        print(f"{sid:<13} {s['name']:<12} {s['environment']:<13} {','.join(s['sequences']):<10} "
              f"{s['description']}")


def _list(args):
    for s in hub.list_scenes(args.format, args.revision):
        print(s)


def _download(args):
    if args.config:
        res = hub.download(config=args.config)
    else:
        res = hub.download("all" if args.all else args.scene, args.format,
                           args.revision)
    for s, d in res.items():
        print(f"{s}\t{d}")


def _raw(args):
    files = cps.download_raw(args.include, args.exclude, args.out, dry_run=args.dry_run)
    if args.dry_run:
        for f in files:
            print(f"{f.size / 1e9:10.3f} GB  {f.path}")


def _train(args):
    extra = dict(kv.split("=", 1) for kv in args.extra)
    for k, v in extra.items():
        try:
            extra[k] = float(v) if "." in v else int(v)
        except ValueError:
            pass
    train.train_yolo(args.scene, args.model, args.epochs, args.imgsz, args.batch, **extra)


def _eval(args):
    r = slam.evaluate_traj(args.est, args.gt, args.align, args.delta)
    print(f"ATE (mean/std): {r['ate'][0]} / {r['ate'][1]} m")
    print(f"RPE (mean/std): {r['rpe'][0]} / {r['rpe'][1]} m")
    if r["sd"]:
        print(f"SD  (mean/std): {r['sd'][0]} / {r['sd'][1]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="envodat", description="EnvoDat dataset toolkit")
    ap.add_argument("--root", help="Data root (default ./envodat_data or $ENVODAT_ROOT)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scenes", help="Show the scene catalog").set_defaults(fn=_scenes)

    p = sub.add_parser("list", help="List scenes available on the Hugging Face Hub")
    p.add_argument("--format", default="yolo")
    p.add_argument("--revision")
    p.set_defaults(fn=_list)

    p = sub.add_parser("download", help="Download annotated scenes from the Hub")
    p.add_argument("scene", nargs="*", help="Scene ids, e.g. mu-hall leo-str")
    p.add_argument("--all", action="store_true", help="Download every scene")
    p.add_argument("--format", default="yolo")
    p.add_argument("--revision", help="Branch, tag or commit to pin")
    p.add_argument("--config", help="YAML file with root/scenes/format/revision")
    p.set_defaults(fn=_download)

    p = sub.add_parser("raw", help="Download raw data (bags, LiDAR, RGB-D, IMU, poses) "
                                   "from the CPS server")
    p.add_argument("include", nargs="+", help="Glob(s) on the share path, e.g. '*mu-hall*.bag'")
    p.add_argument("--exclude", nargs="*")
    p.add_argument("--out", help="Target folder (default <root>/raw)")
    p.add_argument("--dry-run", action="store_true", help="Only list matches and sizes")
    p.set_defaults(fn=_raw)

    p = sub.add_parser("train", help="Train a YOLO detector on a scene",
                       epilog="Extra Ultralytics args as key=value, e.g. lr0=0.01 device=0")
    p.add_argument("scene")
    p.add_argument("extra", nargs="*", metavar="key=value")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="ATE/RPE/scale drift of a trajectory vs ground truth")
    p.add_argument("est", help="Estimated trajectory (TUM)")
    p.add_argument("gt", help="Ground truth (gt-poses-*.csv or TUM)")
    p.add_argument("--align", choices=["none", "se3", "sim3"], default="se3")
    p.add_argument("--delta", type=int, default=1)
    p.set_defaults(fn=_eval)

    args = ap.parse_args(argv)
    if args.root:
        _config.init(args.root)
    if args.cmd == "download" and not (args.scene or args.all or args.config):
        ap.error("download: give scene ids, --all or --config")
    try:
        args.fn(args)
    except (ImportError, FileNotFoundError, ConnectionError, ValueError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
