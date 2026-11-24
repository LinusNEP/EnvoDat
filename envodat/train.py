"""Training helpers."""

from .hub import data_yaml, download, scene_dir


def train_yolo(scene, model="yolo11n.pt", epochs=100, imgsz=640, batch=16, download_missing=True,
               evaluate_test=True, **kwargs):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install Ultralytics:  pip install 'envodat[train]'") from exc
    if download_missing and not (scene_dir(scene) / "data.yaml").exists():
        download(scene)
    cfg = str(data_yaml(scene))
    yolo = YOLO(model)
    yolo.train(data=cfg, epochs=epochs, imgsz=imgsz, batch=batch, **kwargs)
    out = {"model": yolo, "val": yolo.val(data=cfg, split="val")}
    if evaluate_test:
        out["test"] = yolo.val(data=cfg, split="test")
    return out
