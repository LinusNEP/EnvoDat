"""Training helpers."""

from .hub import data_yaml, download, scene_dir


def train_yolo(scene, model="yolo11n.pt", epochs=100, imgsz=640, batch=16, download_missing=True,
               evaluate_test=True, **kwargs):
    """Train an Ultralytics YOLO detector (YOLOv8-YOLO11) on a scene.

    The scene is downloaded first if it is not present locally.

    Args:
        scene: Scene id, e.g. ``"mu-hall"``.
        model: Pretrained weights or model config, e.g. ``"yolo11n.pt"``, ``"yolov8s.pt"``.
        epochs, imgsz, batch: Main training settings.
        download_missing: Fetch the scene from the Hub when it is not downloaded yet.
        evaluate_test: Also validate on the test split after training.
        **kwargs: Any other Ultralytics train argument (``lr0``, ``device``, ``patience``, ...).

    Returns:
        Dict with the trained ``model`` and the ``val`` (and ``test``) metrics objects.
    """
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
