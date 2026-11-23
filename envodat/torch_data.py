"""PyTorch access to downloaded scenes (YOLO layout), without extra dependencies."""


from .hub import SPLIT_DIRS, class_names, scene_dir

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyTorch:  pip install 'envodat[torch]'") from exc
    return torch


class EnvoDatDetection:
    """Map-style detection dataset over one split of a downloaded scene.

    Each item is ``(image, target)``: ``image`` is a float tensor ``[3, H, W]`` in
    ``[0, 1]`` (resized to ``image_size`` if given), and ``target`` is a dict with
    ``boxes`` (``[N, 4]``, normalised ``cx, cy, w, h``), ``labels`` (``[N]``),
    ``image_id`` and ``path``. Polygon labels are converted to their bounding boxes.

    Args:
        scene: Scene id, e.g. ``"mu-hall"``.
        split: ``"train"``, ``"validation"`` (or ``"val"``) or ``"test"``.
        image_size: Int (square) or ``(h, w)`` to resize to; ``None`` keeps the original.
        transform: Optional callable ``(image, target) -> (image, target)``.
    """

    def __init__(self, scene, split="train", image_size=640, transform=None):
        split = {"val": "validation", "valid": "validation"}.get(split, split)
        self.dir = scene_dir(scene) / SPLIT_DIRS[split]
        if not (self.dir / "images").is_dir():
            raise FileNotFoundError(f"{self.dir}/images not found. "
                                    f"Run envodat.download('{scene}') first.")
        self.images = sorted(p for p in (self.dir / "images").iterdir()
                             if p.suffix.lower() in IMG_EXT)
        self.classes = class_names(scene)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _labels(self, img_path):
        lbl = self.dir / "labels" / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                v = line.split()
                if len(v) < 5:
                    continue
                c, xy = int(float(v[0])), [float(x) for x in v[1:]]
                if len(xy) == 4:
                    boxes.append(xy)
                else:
                    xs, ys = xy[0::2], xy[1::2]
                    boxes.append([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                                  max(xs) - min(xs), max(ys) - min(ys)])
                labels.append(c)
        return boxes, labels

    def __getitem__(self, i):
        torch = _torch()
        from PIL import Image
        path = self.images[i]
        img = Image.open(path).convert("RGB")
        if self.image_size:
            img = img.resize((self.image_size[1], self.image_size[0]))
        w, h = img.size
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        x = x.view(h, w, 3).permute(2, 0, 1).float().div(255)
        boxes, labels = self._labels(path)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": i, "path": str(path)}
        if self.transform:
            x, target = self.transform(x, target)
        return x, target


def collate(batch):
    """Stack images (same size) and keep targets as a list."""
    torch = _torch()
    images, targets = zip(*batch)
    try:
        images = torch.stack(images)
    except RuntimeError:
        images = list(images)  # variable sizes (image_size=None)
    return images, list(targets)


def dataloader(scene, split="train", batch_size=8, image_size=640, shuffle=None,
               num_workers=0, transform=None, **kwargs):
    """Return a ``torch.utils.data.DataLoader`` over a downloaded scene.

    Args:
        scene: Scene id, e.g. ``"mu-hall"``.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        batch_size: Images per batch.
        image_size: Resize target (int or ``(h, w)``); ``None`` keeps originals.
        shuffle: Defaults to ``True`` for the train split, else ``False``.
        num_workers: Worker processes for loading.
        transform: Optional ``(image, target) -> (image, target)`` callable.
        **kwargs: Passed to ``DataLoader``.

    Yields:
        ``(images, targets)`` where ``images`` is ``[B, 3, H, W]`` and ``targets`` a list of
        dicts as described in :class:`EnvoDatDetection`.
    """
    torch = _torch()
    ds = EnvoDatDetection(scene, split, image_size, transform)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers, collate_fn=collate, **kwargs)


__all__ = ["EnvoDatDetection", "dataloader", "collate"]
