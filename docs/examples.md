# Examples

All examples use the toolkit, imported as `ed`. Start by telling it where the data should live; everything below is read from and written to that folder.

```python
import envodat as ed

ed.init("/data/envodat")      # or set the ENVODAT_ROOT environment variable
```

## Download Example

### Download via Python API

```{note}
The dataset is **gated**: open [its Hugging Face page](https://huggingface.co/datasets/Linpen/EnvoDat), agree to the conditions (one click), and log in once with `hf auth login` (a token with read access from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Downloads and `load_dataset` then work as shown below.
```

The annotated scenes are fetched from the [Hugging Face Hub](huggingface.md). Only the scenes you ask for are downloaded, and downloads are cached and resumable.

```python
ed.list_scenes()                        # scenes published on the Hub
ed.download("mu-hall")                  # one scene
ed.download(["mu-hall", "leo-str"])     # several scenes
ed.download("all")                      # everything
ed.download("mu-hall", revision="v1.0") # pin a release, e.g. for a paper
```

Each scene lands in `<root>/yolo/<scene>/`, with images, YOLO labels and a ready-to-use `data.yaml`:

```python
ed.data_yaml("mu-hall")     # .../yolo/mu-hall/data.yaml (absolute paths inside)
ed.class_names("mu-hall")   # class names in label-id order
ed.info("mu-hall")          # catalog entry: name, environment, sequences, description
```

### Download via a yaml config file

For reproducible setups, describe the download in a YAML file:

```yaml
# download_config.yaml
root: /data/envodat
scenes: [mu-hall, mu-cor, subt-zab-01]
format: yolo
revision: main
```

```python
ed.download(config="download_config.yaml")
```

### Download from the command line

```bash
envodat list                                   # scenes on the Hub
envodat --root /data/envodat download mu-hall leo-str
envodat download --config download_config.yaml
```

### Download raw data

The raw ROS bags, LiDAR point clouds and intensity images, RGB-D frames, IMU data and ground-truth poses are served from the CPS server. Select them with glob patterns on their path; a dry run lists the matching files and their total size first.

```python
ed.download_raw("*mu-hall*seq01*.bag", dry_run=True)   # what would be downloaded
ed.download_raw("*mu-hall*seq01*.bag")                 # -> <root>/raw/...
ed.download_raw(["*leo-str*gt-poses*", "*leo-str*imu-*"])
```

```bash
envodat raw "*mu-hall*seq01*.bag" --dry-run
```

Interrupted downloads resume where they stopped when you run the same call again.

## Dataloader Example

{func}`envodat.dataloader` returns a standard PyTorch `DataLoader` over a downloaded scene. Images come as `[B, 3, H, W]` float tensors, and each target holds normalised `cx, cy, w, h` boxes and class ids.

```python
loader = ed.dataloader("mu-hall", split="train", batch_size=8, image_size=640, num_workers=4)

for images, targets in loader:
    boxes, labels = targets[0]["boxes"], targets[0]["labels"]
    ...
```

Alternatively, load a scene as a 🤗 `datasets` object, which gives COCO-style pixel boxes and works with the whole Hugging Face ecosystem. It uses the local copy if the scene is downloaded, and the Hub otherwise.

```python
ds = ed.load("mu-hall")                 # DatasetDict: train / validation / test
sample = ds["train"][0]
sample["image"], sample["objects"]["bbox"], sample["objects"]["category_name"]
```

## Training Example

### YOLO (Ultralytics)

{func}`envodat.train_yolo` downloads the scene if needed, trains, and then validates on the validation and test splits. Any Ultralytics training argument can be passed through.

```python
result = ed.train_yolo("mu-hall", model="yolo11n.pt", epochs=100, imgsz=640, batch=16,
                       lr0=0.01, patience=50, device=0)
print(result["test"].box.map50, result["test"].box.map)
```

```bash
envodat train mu-hall --model yolo11n.pt --epochs 100 lr0=0.01 device=0
```

Because `data.yaml` is a standard Ultralytics config, the `yolo` CLI works too: `yolo train model=yolo11n.pt data=$(python -c "import envodat as ed; print(ed.data_yaml('mu-hall'))")`. The [Get Started](get_started.md) page walks through hyperparameters, inference on new images and metric plots in detail.

### 🤗 Transformers (RT-DETR, DETR, ...)

Other detector families train from {func}`envodat.load`. The example below fine-tunes RT-DETR; any `AutoModelForObjectDetection` checkpoint follows the same pattern.

```python
import torch
from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                          Trainer, TrainingArguments)

ds = ed.load("mu-hall")
id2label = dict(enumerate(ed.class_names("mu-hall")))
label2id = {v: k for k, v in id2label.items()}

ckpt = "PekingU/rtdetr_r50vd"
processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModelForObjectDetection.from_pretrained(
    ckpt, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

def to_coco(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    targets = [{"image_id": i,
                "annotations": [{"bbox": b, "category_id": c, "area": b[2] * b[3], "iscrowd": 0}
                                for b, c in zip(o["bbox"], o["category"])]}
               for i, o in enumerate(batch["objects"])]
    return processor(images=images, annotations=targets, return_tensors="pt")

def collate(items):
    return {"pixel_values": torch.stack([x["pixel_values"] for x in items]),
            "labels": [x["labels"] for x in items]}

args = TrainingArguments("rtdetr-envodat-mu-hall", per_device_train_batch_size=8,
                         num_train_epochs=50, learning_rate=5e-5,
                         eval_strategy="epoch", save_strategy="epoch",
                         remove_unused_columns=False)
Trainer(model=model, args=args, data_collator=collate,
        train_dataset=ds["train"].with_transform(to_coco),
        eval_dataset=ds["validation"].with_transform(to_coco)).train()
```

## Evaluation Example

### SLAM trajectories

{func}`envodat.evaluate_traj` computes the benchmark metrics (ATE, RPE and scale drift, Eqs. 1–3 of the paper). It reads the EnvoDat ground-truth CSV (`gt-poses-*.csv`) directly, as well as TUM-format files and `[N, 8]` arrays.

```python
gt = ed.download_raw("*mu-hall*gt-poses*")[0]
metrics = ed.evaluate_traj("fastlio2_mu-hall-01.tum", gt, align="se3")
# {'ate': (mean, std), 'rpe': (mean, std), 'sd': (mean, std)}

traj = ed.get_traj_np(gt)    # [N, 8]: t, x, y, z, qx, qy, qz, qw
ed.to_tum(gt)                # writes gt-poses-*.tum for other tools
```

```bash
envodat eval fastlio2_mu-hall-01.tum gt-poses-mu-hall.csv --align sim3
```

To add a result to the leaderboard, follow [Benchmark → How to submit](benchmark.md#how-to-submit).

### Detection models

After training, evaluate on the held-out `test` split (`train_yolo` does this by default) and report mAP@50 and mAP@50:95 together with the Hub revision you trained on, so that results stay comparable as the dataset evolves.
