# Hugging Face

The annotated, training-ready EnvoDat scenes are mirrored on the Hugging Face Hub at **[huggingface.co/datasets/Linpen/EnvoDat](https://huggingface.co/datasets/Linpen/EnvoDat)**. Every scene is published in YOLO layout and as a 🤗 `datasets` config, so you can go from zero to a training run without reorganising folders by hand. The raw multisensor data stays on the CPS server for now and is fetched with {func}`envodat.download_raw`.

## What is on the Hub

Each scene (e.g. `mu-hall`) is stored as follows:

```
yolo/<scene>/data.yaml                     # Ultralytics config
yolo/<scene>/{train,valid,test}/images/    # images + metadata.jsonl
yolo/<scene>/{train,valid,test}/labels/    # YOLO txt labels
```

The same files therefore serve two workflows. On the one hand, Ultralytics reads `data.yaml` and the `labels/` folders directly. On the other hand, the `metadata.jsonl` next to the images lets `load_dataset` return images with COCO-style boxes (`[x_min, y_min, width, height]` in pixels), class ids (`category`) and class names (`category_name`). Each scene is a separate config because the class vocabularies differ between environments, and the dataset card lists every scene with its split sizes. The annotations follow the 70 / 20 / 10 train / validation / test split of the release.

## Download

With the toolkit, downloading is a single call, and it also rewrites each `data.yaml` with absolute paths so training works from any directory:

```python
import envodat as ed
ed.download("mu-hall")
```

Without the toolkit, use the Hugging Face CLI or Python API directly:

```bash
hf download Linpen/EnvoDat --repo-type dataset --include "yolo/mu-hall/*" --local-dir ./data
```

```python
from huggingface_hub import snapshot_download

snapshot_download("Linpen/EnvoDat", repo_type="dataset",
                  allow_patterns="yolo/mu-hall/*", local_dir="./data")
```

## Load with 🤗 datasets

```python
from datasets import load_dataset

ds = load_dataset("Linpen/EnvoDat", "mu-hall")
sample = ds["train"][0]
sample["image"]                      # PIL image
sample["objects"]["bbox"]            # [[x_min, y_min, w, h], ...] in pixels
sample["objects"]["category_name"]   # class names
```

{func}`envodat.load` does the same, but reads the local copy when the scene has already been downloaded. For training with these datasets, see the [Examples](examples.md#training-example).

## Versions

The Hub repository is versioned with git, and releases are tagged. To keep results reproducible, pin a revision when you download (`ed.download("mu-hall", revision="v1.0")`, `envodat download mu-hall --revision v1.0`, or `revision=` in `load_dataset`) and report it with your results.
