# Hugging Face

The annotated, training-ready EnvoDat scenes are mirrored on the Hugging Face Hub at **[huggingface.co/datasets/Linpen/EnvoDat](https://huggingface.co/datasets/Linpen/EnvoDat)**. Every scene is published in YOLO layout and as a 🤗 `datasets` config, so you can go from zero to a training run without reorganising folders by hand. The raw multisensor data stays on the CPS server for now and is fetched with {func}`envodat.download_raw`.

## What is on the Hub

Each scene (e.g. `mu-hall`) is packed into a few large files, one set per workflow:

```
data/<scene>/{train,validation,test}-*.parquet   # images + boxes, for 🤗 datasets and the data viewer
yolo/<scene>.zip                                 # Ultralytics layout: images, labels, data.yaml
```

On the one hand, `load_dataset` reads the Parquet files and returns images with COCO-style boxes (`[x_min, y_min, width, height]` in pixels), class ids (`category`, with the class names attached) and class names (`category_name`). On the other hand, the zip unpacks to the standard Ultralytics layout. Packing matters because the Hub rate-limits requests per file, so a scene stored as thousands of single images would lock users out. Each scene is a separate config because the class vocabularies differ between environments, and the dataset card lists every scene with its split sizes. The annotations follow the 70 / 20 / 10 train / validation / test split of the release.

## Download

With the toolkit, downloading is a single call: it fetches the scene's zip, unpacks it, and rewrites its `data.yaml` with absolute paths so training works from any directory:

```python
import envodat as ed
ed.download("mu-hall")
```

Without the toolkit, use the Hugging Face CLI or Python API directly:

```bash
hf download Linpen/EnvoDat yolo/mu-hall.zip --repo-type dataset --local-dir ./data
unzip -q ./data/yolo/mu-hall.zip -d ./data/mu-hall
```

```python
import zipfile
from huggingface_hub import hf_hub_download

path = hf_hub_download("Linpen/EnvoDat", "yolo/mu-hall.zip", repo_type="dataset")
zipfile.ZipFile(path).extractall("./data/mu-hall")
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
