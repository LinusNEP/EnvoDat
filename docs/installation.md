# Installation

The `envodat` toolkit downloads the data, loads it for training, trains detectors and evaluates SLAM trajectories. Its core needs only a few small packages, while the heavier ones (PyTorch, Ultralytics, 🤗 `datasets`, `evo`) are optional extras that are imported only when you use the matching feature.

## Requirements

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| OS        | Ubuntu 20.04 / 22.04 (macOS and Windows also work) | Ubuntu 22.04 |
| Python    | 3.8     | 3.10+ |
| RAM       | 8 GB    | 16 GB+ |
| GPU       | optional | NVIDIA GPU with CUDA 11.8+ for training |
| Disk      | depends on the scenes and formats you download | 50 GB+ free |

## Install with pip

```bash
python3 -m venv EnvoDatEnv && source EnvoDatEnv/bin/activate   # Windows: EnvoDatEnv\Scripts\activate
pip install envodat
```

Then add the extras for what you plan to do:

| Extra | Adds | Needed for |
| ----- | ---- | ---------- |
| `envodat[hf]` | 🤗 `datasets` | {func}`envodat.load` |
| `envodat[torch]` | PyTorch | {func}`envodat.dataloader` |
| `envodat[train]` | Ultralytics (YOLOv8–YOLO11) | {func}`envodat.train_yolo` |
| `envodat[slam]` | `evo` | {func}`envodat.evaluate_traj` |
| `envodat[all]` | all of the above | everything |

```bash
pip install "envodat[all]"
```

```{note}
**GPU builds.** pip installs the default PyTorch build. For a specific CUDA version, install `torch` first from the [official selector](https://pytorch.org/get-started/locally/), then install `envodat`. Check with `python -c "import torch; print(torch.cuda.is_available())"`.
```

## Install from source

To work on the toolkit itself, or to use the benchmark scripts in `scripts/`, clone the repository and install it in editable mode:

```bash
git clone https://github.com/LinusNEP/EnvoDat.git
cd EnvoDat
pip install -e ".[all]"
```

Until the first PyPI release is out, you can also install the latest version straight from GitHub with `pip install "envodat[all] @ git+https://github.com/LinusNEP/EnvoDat"`.

## SLAM pipelines

The SLAM track of the [benchmark](benchmark.md) compares RTAB-Map, ORB-SLAM3, HDL-SLAM, FAST-LIO2 and DLIO. These pipelines are **not** redistributed here, so install them from their upstream repositories (most are ROS packages), run them on the EnvoDat sequences, and evaluate the estimated trajectories with {func}`envodat.evaluate_traj` or `envodat eval`.

## Get access to the data

The annotated scenes on Hugging Face are gated: open the [dataset page](https://huggingface.co/datasets/Linpen/EnvoDat), agree to the conditions (one click), then log in once:

```bash
hf auth login
```

## Verify the installation

```bash
envodat scenes                          # prints the scene catalog
python -c "import envodat; print(envodat.__version__)"
```

If both commands work, continue with the [Examples](examples.md).
