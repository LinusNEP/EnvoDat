# Installation

This guide sets up the environment for the EnvoDat development kit: the object-detection demo, the trajectory-evaluation tooling, and the leaderboard scripts.

## 1. System requirements

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| OS        | Ubuntu 20.04 / 22.04 (also works on macOS, Windows) | Ubuntu 22.04 |
| Python    | 3.8     | 3.10+ |
| RAM       | 8 GB    | 16 GB+ |
| GPU       | optional | NVIDIA GPU with CUDA 11.8+ for training |
| Disk      | depends on how many scenes/formats you download | 50 GB+ free |

The detection demo runs CPU-only; a GPU mainly speeds up training. The SLAM evaluation tooling is CPU-only.

## 2. Clone the repository

```bash
git clone https://github.com/LinusNEP/EnvoDat.git
cd EnvoDat
```

## 3. Create a virtual environment

```bash
# If venv is not available:
sudo apt-get update && sudo apt-get install -y python3-venv

python3 -m venv EnvoDatEnv
source EnvoDatEnv/bin/activate    # Windows: EnvoDatEnv\Scripts\activate
python -m pip install --upgrade pip
```

You should now see `(EnvoDatEnv)` at the start of your prompt.

## 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `ultralytics` (YOLOv8–YOLO11), `torch`/`torchvision`, OpenCV, pandas, and matplotlib etc.

> **GPU builds.** `requirements.txt` installs the default PyTorch build. For a specific CUDA build, install torch first from the [official selector](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`. Verify CUDA with:
> ```bash
> python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
> ```

## 5. (Optional) SLAM evaluation

The leaderboard's SLAM track scores trajectories with **ATE/RPE** via [`evo`](https://github.com/MichaelGrupp/evo):

```bash
pip install evo --upgrade --no-binary evo
```

The SLAM pipelines (RTAB-Map, ORB-SLAM3, HDL-SLAM, FAST-LIO2, DLIO) are **not** redistributed here. Install them from their upstream repositories, run them on the EnvoDat sequences to produce estimated trajectories, then score those trajectories with `docs/scripts/evaluate_slam.py`. Most of these are ROS packages; see each project's instructions for ROS 1/2 setup.

## 6. Download the dataset

Use the helper script below to download the data to `./data`:

```bash
python docs/scripts/download_envodat.py --scene mu-hall --format yolo --out ./data
```

Or download manually from the [EnvoDat download page](https://sites.google.com/view/envodat/download) and place the files under `./data`.

## 7. Verify the installation

```bash
# Detection stack
python -c "import ultralytics, torch, cv2, pandas, matplotlib; print('detection deps OK')"

# Trajectory eval stack (only if you installed evo)
python -c "import evo; print('evo OK')"
```

If both print OK, you're ready. Continue with the object-detection walkthrough in [`GET_STARTED.md`](GET_STARTED.md), or the leaderboard in [`BENCHMARK.md`](BENCHMARK.md).

## Troubleshooting

- **`yolo: command not found`** — the `ultralytics` CLI lives in the venv; make sure it's activated (`source EnvoDatEnv/bin/activate`).
- **`results.csv` columns not found in `metrics_eval.py`** — your Ultralytics version writes a different CSV schema. `requirements.txt` pins a compatible range; reinstall with `pip install -r requirements.txt`.
