# Troubleshooting

## Installation

**`envodat: command not found` or `yolo: command not found`.** Both commands are installed into the active environment, so make sure it is activated (`source EnvoDatEnv/bin/activate`).

**`ImportError: Install ... pip install 'envodat[...]'`.** The feature you called needs an optional extra. Install the one named in the message, or everything with `pip install "envodat[all]"`.

**`File "setup.py" not found` when installing from source.** An older pip (often the one a new virtual environment starts with) cannot install `pyproject.toml` projects in editable mode. Upgrading it first solves this: `pip install --upgrade pip setuptools wheel`, then `pip install -e .` again.

**PyTorch does not see the GPU.** pip installs the default PyTorch build. Install `torch` for your CUDA version from the [official selector](https://pytorch.org/get-started/locally/) first, then reinstall `envodat`.

## Downloading from Hugging Face

**`scene 'xyz' is not on the Hub`.** Check the scene id with `envodat list`. Ids are lower case with dashes, for example `mu-hall` and `subt-zab-01`; names such as `MU-Hall` are converted automatically.

**"EnvoDat on Hugging Face is gated", or `401` / `403` errors.** Open the [dataset page](https://huggingface.co/datasets/Linpen/EnvoDat), agree to the conditions while logged in, and log in once on your machine with `hf auth login`. A token with read access is enough.

**`429 Too Many Requests`.** The Hub limits how many requests an account makes in five minutes. Each scene is only a few files, so this should be rare; if it happens, wait five minutes and run the same command again. Logging in with `hf auth login` raises the limit.

**A downloaded scene is incomplete, or Ultralytics reports missing `train/` or `valid/` folders.** A scene that was not unpacked completely, for example after an interrupted download, is downloaded again automatically the next time `envodat download` or `ed.download()` runs. Deleting the scene's folder (`<root>/yolo/<scene>`) or calling `ed.download("<scene>", force=True)` forces a fresh copy.

**Disk space and cache.** Scenes are written to the data root (`ed.init(...)`, `--root` or `ENVODAT_ROOT`). The Hugging Face cache lives in `~/.cache/huggingface` and can be moved with the `HF_HOME` environment variable.

**`load_dataset` complains about the config name.** Every scene is its own config, so pass one: `load_dataset("Linpen/EnvoDat", "mu-hall")`.

## Downloading raw data

**`Cannot reach cloud.cps.unileoben.ac.at`.** Single timeouts and server errors are retried automatically, so this message means the server stayed unreachable. A firewall or proxy on the network may block it; urllib honours the standard `HTTPS_PROXY` variable for institutions that use a proxy. Otherwise, the server may be temporarily down.

**Interrupted downloads.** Run the same `download_raw` call or `envodat raw` command again: partial `.part` files are resumed, and complete files are skipped.

**`Need X GB but only Y GB free`.** Narrow the glob patterns, or check the size first with `dry_run=True` / `--dry-run`.

## Training and evaluation

**Ultralytics skips images with "labels mix segment and detection rows".** This happens with the YOLO exports of the full release, where some label files contain both boxes and polygons. The scenes on Hugging Face are already fixed: boxes are stored as polygons there, so no image is skipped.

**Ultralytics reports "duplicate labels removed".** A few label files list the same object twice. Ultralytics drops the extra copy, so no action is needed.

**`data.yaml not found`.** The scene has not been downloaded into the current data root. Call `ed.download("<scene>")`, or point `ed.init()` at the folder you used before.

**`results.csv` columns not found in `metrics_eval.py`.** Your Ultralytics version writes a different CSV schema. Install a recent version with `pip install -U ultralytics`.

**`Degenerate covariance rank, Umeyama alignment is not possible`.** The trajectories have too few associated poses, or they are almost a straight line. Check that the timestamps of the estimate and the ground truth overlap (both in seconds); you can loosen the association with `max_diff`.

Still stuck? Open an issue on [GitHub](https://github.com/LinusNEP/EnvoDat/issues) with the command you ran, the full error, and the output of `pip freeze | grep -iE "envodat|torch|ultralytics|datasets|huggingface"`.
