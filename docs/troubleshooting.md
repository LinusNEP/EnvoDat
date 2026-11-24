# Troubleshooting

## Installation

**`envodat: command not found` or `yolo: command not found`.** Both commands are installed into the active environment, so make sure it is activated (`source EnvoDatEnv/bin/activate`).

**`ImportError: Install ... pip install 'envodat[...]'`.** The feature you called needs an optional extra. Install the one named in the message, or everything with `pip install "envodat[all]"`.

**PyTorch does not see the GPU.** pip installs the default PyTorch build. Install `torch` for your CUDA version from the [official selector](https://pytorch.org/get-started/locally/) first, then reinstall `envodat`.

## Downloading from Hugging Face

**`scene 'xyz' is not on the Hub`.** Check the scene id with `envodat list`. Ids are lower case with dashes, for example `mu-hall` and `subt-zab-01`; names such as `MU-Hall` are converted automatically.

**"EnvoDat on Hugging Face is gated", or `401` / `403` errors.** Open the [dataset page](https://huggingface.co/datasets/Linpen/EnvoDat), agree to the conditions while logged in, and log in once on your machine with `hf auth login`. A token with read access is enough.

**`429 Too Many Requests`.** The Hub limits how many requests an account makes in five minutes. Each scene is only a few files, so this should be rare; if it happens, wait five minutes and run the same command again. Logging in with `hf auth login` raises the limit.

**Disk space and cache.** Scenes are written to the data root (`ed.init(...)`, `--root` or `ENVODAT_ROOT`). The Hugging Face cache lives in `~/.cache/huggingface` and can be moved with the `HF_HOME` environment variable.

**`load_dataset` complains about the config name.** Every scene is its own config, so pass one: `load_dataset("Linpen/EnvoDat", "mu-hall")`.

## Downloading raw data

**`Cannot reach cloud.cps.unileoben.ac.at`.** The CPS server may be blocked by a firewall or proxy on your network. urllib honours the standard `HTTPS_PROXY` variable, so set it if your institution uses a proxy.

**Interrupted downloads.** Run the same `download_raw` call or `envodat raw` command again: partial `.part` files are resumed, and complete files are skipped.

**`Need X GB but only Y GB free`.** Narrow the glob patterns, or check the size first with `dry_run=True` / `--dry-run`.

## Training and evaluation

**`data.yaml not found`.** The scene has not been downloaded into the current data root. Call `ed.download("<scene>")`, or point `ed.init()` at the folder you used before.

**`results.csv` columns not found in `metrics_eval.py`.** Your Ultralytics version writes a different CSV schema. Install a recent version with `pip install -U ultralytics`.

**`Degenerate covariance rank, Umeyama alignment is not possible`.** The trajectories have too few associated poses, or they are almost a straight line. Check that the timestamps of the estimate and the ground truth overlap (both in seconds); you can loosen the association with `max_diff`.

Still stuck? Open an issue on [GitHub](https://github.com/LinusNEP/EnvoDat/issues) with the command you ran, the full error, and the output of `pip freeze | grep -iE "envodat|torch|ultralytics|datasets|huggingface"`.
