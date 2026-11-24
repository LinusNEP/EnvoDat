# For Maintainers

This page covers publishing the data, the toolkit and the documentation. Users do not need any of it.

## Order: verify privately, then go public

Nothing below becomes public until you take an explicit step, as long as you keep this order:

| Step | Stays private while you check | Becomes public when |
| ---- | ----------------------------- | ------------------- |
| Code and docs | Build and test on your machine (see [Documentation site](#documentation-site)) | You push to the public GitHub repo (any branch is visible there) |
| Docs website | Pushes to other branches and pull requests only *build* the site | A push to `main` deploys it to GitHub Pages |
| Dataset on Hugging Face | Uploads go to a **private** repo by default | You switch the repo to public in its settings (or pass `--public`) |
| `envodat` on PyPI | Nothing is uploaded until you publish a GitHub release | You publish a release; a PyPI version can never be replaced |

## Publishing the data to Hugging Face

`scripts/cps_to_hf.py` moves data from the CPS server share to the Hub. Run it on a machine that can reach both `cloud.cps.unileoben.ac.at` and `huggingface.co`, ideally a CPS server, so the download stays local. First install the repository and log in with a token that has **write** access:

```bash
git clone https://github.com/LinusNEP/EnvoDat.git && cd EnvoDat
pip install -e ".[hf]"
hf auth login
```

### 1. Inspect the share

```bash
python scripts/cps_to_hf.py tree
```

This lists every file on the share and writes `cps_tree.tsv` (size, matched scene id, path). It also summarises sizes by top-level folder and by file extension, and it flags any file above the Hub's 50 GB per-file limit. Use the listing to choose the `--include` patterns below.

### 2. Publish the annotated scenes

```bash
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip" --dry-run      # files that will be used
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip" --stage-only   # check ./hf_staging
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip"                # upload
```

`--tree cps_tree.tsv` reuses the listing from step 1 instead of listing the whole share again, which takes thousands of requests and can time out on a busy server. Rerun `tree` when files on the share change. Timeouts and server errors on single requests are retried automatically.

One YOLO flavour is enough, since the YOLOv7–v12 exports share the same label format. When a scene exists under several folder spellings (`MU-COR`, `MU-Cor`, `mu-cor`), only one copy is used, preferring the spelling of the scene's official name. Copies that differ in size are listed so you can check them; `--no-dedupe` turns this off.

The command downloads the files that match `--include` and extracts any `.zip` or `.tar(.gz)` archives. It then finds every YOLO export (a folder with `train/images`) and maps it to its scene from the path. For example, `Indoors/MU-Hall/annotations/MU-Hall_YOLOv8.zip` becomes the `mu-hall` config. Finally, it hands everything to `scripts/hf_upload.py`, which packs each scene into one zip (`yolo/<scene>.zip`) and a few Parquet files (`data/<scene>/`) and generates the dataset card from `hf/dataset_card_template.md`. Only these packed files are uploaded, because the Hub limits each account to about 1000 requests per five minutes and a scene stored as single images would exceed that; loose per-image files from an earlier upload are deleted in the same commit. The upload is resumable: if it is interrupted, run the same command again. The repository is created **private**, so only you (and members of your organisation, if it is an organisation repo) can see it. Check the dataset page, the viewer and `envodat download` while logged in. When everything looks right, open the dataset's **Settings** on Hugging Face and change its visibility to public, optionally with gated access (see below). Private storage on a free account is limited (about 100 GB), so if the annotated scenes are larger, check a subset first, for example with `--include "*mu-hall*yolo*"`. The scripts refuse to upload into an existing *public* repo unless you pass `--public`.

Before the real upload, check the `--stage-only` output for two things. First, every scene should be listed with the expected split sizes. Second, there should be no warning about a second export for the same scene, which would mean, for instance, that per-sequence exports exist and need merging first.

When new scenes or fixes are uploaded, tag the release so that users can pin it:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_tag('Linpen/EnvoDat', tag='v1.0', repo_type='dataset')"
```

### 3. Raw data (later)

The full release is about 1.9 TB. Hugging Face generally supports repositories up to about 300 GB and asks for larger ones to be arranged in advance (`datasets@huggingface.co`), so request the storage before you start. Once it is confirmed, the raw files can be transferred one at a time, which keeps local disk use to a single file:

```bash
python scripts/cps_to_hf.py raw --include "*.bag" "*gt-poses*" --dry-run
python scripts/cps_to_hf.py raw --include "*.bag" "*gt-poses*" --yes
```

Files go to `raw/<share path>` on the Hub, and files already uploaded with the same size are skipped, so the command can be rerun at any time. Files larger than 50 GB are skipped and listed; split them first (for example with `rosbag filter` by time or topic). Until a toolkit release reads `raw/` from the Hub, users keep downloading raw data from the CPS server with `envodat.download_raw`.

### Access control

If the data should only be shared after users agree to terms (as with the current request form), enable **gated access** in the dataset's settings on Hugging Face when you make it public. You can also add custom fields, such as name and affiliation, to the access request. The toolkit then works for everyone who has been granted access and has logged in with `hf auth login`.

## Releasing the toolkit on PyPI

The workflow `.github/workflows/publish.yml` builds the package and publishes it to PyPI whenever a GitHub release is published. It uses PyPI trusted publishing, so no token is stored in the repository. One-time setup:

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a pending trusted publisher for project `envodat`, owner `LinusNEP`, repository `EnvoDat`, workflow `publish.yml`, environment `pypi`.
2. In the GitHub repository settings, create an environment named `pypi`.

To release, bump `version` in `pyproject.toml` and `__version__` in `envodat/__init__.py`, then publish a GitHub release (e.g. tag `v0.1.0`). Do this only after the toolkit has been tested against the real Hub data: a version published on PyPI can be yanked, but its number can never be reused.

## Documentation site

These pages are built with Sphinx (Read the Docs theme, Markdown via MyST) from the `docs/` folder. The workflow `.github/workflows/docs.yml` builds them on every push and pull request, but deploys to GitHub Pages at [linusnep.github.io/EnvoDat](https://linusnep.github.io/EnvoDat/) only on pushes to `main`. One-time setup: in the repository's **Settings → Pages**, set **Source** to **GitHub Actions**, and do this *before* merging into `main`. Under the old "Deploy from a branch" setting, GitHub would publish the raw `docs/` Markdown in place of the built site.

To preview locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
python -m http.server -d docs/_build/html
```

## Leaderboard

The [Benchmark](benchmark.md) page is generated from `results/*.yaml` by `python scripts/generate_leaderboard.py`; see [CONTRIBUTING.md](https://github.com/LinusNEP/EnvoDat/blob/main/CONTRIBUTING.md).
