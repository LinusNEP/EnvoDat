# Maintaining EnvoDat

> **This guide is for EnvoDat maintainers.** It covers publishing data on the Hugging Face Hub, releasing the `envodat` package and updating the documentation website. To use the dataset, see the [documentation](https://linusnep.github.io/EnvoDat/) instead; to submit benchmark results, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Access required

Each task needs its own permissions, which the project owner grants:

| Task | Requires |
| ---- | -------- |
| Publishing data | Write access to the dataset repository on Hugging Face, and read access to the EnvoDat share on the CPS server |
| Releasing the toolkit | Permission to publish releases in the GitHub repository |
| Updating the documentation | Permission to merge into `main` in the GitHub repository |

## Release principle: verify privately, then publish

Every change is checked privately before it becomes public, and each kind of change goes public through one explicit step:

| Change | Private while it is checked | Public once |
| ------ | --------------------------- | ----------- |
| Code and documentation | Built and tested locally (see [Documentation website](#documentation-website)) | It is pushed to the GitHub repository, where every branch is visible |
| Documentation website | Pull requests and other branches only *build* the site | It is merged into `main`, which deploys the site |
| Dataset on Hugging Face | New dataset repositories are created **private** | The repository's visibility is set to public |
| `envodat` on PyPI | Nothing is uploaded before a GitHub release | A GitHub release is published; a PyPI version number can never be reused |

## Publishing data to Hugging Face

The annotated scenes are published at [huggingface.co/datasets/Linpen/EnvoDat](https://huggingface.co/datasets/Linpen/EnvoDat), and `scripts/cps_to_hf.py` moves data there from the EnvoDat share on the CPS server. The transfer runs best on a machine with a fast connection to both servers, ideally a CPS server itself, inside `tmux` or `screen` for long uploads. Install the repository and log in to Hugging Face with a token that has **write** access to the dataset repository:

```bash
git clone https://github.com/LinusNEP/EnvoDat.git && cd EnvoDat
pip install -e ".[hf]"
hf auth login
```

### 1. List the share

```bash
python scripts/cps_to_hf.py tree
```

The command lists every file on the share and writes `cps_tree.tsv` (size, matched scene id, path). It also summarises sizes by top-level folder and by file type, and it flags files above the Hub's 50 GB per-file limit. Later commands reuse this listing through `--tree cps_tree.tsv`, which avoids listing the whole share again (thousands of requests that can time out on a busy server); `tree` therefore only needs to be rerun after files on the share change.

### 2. Publish the annotated scenes

```bash
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip" --dry-run      # files that will be used
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip" --stage-only   # build ./hf_staging locally
python scripts/cps_to_hf.py --tree cps_tree.tsv annotated --include "*-yolov8.zip"                # upload
```

The command downloads the YOLO exports that match `--include`, extracts them and maps each one to its scene from the path; `Indoors/MU-Hall/annotations-mu-hall/mu-hall-yolov8.zip`, for example, becomes the `mu-hall` config. One YOLO flavour is enough, because the YOLOv7–v12 exports share the same label format. Several details are handled along the way:

- **Duplicate folders.** When a scene exists under several folder spellings (`MU-COR`, `MU-Cor`, `mu-cor`), only one copy is used, preferring the scene's official spelling. Copies that differ in size are listed for a manual check; `--no-dedupe` turns this off.
- **Mixed label files.** Label files that mix plain boxes and polygons are rewritten with the boxes as 4-corner polygons, because Ultralytics otherwise skips those images. No object is lost.
- **Packing.** `scripts/hf_upload.py` packs each scene into one zip (`yolo/<scene>.zip`) and a few Parquet files (`data/<scene>/`), and generates the dataset card from `hf/dataset_card_template.md`. Only these packed files are uploaded, since the Hub limits each account to about 1000 requests per five minutes and a scene stored as single images would exceed that. Loose per-image files from an older layout are deleted in the same commit.
- **Retries.** Timeouts and server errors on single requests are retried automatically, and an interrupted upload resumes when the same command is run again.

Before the real upload, the `--stage-only` output should list every scene with the expected split sizes and no warning about a second export for the same scene; such a warning usually means per-sequence exports that need merging first. The scripts create a new dataset repository as **private** and refuse to upload into an existing public one unless `--public` is given. A first release can therefore be checked on the Hub (dataset page, data viewer, `envodat download`) before its visibility is switched to public in the repository's **Settings**.

After an upload that replaced older files, squashing the history of `main` removes the old versions from it (files that an existing tag points to stay reachable through that tag), and a new tag lets users pin the release:

```bash
python -c "from huggingface_hub import HfApi; HfApi().super_squash_history('Linpen/EnvoDat', repo_type='dataset')"
python -c "from huggingface_hub import HfApi; HfApi().create_tag('Linpen/EnvoDat', tag='v1.1', repo_type='dataset')"
```

Squashing rewrites the history, so it should come before the tag, and existing tags are best left untouched; a new release gets the next tag (`v1.1`, `v1.2`, …).

### 3. Raw data

The raw recordings (ROS bags, ground-truth poses, IMU data, RGB-D archives) exceed 1 TB, which Hugging Face prefers to see in a repository of its own or arranged with them in advance (`datasets@huggingface.co`). `--repo-id` selects the target repository, so the raw data can go to a separate dataset such as `Linpen/EnvoDat-raw`:

```bash
python scripts/cps_to_hf.py --tree cps_tree.tsv --repo-id Linpen/EnvoDat-raw raw \
    --include "*.bag" "*gt-poses*" "*imu-*.csv" "*rgb-images.zip" "*depth-images.zip" --dry-run
```

Without `--dry-run`, and with `--yes` to confirm a transfer above about 300 GB, the files are transferred one at a time, so local disk use stays at a single file. They keep their share path under `raw/`, and files already on the Hub with the same size are skipped, which makes the command safe to rerun. Files above 50 GB are skipped and listed; they need splitting first, for example with `rosbag filter` by time or topic. Large numbers of small files, such as the loose LiDAR PNG layers, should be zipped per sequence before uploading, because every transferred file is its own commit. Until a toolkit release reads raw data from the Hub, `envodat.download_raw` downloads it from the CPS server.

### Access control

The dataset is **gated**: users agree to its conditions on the Hugging Face page and log in with `hf auth login` before they can download, and the toolkit tells them so when access is missing. The gate is configured in the dataset's **Settings** on Hugging Face (automatic or manual approval). Custom gate text and form fields, however, are stored in the YAML header of the dataset card as `extra_gated_prompt` and `extra_gated_fields`. Since every upload regenerates the card from `hf/dataset_card_template.md`, such fields belong in that template; otherwise the next upload removes them.

## Releasing the toolkit on PyPI

The workflow `.github/workflows/publish.yml` builds the package and publishes it to [PyPI](https://pypi.org/project/envodat/) whenever a GitHub release is published. It uses PyPI trusted publishing, so no token is stored in the repository. To release a new version:

1. Raise `version` in `pyproject.toml` and `__version__` in `envodat/__init__.py` (for example to `0.1.3`) and merge the change into `main`.
2. Optionally, test the package locally: `pip install build twine && python -m build && twine check dist/*`.
3. On GitHub, draft a new release with the tag `v<version>` on `main`, and publish it.
4. Check the **Publish to PyPI** run in the **Actions** tab and the new version on PyPI.

A version published on PyPI can be yanked but never replaced, so a broken release is fixed by publishing the next version.

Trusted publishing and the `pypi` environment are already configured for this repository. A fork or a moved repository needs them set up again: a trusted publisher on PyPI (project `envodat`, the repository's owner and name, workflow `publish.yml`, environment `pypi`) and an environment named `pypi` in the GitHub repository settings.

## Documentation website

The documentation is built with Sphinx (Read the Docs theme, Markdown via MyST) from the `docs/` folder. The workflow `.github/workflows/docs.yml` builds them on every push and pull request, but deploys them to [linusnep.github.io/EnvoDat](https://linusnep.github.io/EnvoDat/) only on pushes to `main`; it can also be started by hand from the **Actions** tab. The repository's Pages source is therefore set to **GitHub Actions** (**Settings → Pages**). A fork needs the same setting, because under "Deploy from a branch" GitHub publishes the raw Markdown files instead of the built site.

The files in `docs/` are Sphinx sources, so GitHub's file preview shows their navigation directives as code blocks; the website is where they are meant to be read. To preview the site locally:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
python -m http.server -d docs/_build/html
```

`-W` turns warnings into errors, as the deployment workflow does.

## Leaderboard

The [Benchmark](https://linusnep.github.io/EnvoDat/benchmark.html) page (source: `docs/benchmark.md`) is generated from `results/*.yaml` by `python scripts/generate_leaderboard.py`. New results arrive as pull requests that follow [CONTRIBUTING.md](CONTRIBUTING.md).
