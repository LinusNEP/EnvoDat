# API Reference

```python
import envodat as ed
```

All functions are available at the top level of this package. Heavy dependencies are imported lazily, so `import envodat` stays fast and only the feature you use needs its extra (see [Installation](installation.md#install-with-pip)).

## Initialization

```{eval-rst}
.. autofunction:: envodat.init
.. autofunction:: envodat.root
```

## Scene Catalog

```{eval-rst}
.. autodata:: envodat.catalog.SCENES
   :no-value:

.. autofunction:: envodat.scene_ids
.. autofunction:: envodat.info
```

## Download

```{eval-rst}
.. autofunction:: envodat.list_scenes
.. autofunction:: envodat.download
.. autofunction:: envodat.download_raw
.. autofunction:: envodat.cps_list
```

## Loading Data

```{eval-rst}
.. autofunction:: envodat.load
.. autofunction:: envodat.dataloader
.. autoclass:: envodat.EnvoDatDetection
.. autofunction:: envodat.data_yaml
.. autofunction:: envodat.class_names
.. autofunction:: envodat.scene_dir
```

## Training

```{eval-rst}
.. autofunction:: envodat.train_yolo
```

## Evaluation

```{eval-rst}
.. autofunction:: envodat.evaluate_traj
.. autofunction:: envodat.get_traj_np
.. autofunction:: envodat.to_tum
```

## Command Line

Installing the package also installs the `envodat` command. Run `envodat <command> --help` for every option.

| Command | Does |
| ------- | ---- |
| `envodat scenes` | Print the scene catalog |
| `envodat list` | List the scenes on the Hugging Face Hub |
| `envodat download <scene>... [--all] [--revision R] [--config F]` | Download annotated scenes |
| `envodat raw <glob>... [--dry-run]` | Download raw data from the CPS server |
| `envodat train <scene> [--model M] [--epochs N] [key=value ...]` | Train a YOLO detector |
| `envodat eval <est> <gt> [--align se3\|sim3\|none]` | ATE / RPE / scale drift |

The global option `--root` (or `ENVODAT_ROOT`) sets the data folder. Two more environment variables override the defaults: `ENVODAT_HF_REPO` (Hub repository) and `ENVODAT_CPS_SHARE` (CPS share link), with `ENVODAT_CPS_PASSWORD` for a protected share.
