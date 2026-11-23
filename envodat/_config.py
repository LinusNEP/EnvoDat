import os
from pathlib import Path

HF_REPO_ID = os.environ.get("ENVODAT_HF_REPO", "LinusNEP/EnvoDat")
CPS_SHARE_URL = os.environ.get(
    "ENVODAT_CPS_SHARE", "https://cloud.cps.unileoben.ac.at/index.php/s/cs7qF4Qq5bZbt9k")

_state = {"root": Path(os.environ.get("ENVODAT_ROOT", "./envodat_data"))}


def init(root_path):
    """Set the local root folder where EnvoDat data is downloaded and read from.

    Args:
        root_path: Folder for all EnvoDat data. Created if it does not exist.
            Can also be set with the ``ENVODAT_ROOT`` environment variable.

    Returns:
        The resolved root path.
    """
    root = Path(root_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _state["root"] = root
    return root


def root():
    """Return the current data root (see :func:`init`)."""
    return Path(_state["root"]).expanduser().resolve()
