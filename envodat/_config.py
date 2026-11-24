import os
from pathlib import Path

HF_REPO_ID = os.environ.get("ENVODAT_HF_REPO", "Linpen/EnvoDat")
CPS_SHARE_URL = os.environ.get(
    "ENVODAT_CPS_SHARE", "https://cloud.cps.unileoben.ac.at/index.php/s/cs7qF4Qq5bZbt9k")

_state = {"root": Path(os.environ.get("ENVODAT_ROOT", "./envodat_data"))}


def init(root_path):
    """Sets the local folder where EnvoDat data is stored and read.

    Args:
        root_path: Data folder, created if missing. ``ENVODAT_ROOT`` sets the default.

    Returns:
        The resolved folder path.
    """
    root = Path(root_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _state["root"] = root
    return root


def root():
    """Return the current data root (see :func:`init`)."""
    return Path(_state["root"]).expanduser().resolve()
