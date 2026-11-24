"""EnvoDat toolkit: download, load, train on and evaluate the EnvoDat dataset.
"""

from ._config import init, root
from .catalog import SCENES, scene_ids
from .cps import cps_list, download_raw
from .hub import class_names, data_yaml, download, info, list_scenes, load, scene_dir
from .slam import evaluate_traj, get_traj_np, to_tum
from .torch_data import EnvoDatDetection, dataloader
from .train import train_yolo

__version__ = "0.1.2"

__all__ = [
    "init", "root", "SCENES", "scene_ids", "list_scenes", "info", "download", "download_raw",
    "cps_list", "load", "data_yaml", "scene_dir", "class_names", "dataloader",
    "EnvoDatDetection", "train_yolo", "get_traj_np", "to_tum", "evaluate_traj",
]
