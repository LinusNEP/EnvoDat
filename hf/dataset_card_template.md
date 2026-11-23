---
license: cc-by-4.0
pretty_name: EnvoDat
task_categories:
- object-detection
tags:
- robotics
- autonomous-navigation
- subterranean
- indoor
- outdoor
- yolo
language:
- en
__CONFIGS__
---

# EnvoDat

**A Large-Scale Multisensory Dataset for Robotic Spatial Awareness and Semantic Reasoning in Heterogeneous Environments** (ICRA 2025)

[Documentation](https://linusnep.github.io/EnvoDat/) · [Project website](https://sites.google.com/view/envodat/overview) · [Paper](https://ieeexplore.ieee.org/document/11127594) · [arXiv](https://arxiv.org/abs/2410.22200) · [Code & benchmark](https://github.com/LinusNEP/EnvoDat)

EnvoDat was captured across heterogeneous indoor, outdoor and subterranean environments, including dynamic halls, natural fields and underground tunnels, under varying illumination, fog, rain and low or zero visibility. This Hub repository hosts the **annotated, training-ready scenes** for object detection and semantic perception; the raw multisensor ROS bags (RGB, LiDAR, depth, IMU, ground-truth poses) remain available from the [EnvoDat download page](https://sites.google.com/view/envodat/download).

## Scenes

Each scene is a separate config, because the class vocabularies differ between environments.

__SCENES_TABLE__

## Layout

```
data/<scene>/{train,validation,test}-*.parquet   # images + boxes, for 🤗 datasets and the viewer
yolo/<scene>.zip                                 # Ultralytics layout: train/valid/test images + labels, data.yaml
```

Each scene is packed into a few large files, so downloading it takes one or a few requests. In the Parquet files, `objects.bbox` holds COCO-style pixel boxes `[x_min, y_min, width, height]`, `objects.category` the class ids (a `ClassLabel` with the scene's class names) and `objects.category_name` the names directly. The zip unpacks to the standard Ultralytics layout with a `data.yaml` whose paths are relative to it.

## Usage

**EnvoDat toolkit**

```bash
pip install envodat
envodat download __FIRST_SCENE__        # YOLO-ready, with an absolute-path data.yaml
envodat train __FIRST_SCENE__ --model yolo11n.pt
```

**🤗 datasets**

```python
from datasets import load_dataset

ds = load_dataset("__REPO_ID__", "__FIRST_SCENE__")
sample = ds["train"][0]
sample["image"], sample["objects"]["bbox"], sample["objects"]["category_name"]
```

**Ultralytics YOLO (download one scene, then train)**

```bash
pip install huggingface_hub ultralytics
hf download __REPO_ID__ yolo/__FIRST_SCENE__.zip --repo-type dataset --local-dir ./envodat
unzip -q ./envodat/yolo/__FIRST_SCENE__.zip -d ./envodat/__FIRST_SCENE__
yolo train model=yolo11n.pt data=./envodat/__FIRST_SCENE__/data.yaml epochs=100 imgsz=640
```

Full documentation, including PyTorch dataloaders, training with 🤗 Transformers and SLAM evaluation, is at [linusnep.github.io/EnvoDat](https://linusnep.github.io/EnvoDat/).

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@INPROCEEDINGS{11127594,
  author={Nwankwo, Linus and Ellensohn, Björn and Dave, Vedant and Hofer, Peter and Forstner, Jan and Villneuve, Marlene and Galler, Robert and Rueckert, Elmar},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={EnvoDat: A Large-Scale Multisensory Dataset for Robotic Spatial Awareness and Semantic Reasoning in Heterogeneous Environments},
  year={2025},
  pages={153-160},
  doi={10.1109/ICRA55743.2025.11127594}}
```

## Acknowledgement

This project has received funding from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), No. #430054590 (TRAIN).
