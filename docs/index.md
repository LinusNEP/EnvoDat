# EnvoDat Dataset Documentation

```bash
pip install envodat
```

## Welcome to EnvoDat!

EnvoDat is a large-scale, multisensory dataset for **robotic spatial awareness and semantic reasoning in heterogeneous environments**. It was recorded in 13 indoor, outdoor and subterranean environments, from an exhibition hall and a glass building to a forest path and underground tunnels, under conditions that challenge today's perception and SLAM algorithms: dynamic entities, high illumination and shadows, fog and rain, partial and zero visibility, and different times of day.

The release spans more than 1.9 TB of time-synchronised data from ten sensing modalities (RGB, depth, 3D LiDAR, LiDAR intensity images, IMU and more), recorded with a wheeled robot indoors and a quadruped outdoors. On top of the raw data, EnvoDat provides 89K fine-grained, polygon-based annotations of 82 object and terrain classes, together with ground-truth trajectories for SLAM benchmarking. As a result, the same dataset supports SLAM and odometry evaluation, object detection, supervised learning and the fine-tuning of multimodal foundation models.

<div class="video-wrap">
<iframe src="https://www.youtube.com/embed/5OcByVmTUPQ" title="EnvoDat summary video" allowfullscreen></iframe>
</div>

🆕 **EnvoDat is now on the Hugging Face Hub.** The annotated scenes can be downloaded training-ready with `envodat.download("mu-hall")` or loaded directly with 🤗 `datasets`; see [Hugging Face](huggingface.md).

```python
import envodat as ed

ed.init("./envodat_data")                       # where the data is stored
ed.download("mu-hall")                          # one annotated scene, YOLO-ready
loader = ed.dataloader("mu-hall", split="train", batch_size=8)
ed.train_yolo("mu-hall", model="yolo11n.pt", epochs=100)
```

![EnvoDat scene characteristics](scene_characteristics.gif)

# Useful Links

- Paper (ICRA 2025): [IEEE Xplore](https://ieeexplore.ieee.org/document/11127594) · [arXiv:2410.22200](https://arxiv.org/abs/2410.22200)
- Dataset on Hugging Face: [LinusNEP/EnvoDat](https://huggingface.co/datasets/LinusNEP/EnvoDat)
- Code, toolkit and leaderboard: [github.com/LinusNEP/EnvoDat](https://github.com/LinusNEP/EnvoDat)
- Full release, including raw ROS bags: [EnvoDat download page](https://sites.google.com/view/envodat/download)
- Project website (figures, poster, presentation): [sites.google.com/view/envodat](https://sites.google.com/view/envodat/overview)

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
examples
huggingface
api
```

```{toctree}
:maxdepth: 2
:caption: Dataset

modalities
environments
```

```{toctree}
:maxdepth: 2
:caption: Benchmark

get_started
benchmark
```

```{toctree}
:maxdepth: 2
:caption: Help

troubleshooting
maintainers
```

# Citation

If you use EnvoDat in your research, please cite:

```bibtex
@INPROCEEDINGS{11127594,
  author={Nwankwo, Linus and Ellensohn, Björn and Dave, Vedant and Hofer, Peter and Forstner, Jan and Villneuve, Marlene and Galler, Robert and Rueckert, Elmar},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={EnvoDat: A Large-Scale Multisensory Dataset for Robotic Spatial Awareness and Semantic Reasoning in Heterogeneous Environments},
  year={2025},
  pages={153-160},
  doi={10.1109/ICRA55743.2025.11127594}}
```

# License

The EnvoDat dataset and documentation are released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The toolkit and scripts are released under the [MIT License](https://github.com/LinusNEP/EnvoDat/blob/main/LICENSE-CODE).

This project has received funding from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), No. #430054590 (TRAIN).
