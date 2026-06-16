# EnvoDat Benchmark Leaderboard

Public leaderboard for the EnvoDat dataset. Baseline numbers are from the
[ICRA 2025 paper](https://ieeexplore.ieee.org/document/11127594). Last generated: 2025-09-08.


## SLAM track

Five SOTA SLAM pipelines on five representative EnvoDat scenes. ATE/RPE: lower is better. Scale Drift (SD): **optimum is 1.0** (values < 1 underestimate, > 1 overestimate scale). **Bold** = best per scene; DNF = tracking lost / did not complete; — = not reported.

| Scene | Dominant challenge |
| ----- | ------------------ |
| MU-Hall-01 | Dynamic entities |
| MU-TXN-01 | Varying illumination, opaque surfaces |
| SubT-ZAB-01 | Zero/partial visibility |
| MU-Cor-03 (night) | Zero/partial visibility |
| Leo-Str-01 | High-dimensional, feature-sparse |

### ATE (μ / σ) [m]

| Method | Category | MU-Hall-01 | MU-TXN-01 | SubT-ZAB-01 | MU-Cor-03 (night) | Leo-Str-01 |
| --- | --- | --- | --- | --- | --- | --- |
| FAST-LIO2 | LiDAR (filter) | 1.15 / 0.67 | **1.51 / 0.82** | 9.77 / 13.51 | **0.87 / 0.30** | 2.50 / 0.40 |
| DLIO | LiDAR (filter) | **1.14 / 0.49** | 2.64 / 1.14 | **6.93 / 8.81** | 3.29 / 2.47 | **1.76 / 1.65** |
| HDL-SLAM | LiDAR (graph) | 17.78 / 7.76 | 45.86 / 21.44 | 7.54 / 4.90 | 5.12 / 3.83 | 3.29 / 6.60 |
| RTAB-Map | Visual | 12.36 / 6.85 | 46.06 / 22.23 | DNF | 3.25 / 2.15 | 5.63 / 2.64 |
| ORB-SLAM3 | Visual | 23.37 / 9.99 | 21.93 / 10.31 | DNF | 54.74 / 35.04 | 53.98 / 24.86 |

### RPE (μ / σ) [m]

| Method | Category | MU-Hall-01 | MU-TXN-01 | SubT-ZAB-01 | MU-Cor-03 (night) | Leo-Str-01 |
| --- | --- | --- | --- | --- | --- | --- |
| FAST-LIO2 | LiDAR (filter) | **0.01 / 0.00** | **0.01 / 0.00** | 1.13 / 0.40 | **0.00 / 0.00** | **0.04 / 0.00** |
| DLIO | LiDAR (filter) | 0.04 / 0.00 | 0.03 / 0.00 | **0.29 / 0.00** | 0.04 / 0.00 | 0.05 / 0.00 |
| HDL-SLAM | LiDAR (graph) | 0.04 / 0.00 | 0.07 / 0.00 | **0.29 / 0.00** | 0.09 / 0.00 | 0.12 / 0.00 |
| RTAB-Map | Visual | 0.03 / 0.00 | 0.08 / 0.00 | DNF | 0.09 / 0.00 | 0.09 / 0.00 |
| ORB-SLAM3 | Visual | 0.04 / 0.00 | 0.05 / 0.00 | DNF | 0.05 / 0.00 | 0.08 / 0.00 |

### Scale Drift (μ / σ)

| Method | Category | MU-Hall-01 | MU-TXN-01 | SubT-ZAB-01 | MU-Cor-03 (night) | Leo-Str-01 |
| --- | --- | --- | --- | --- | --- | --- |
| FAST-LIO2 | LiDAR (filter) | 1.13 / 0.70 | 1.67 / 4.09 | 1.95 / 6.40 | 1.03 / 0.67 | **1.02 / 0.58** |
| DLIO | LiDAR (filter) | **1.10 / 0.50** | 2.19 / 6.34 | 1.45 / 3.94 | 1.19 / 3.19 | 1.12 / 2.05 |
| HDL-SLAM | LiDAR (graph) | 1.89 / 3.97 | **1.58 / 4.11** | **1.22 / 3.32** | **0.98 / 0.65** | 1.43 / 8.02 |
| RTAB-Map | Visual | 0.24 / 4.37 | 2.05 / 4.58 | DNF | 0.94 / 0.89 | 0.46 / 8.37 |
| ORB-SLAM3 | Visual | 0.88 / 2.72 | 1.89 / 6.99 | DNF | 0.13 / 1.58 | 0.03 / 0.15 |

## Detection track

Models trained across all scenes (lr=0.00025, batch=16, epochs=150, split 70/20/10). Ranked by mAP@50:95; **bold** = best per metric.

| Rank | Model | mAP@50 (%) | mAP@50:95 (%) | Precision (%) | Recall (%) | F1 | Infer (ms) | FPS | Mem (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | YOLOv8 | **71.85** | **61.00** | **72.85** | **68.90** | **0.71** | **8.75** | **114.29** | **4.71** |
| 2 | Fast R-CNN | 60.40 | 42.10 | 50.40 | 48.50 | 0.54 | 20.20 | 49.50 | 16.70 |
| 3 | Detectron2 | 50.10 | 41.70 | 50.01 | 48.65 | 0.54 | 18.20 | 54.95 | 16.10 |

## How to submit

1. Reproduce your result using the EnvoDat sequences and the tooling in
   [`docs/scripts/`](scripts/) (`evaluate_slam.py` for the SLAM track,
   `yolov8_training.py` + `model.val()` for the detection track).
2. Add an entry to the relevant file under [`results/`](../results/).
3. Run `python docs/scripts/generate_leaderboard.py` to regenerate this page.
4. Open a pull request with the YAML diff, the regenerated `BENCHMARK.md`, your
   exact command/config, and a link to logs or a short report.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full checklist.