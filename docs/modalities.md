# Modalities

EnvoDat combines data from ten sensing modalities, synchronised with a ROS ApproximateTime policy. The raw data is curated as ROS bags and post-processed into per-modality files, organised by scene and sequence, so that you can download only the parts you need. The sensor specification table, mounting diagrams and calibration figures are on the project website's [Robots & Sensor Suites](https://sites.google.com/view/envodat/sensor-suites/robots-sensor-suites) page.

## Robots

| Platform | Used in | Notes |
| -------- | ------- | ----- |
| ROMR (wheeled) | Indoor scenes | Open-source ROS mobile robot ([LinusNEP/ROS-Mobile-Robot](https://github.com/LinusNEP/ROS-Mobile-Robot)) |
| Unitree Go1 (quadruped) | Outdoor and off-road scenes | Legged platform for unstructured terrain |

## Folder layout

Each scene follows the same structure, where `xxx` is the scene name and `0x` the sequence number:

```
<scene>/
├── gt-poses-xxx.csv            # ground-truth trajectory
├── seq0x/
│   ├── xxx-seq0x.bag           # ROS bag with all raw recorded data
│   ├── imu-xxx-seq0x.csv       # inertial data
│   ├── lidar-xxx-seq0x/        # point clouds (PCD) + 2D layer images (PNG)
│   └── rgbd-xxx-seq0x/         # RGB and depth images + index files
└── annotations/                # YOLO, COCO, OpenAI-CLIP, VOC, ... exports
```

Raw files are downloaded with glob patterns on these paths, e.g. `ed.download_raw("*mu-hall*seq01*/imu-*")` (see [Examples](examples.md#download-raw-data)).

## RGB and depth

RGB frames come from monocular cameras, including an ELP 4K camera. The `rgbd-xxx-seq0x` folder holds the post-processed RGB and depth images, indexed by `rgb-xxx-seq0x.csv` and `depth-xxx-seq0x.csv` with the columns:

```
<index>, <timestamp>, <frame_name>, <frame_width>, <frame_hight>
```

## LiDAR

3D LiDAR scans are stored as point clouds in PCD format. In addition, the LiDAR's 2D layers are provided as PNG images: near-infrared (NIR), range, reflectivity and signal.

## IMU

Accelerometer and gyroscope readings are stored in `imu-xxx-seq0x.csv`. The IMU noise parameters (angle and velocity random walk, accelerometer and gyroscope bias instability) were identified with Allan variance analysis.

## Ground-truth poses

Each scene has a pseudo-ground-truth trajectory, `gt-poses-xxx.csv`, built from dense LiDAR-inertial odometry. During recording, the robot slowed down at waypoints to take repeated measurements, which were used to refine the pose estimates. Each row is one timestamped pose in a global, fixed reference frame:

```
timestamp, position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w
```

The timestamp is ROS Unix epoch time, and the orientation is a quaternion. {func}`envodat.get_traj_np` loads this file as an `[N, 8]` array, {func}`envodat.to_tum` converts it to TUM format, and {func}`envodat.evaluate_traj` accepts it directly.

## Coordinate frames

EnvoDat uses KITTI-like conventions:

| Sensor | x | y | z |
| ------ | - | - | - |
| Cameras | left | down | forward |
| LiDAR | forward | left | up |

## Calibration

- **Camera intrinsics and extrinsics:** Kalibr's target-based pipeline and the MATLAB Camera Calibrator, with a 300 × 390 mm chessboard (15 × 10 inner corners, 21 mm squares). Twenty views were taken at 1–3 m, rotated left–right between 0° and 180° and tilted by about 30°.
- **Camera–LiDAR extrinsics:** the MATLAB calibration app and Kalibr.
- **Camera–IMU:** Kalibr's visual-inertial toolbox, for spatial and temporal calibration with the ELP 4K camera.

## Annotations

EnvoDat contains 89K layered, polygon-based annotations (10.3 per frame on average) across 82 object and terrain classes, split 70 / 20 / 10 into train, validation and test. The classes are grouped into 13 semantic categories:

| Category | Examples |
| -------- | -------- |
| Human | person |
| Transport | truck, car, van, bus, excavator, tractor, bicycle, scooter, motorcycle, train |
| Furniture and appliances | table, chair, desk, ventilator, socket |
| Lighting | light bulb, reflection, shadow, torch, ceiling light |
| Architecture and structure | door, pillar, staircase, window, window frame, pole, house, building |
| Safety | fire extinguisher, exit sign, warning sign, helmet, signpost |
| Utility | trash bin, bag, box, bucket, backpack, handbag, banner, cloth, fan, monitor, personal computer, sofa, vacuum cleaner |
| Nature and vegetation | bush, grass, flower, tree, sky, river, water pool |
| Hazard | wet floor, dark room |
| Flooring and hard surfaces | concrete, asphalt, textureless wall |
| Construction | ladder, rail, debris, guard rail, barricade |
| Infrastructure and traffic | traffic light, crosswalk |
| Miscellaneous | used sparingly, for objects that fit no other category |

The annotations are exported in several formats: TXT for the YOLO series and YOLO Darknet, JSON for COCO, OpenAI CLIP, PaliGemma and Florence-2, XML for Pascal VOC, and CSV for TensorFlow Object Detection and RetinaNet. The YOLO exports are the ones published on the [Hugging Face Hub](huggingface.md); the other formats are part of the full release.
