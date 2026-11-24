# Get Started: Train a YOLO Detector
In this walkthrough, we show how to train, evaluate, and benchmark supervised learning models on EnvoDat. Further, we show how to perform inference on new data and provide a walk-through on key steps in model selection and fine-tuning.

## Prerequisites
- Python 3 is installed on your system
- Basic knowledge of Python and command-line operations
- GPU for faster training (optional but recommended)

## Environment Setup
1.  Clone the Envodat repository:
```bash
git clone https://github.com/LinusNEP/EnvoDat.git
cd EnvoDat
```
2.  Create a virtual environment with Python 3.8+ and install the necessary Python requirements.
If Python 3 is not installed, download and install it from [Python's official site](https://www.python.org/downloads/).
```bash
sudo apt-get install python3-venv
python3 -m venv EnvoDatEnv
source EnvoDatEnv/bin/activate
```
After the above step, you should see `(EnvoDatEnv)` at the start of your command prompt, indicating that the virtual environment is active. Once activated, install the necessary packages, e.g., ultralytics for YOLO and other relevant Python libraries:
```bash
pip install torch ultralytics
pip install matplotlib opencv-python
pip install pandas matplotlib
```

## Prepare the EnvoData Dataset

### Option A (recommended): fetch a training-ready scene from Hugging Face
The annotated scenes are published on the [Hugging Face Hub](https://huggingface.co/datasets/Linpen/EnvoDat) already in YOLO's directory structure, together with a ready-made config. Hence, a single command replaces steps 1 and 2 of Option B:
```bash
pip install envodat
envodat --root ./data download mu-hall
```
The scene then sits in `./data/yolo/mu-hall/`, and its config is `./data/yolo/mu-hall/data.yaml`. Use that path wherever this guide writes `/.../dataset/envodata-mu-hall.yaml`, and continue with [Training the YOLO model on EnvoDat](#training-the-yolo-model-on-envodat). Alternatively, `envodat train mu-hall` downloads and trains in one go. For other model families, see the [Examples](examples.md#training-example).

### Option B: download manually and reorganise
1.   Download the annotated data at [EnvoDat annotations](https://sites.google.com/view/envodat/download). Depending on your task, download only the format that satisfies your requirements, e.g., YOLOv*, COCO, OpenAI-CLIP, VOC, etc. We organised the EnvoDat dataset in the hierarchical structure shown below:

```
EnvoDat/
├── Indoors/
│   ├── mu-hall/
│   │	├── annotations/
│   │	└── ...
│   │
│   └── ...
├── Outdoors/
│   └── ...
├── SubTunnels/
│   └── ...
└── README.md
```

2.   For this demo, we show how to train YOLOv8 model on the MU-Hall data. Create a folder named dataset, and reorganise the downloaded annotation data in YOLO’s required directory structure as shown in the example below:
```
dataset/
├── test/
│   ├── images/
│   └── labels/
├── train/
│    ├── images/
│    └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── envodata-mu-hall.yaml
```
In each of the scenes, provide a YOLOv* configuration file named `envodata-*.yaml` which specifies the dataset paths and class names. If it does not exist, create one in the following form:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: <number of classes>
names: ["class_name1", "class_name2", ..., "class_nameN"]
```

## Training the YOLO model on EnvoDat

1.	Download the pre-trained model checkpoints (e.g., `yolov8s.pt` for YOLOv8 small or `yolo11n.pt` for YOLO11 nano):

| Example models | Pretrained checkpoints download |
|----------------|--------------------------------|
| YOLOv8    | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt), [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt), [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt), [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt), [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)  |
| YOLOv9    |  [YOLOv9t](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt), [YOLOv9s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt), [YOLOv9m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt), [YOLOv9c](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt), [YOLOv9e](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt)   |
| YOLOv10    |  [YOLOv10n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt), [YOLOv10s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt), [YOLOv10m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt), [YOLOv10b](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt), [YOLOv10l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt), [YOLOv10x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt)  |
| YOLO11     |  [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt), [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt), [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt), [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt), [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)  |

2.	Train the model with the following command:
```bash
# For YOLOv8
yolo train model=yolov8n.pt data=/.../dataset/envodata-mu-hall.yaml epochs=100 imgsz=640

# For YOLO11
yolo train model=yolo11x.pt data=/.../dataset/envodata-mu-hall.yaml epochs=100 imgsz=640
```
Adjust the model and hyperparameters as needed (e.g., `model`, `epochs`, `batch size`) for optimal results:
```bash
yolo train model=yolo11x.pt \
    data=/.../dataset/envodata-mu-hall.yaml \
    epochs=150 imgsz=640 batch=16 \
    lr0=0.01 lrf=0.1 momentum=0.937 weight_decay=0.0005 \
    save_period=10 patience=50 device=0
```

| Argument | Meaning |
| --- | --- |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `lr0`, `lrf` | Initial learning rate, and final learning rate as a fraction of `lr0` |
| `momentum` | Momentum for the SGD optimiser |
| `weight_decay` | Regularisation strength |
| `save_period` | Save a checkpoint every n epochs |
| `patience` | Stop after this many epochs without improvement |
| `device` | `0` for the first GPU, `cpu` for the CPU |
3.	Models' Performance Evaluation

After training, evaluate the performance on the validation data:
```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')  # the weights from your training run
validation_metrics = model.val(split='val')  # This automatically uses the validation split defined in envodata-*.yaml
```
You could also run the evaluation on the test set to see how the model performs on unseen data:
```python
test_metrics = model.val(split='test')  # This uses the test split from the envodata-*.yaml
```
Once you are done training, you can save the trained model weights for inference or for further fine-tuning:
```python
model.export(format='onnx')  # Export to ONNX format (alternatively use 'torchscript', 'engine' etc.)
```
## Run inference on multiple unseen images
1.	Prepare new images: Ensure all the new images you want to run inference on are in a single directory.
2.	You can use the following command to run inference on all the new images in the directory.
```bash
yolo predict model=best.pt source=path/to/unseen_images
```
Alternatively, you can simply run the following Python script:
```bash
python scripts/unseenImgInference.py \
        --model runs/detect/train/weights/best.pt \
        --source path/to/unseen_images \
        --project runs/detect \
        --name exp \
        --conf 0.4
```
## Metrics and Analysis
After training the model, the training results `results.csv` will be generated. You can analyse the validation losses, validation metrics (Precision, Recall, mAP), and learning rate over time using the following python script:

```bash
python scripts/metrics_eval.py
```



