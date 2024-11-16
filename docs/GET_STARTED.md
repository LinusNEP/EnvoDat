# Training, evaluation, and benchmarking of object detector and classifier models on EnvoData
In this demo, we will demonstrate how to train supervised learning models on EnvoData, perform inference on new data, and evaluate the model's performance. We will walk through key steps, including model selection, training, benchmarking, and fine-tuning.

## Prerequisites
- Python 3 installed on your system
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
1.   Download the annotated data at [EnvoDat annotations](https://sites.google.com/view/envodat/download). Depending on your task, download only the format that satisfies your requirements e.g., YOLOv*, COCO, OpenAI-CLIP, VOC, etc. We organised the EnvoDat dataset in the hierarchical structure shown below:

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

2.   For this demo, we will be training YOLOv8 model on the MU-Hall data. Create a folder named dataset, and reorganize the downloaded annotation data in YOLO’s required directory structure as shown in the example below:
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
In each of the scenes, we provide a YOLOv* configuration file named `envodata-*.yaml` which specifies the dataset paths and class names. If it does not exist, create one in the following form:

```yaml
train: ../train/images	
val: ../valid/images	
test: ../test/images	

nc: <number of classes>
names: ["class_name1", "class_name2", ..., "class_nameN"]
```

## Training the YOLO model on EnvoDat

1.	Download the pre-trained model checkpoints (e.g., `yolov8s.pt` for YOLOv8 small or `yolov11n.pt` for YOLOv11 nano):

| Example models | Pretrained checkpoints download |
|----------------|--------------------------------|
| YOLOv8    | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt), [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt), [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt), [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt), [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)  |
| YOLOv9    |  [YOLOv9t](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt), [YOLOv9s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt), [YOLOv9m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt), [YOLOv9c](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt), [YOLOv9e](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt)   |
| YOLOv10    |  [YOLOv10n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt), [YOLOv10s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt), [YOLOv10m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt), [YOLOv10b](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt), [YOLOv10l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt), [YOLOv10x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt)  |
| YOLOv11    |  [YOLOv11n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt), [YOLOv11s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11s.pt), [YOLOv11m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11m.pt), [YOLOv11l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11l.pt), [YOLOv11x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11x.pt)  |

2.	Train the model with the following command:
```bash
# For YOLOv8
yolo train model=yolov8n.pt data=yolo train model=yolov8n.pt data=/.../dataset/envodata-mu-hall.yaml epochs=100 imgsz=640
 epochs=100 imgsz=640

# For YOLOv11
yolo train model=yolov11x.pt data=yolo train model=yolov11x.pt data=/.../dataset/envodata-mu-hall.yaml epochs=100 imgsz=640
 epochs=100 imgsz=640
```
Adjust the model and hyperparameters as needed (e.g., `model`, `epochs`, `batch size`) for optimal results:
```bash
yolo train model=yolov11x.pt \
    data=/.../dataset/envodata-mu-hall.yaml \
    epochs=150 \		#Number of training epochs
    imgsz=640 \
    batch=16 \           #Batch size
    lr0=0.01 \			#Initial learning rate
    lrf=0.1 \			#Final learning rate
    momentum=0.937 \		#Momentum for SGD optimizer
    weight_decay=0.0005 \	#Regularization parameter
    save_period=10 \		#Save model every n epochs
    patience=50 \		#Number of epochs with no improvement after which training will be stopped
    device=0 \			#Specify the device for training (CPU or GPU)
```
3.	Models' Performance Evaluation

After training, evaluate the performance on the validation data:
```bash
validation_metrics = model.val(split='val')  # This automatically uses the validation split defined in envodata-*.yaml
```
You could also run the evaluation on the test set to see how the model performs on unseen data:
```bash
test_metrics = model.val(split='test')  # This uses the test split from the envodata-*.yaml
```
Once you are done training, you can save the trained model weights for inference or for further fine-tuning:
```bash
model.export(format='onnx')  # Export to ONNX format (alternatively use 'torchscript', 'engine' etc.)
```
## Run inference on multiple unseen images
1.	Prepare new images: Ensure all the new images you want to run inference on are in a single directory.
2.	You can use the following command to run inference on all the new images in the directory.
```bash
yolo predict model=best.pt source=path/to/unseen_images
```
Alternatively, you can simply run the following Python script:
```python
import os
import cv2
import matplotlib.pyplot as plt
import subprocess

model_path = '.../best.pt'  # Path to the trained model
source_dir = 'path/to/unseen_images'  # Directory containing the unseen images for inference
output_dir = 'runs/detect/exp'  # Directory to save the inference results
conf_threshold = 0.4  # Confidence threshold for detections

def run_inference():
    command = [
        'yolo', 'predict',
        f'model={model_path}',
        f'source={source_dir}',
        f'project={output_dir}',
        f'conf={conf_threshold}',
        'save_txt=True',
        'save_conf=True'
    ]
    subprocess.run(command)

def plot_images_with_boxes(image_path, labels_path=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  

    if labels_path and os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height, conf = map(float, line.strip().split())
                img_height, img_width, _ = image.shape
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(image)
    plt.show()

def main():
    run_inference()
    image_files = os.listdir(source_dir)
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            label_file = os.path.splitext(image_file)[0] + '.txt'
            plot_images_with_boxes(os.path.join(source_dir, image_file), os.path.join(output_dir, label_file))

if __name__ == "__main__":
    main()

```
## Metrics and Analysis
After training the model, the training results `results.csv` will be generated. You can analyse the validation losses, validation metrics (Precision, Recall, mAP), and learning rate over time using the following python script:

```python
import pandas as pd
import matplotlib.pyplot as plt

results_file = '.../results.csv'  # Path to the training results CSV file

def load_training_results(file_path):
    df = pd.read_csv(file_path)
    return df

# Training and validation losses
def plot_losses(df):
    # Extract relevant columns
    train_box_loss = df['train/box_loss']
    train_cls_loss = df['train/cls_loss']
    train_dfl_loss = df['train/dfl_loss']
    val_box_loss = df['val/box_loss']
    val_cls_loss = df['val/cls_loss']
    val_dfl_loss = df['val/dfl_loss']
    epochs = df['epoch']

    # Training losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_box_loss, label='Box Loss', color='blue')
    plt.plot(epochs, train_cls_loss, label='Classification Loss', color='orange')
    plt.plot(epochs, train_dfl_loss, label='DFL Loss', color='green')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Validation losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_box_loss, label='Validation Box Loss', color='blue')
    plt.plot(epochs, val_cls_loss, label='Validation Classification Loss', color='orange')
    plt.plot(epochs, val_dfl_loss, label='Validation DFL Loss', color='green')
    plt.title('Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Validation metrics
def plot_validation_metrics(df):
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']
    map_value = df['metrics/mAP50(B)']
    epochs = df['epoch']

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, precision, label='Precision', color='blue')
    plt.plot(epochs, recall, label='Recall', color='orange')
    plt.plot(epochs, map_value, label='mAP', color='green')
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning rate over time
def plot_learning_rate(df):
    # Assuming you want to plot the learning rate from the first group (pg0)
    learning_rate = df['lr/pg0']
    epochs = df['epoch']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rate, label='Learning Rate (pg0)', color='purple')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df = load_training_results(results_file)
    plot_losses(df)
    plot_validation_metrics(df)
    plot_learning_rate(df)

if __name__ == "__main__":
    main()

```

# Benchmark SoTA SLAM Algorithms
## Details coming soon!


