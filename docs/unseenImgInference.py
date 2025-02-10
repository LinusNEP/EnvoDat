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

