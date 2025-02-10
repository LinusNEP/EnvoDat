from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="data.yaml", epochs=150, imgsz=640)
"""
results = model.train(
    data="data.yaml",
    epochs=150,	   
    imgsz=640,           
    lr0=0.001,		   
    batch=16,		   
    weight_decay=0.0005,  
    momentum=0.9,         
    warmup_epochs=3	  
)
"""

