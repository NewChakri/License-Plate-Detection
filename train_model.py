from ultralytics import YOLO

# Load a pre-trained YOLOv9 model
model = YOLO('yolov9c.pt')

# Train the model on your dataset
results = model.train(data='data.yaml', epochs=50, imgsz=640)
