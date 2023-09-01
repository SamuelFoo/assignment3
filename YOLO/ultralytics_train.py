from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
# model = YOLO('yolov8s.yaml')
# model = YOLO('models/YOLO/yolov8n_230523_1.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('models/YOLO/yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(
    data="detect/detect-train.yaml",
    epochs=200,
    imgsz=640,
    patience=0,
    batch=-1,
)
