from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT format
model.export(
    format = "engine",
    dynamic = True,
    batch = 8,
    workspace = 3,
    int8 = True
    # half=True
    )  # creates 'yolov8n.engine'

# Load the exported TensorRT model

# tensorrt_model = YOLO("yolov8n.engine")