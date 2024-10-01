from ultralytics import YOLO
import cv2
import argparse

parser = argparse.ArgumentParser(description="Example of argument passing")
parser.add_argument("--model", type=str, help="path of your model, if using model yolo, it will be auto downloaded", default="yolov8n.pt")
parser.add_argument("--dynamic", action="store_true", help="using model with dynamic input size")
parser.add_argument("--half", action="store_true", help="quantization model to float16(FP16)")
parser.add_argument("--int8", action="store_true", help="quantization model to int8")
parser.add_argument("--batch-size", type=int, help="batch size for inference", default=1)
args = parser.parse_args()

model_path, dynamic, half, int8, batch_size = vars(args).values()

print(f"model: {model_path}, dynamic: {dynamic}, half: {half}, int8: {int8}, batch size: {batch_size}")
model = YOLO(model_path)
model.export(format="openvino",dynamic=dynamic, half=half, int8=int8, batch = batch_size)
