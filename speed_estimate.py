import cv2
from ultralytics import YOLO, solutions

import argparse
import time

parser = argparse.ArgumentParser(description="Example of argument passing")
parser.add_argument("--model", type=str, help="path of your model, if using model yolo, it will be auto downloaded",default="yolov8n_openvino_model/")
parser.add_argument("--input", type=str, help="0 to use camera device in your lap or path of a video", default="./input/traffic_highway_1280_720.mp4")
parser.add_argument("--output",type=str, help="save output to a specific path", default="./output/output.avi")
parser.add_argument("--imgsz",type=str, help="image input size", default="640")
parser.add_argument("--save", action="store_true", help="save output to output path")
parser.add_argument("--show", action="store_true", help="display inference frame on a window")
parser.add_argument("--fps", action="store_true", help="show fps when running")
parser.add_argument("--verbose", action="store_true", help="show log or not")
args = parser.parse_args()

model_path, input, output, imgsz, save ,show, fps, verbose = vars(args).values()


model = YOLO(model_path)

# model.to('cuda')
classes = [0,1,2,3,5,7]
names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

cap = cv2.VideoCapture(input)


assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


frame_count = 0
start_time = time.time()

total_count = 0
total_time = 0

# Video writer
video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=False,
)



while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0,  imgsz=imgsz,persist=True,classes=classes, verbose = verbose)

    im0 = speed_obj.estimate_speed(im0, tracks)
    
    
    if show:
        cv2.imshow("Video", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if save:
        video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
