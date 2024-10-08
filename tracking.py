import cv2
from ultralytics import YOLO
# Kiểm tra xem GPU có sẵn hay không
import time
import argparse

# Tạo đối tượng phân tích tham số
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




model = YOLO(model=model_path,task="detect")

# model.to("cuda")
classes = [1,2,3,5,7]
names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

cap = cv2.VideoCapture(input)


assert cap.isOpened(), "Error reading video file"

w, h, _fps_input= (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Video writer
video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), _fps_input, (w, h))

frame_count = 0
start_time = time.time()

total_count = 0
total_time = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    tracks = model.track(im0, persist=True,imgsz=imgsz,classes=classes,show=False,verbose=verbose) 
    
    if fps:
        frame_count+=1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            total_count += frame_count
            total_time += elapsed_time
            # Reset biến
            frame_count = 0
            start_time = time.time() 
    
    annotated_frame = tracks[0].plot()
    if show:
        cv2.imshow("Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if save:
        video_writer.write(annotated_frame)
    
        
print(f" fps averega {total_count/total_time if total_time != 0 else 0 }")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
