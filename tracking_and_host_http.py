from ultralytics import YOLO
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from multiprocessing import Process, Queue, Semaphore, shared_memory
import numpy as np
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


# YOLOv8 detection process
def run_detection(shared_name, frame_shape):
    global fps
    model = YOLO(model_path)
    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    classes = [0, 1, 2, 3, 5, 7]

    # Open the video file or camera
    
    cap = cv2.VideoCapture(input)
    
    shm = shared_memory.SharedMemory(name=shared_name)
    
    frame_count = 0
    start_time = time.time()

    total_count = 0
    total_time = 0

    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLOv8 detection
            results = model.track(frame,  imgsz=imgsz ,persist=True, classes=classes, show=show,verbose=verbose)
        
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
            

            # Annotate the frame
            annotated_frame = results[0].plot()

            # # Send the annotated frame to the queue
            # if queue.full():
            #     queue.get_nowait() # if empty throw an exception "Queue empty"
            # queue.put(annotated_frame)
            if annotated_frame.shape != frame_shape:
                annotated_frame = cv2.resize(annotated_frame, (frame_shape[1], frame_shape[0]))

            # Write the frame into shared memory
            np_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            np.copyto(np_frame, annotated_frame)
        
            
            
        else:
            print(f" fps averega {total_count/total_time if total_time != 0 else 0 }")
            break

    cap.release()

# HTTP server process to serve video feed
class VideoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            shm = shared_memory.SharedMemory(name=self.server.shared_name)
            

            while True:
                # Check if there's a frame in the queue
                # 
                frame = np.ndarray(self.server.frame_shape, dtype=np.uint8, buffer=shm.buf)
                # frame = self.server.queue.get()
                # frame = self.server.queue[0]

                # Encode the frame to JPEG and send it to the client
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                self.wfile.flush()

                # Optionally, break the loop if the client disconnects
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def run_server(shared_name, frame_shape, port=5000):
    server_address = ('', port)
    httpd =  ThreadingHTTPServer(server_address, VideoHandler)
    httpd.shared_name = shared_name
    httpd.frame_shape = frame_shape
    # httpd.queue = queue  # Attach the queue to the server
    print(f'Serving on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    # Create a Queue to share frames between processes
    # Create shared memory
    shared_name = "shared_frame"
    frame_shape = (640, 640, 3)  # Example frame shape (height, width, channels)
    frame_size = int(np.prod(frame_shape)) 
    
    print(np.prod(frame_shape))
    shm = shared_memory.SharedMemory(create=True, size=frame_size, name=shared_name)
    # Start the detection process
    detection_process = Process(target=run_detection, args=(shared_name,frame_shape ), daemon=True)
    detection_process.start()

    # Start the HTTP server process
    run_server(shared_name, frame_shape)
    detection_process.join()
    shm.close()
    shm.unlink()
