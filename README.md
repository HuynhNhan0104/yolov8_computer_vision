# yolov8_computer_vision 
Tracking and estimate speed of vehicle traffic using yolov8, BoTSORT, transform perspective image
## Prerequisites
- Python 3.x
## Installation
1. Clone the repository:
   ```bash
        git clone https://github.com/HuynhNhan0104/yolov8_computer_vision.git
2. Create an virtual environment python
    ```bash
        python -m venv virtual_environment_name
3. Install the Dependency
    ```bash
        pip install -r requirements.txt
## Usage
### Run On CPU Using OPENVINO 
1. Export pytorch model (ex: yolov8n.pt or yolov5n.pt)  to openvino model
* --model="yolov8n.pt" ( or "yolov5n.py") : the path of model want to run, 
* --dynamic flags: set if you using with dynamic input
* --half flags: set if you want to quantilize your model to float16(decrease presicion but increase fps)
* --int8 flags: set if you want to quantilize your model to int8(decrease presicion but increase fps, should not check hard ware support it or not) , preferece at: [link](https://docs.ultralytics.com/integrations/openvino/)
    ```bash
    python export_model_to_openvino.py --model yolov8n.pt --half 
2. Run model to test tracking, 
* --model="yolov5nu_openvino_model" ( or "yolov8_openvino_model") : the path of model want to run, 
* --show flags: to show result (may decrease fps) 
* --save flags: to save output default is "./output/output.avi".
* --fps flags: to see fps(may not correct)
* --verbose flags: show log or not
    ```bash
    python tracking.py --model yolov8n_openvino_model/  --show --verbose
        
3. Run model to test detect, 
* --model="yolov5nu_openvino_model/" ( or "yolov8_openvino_model/") : the path of model want to run, 
* --show flags: to show result (may decrease fps) 
* --save flags: to save output.
* --fps flagas: to see fps(may not correct)
* --verbose flags: show log or not
    ```bash
    python detect.py --model yolov8n_openvino_model/  --show --verbose
4. Run model to test speed estimate, 
* --model="yolov5nu_openvino_model/" ( or "yolov8_openvino_model/") : the path of model want to run, 
* --show flags: to show result (may decrease fps) 
* --save flags: to save output.
* --fps flagas: to see fps(may not correct)
* --verbose flags: show log or not
    ```bash
    python speed_estimate.py --model yolov8n_openvino_model/  --show --verbose




