# yolov8_computer_vision 
Tracking and estimate speed of vehicle traffic using yolov8, BoTSORT, transform perspective image
## Prerequisites
- Python 3.x
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/project-name.git
2. Create an virtual environment python
    ```bash
    python -m venv virtual_environment_name
3. Install the Dependency
    ```bash
    pip install -r requirements.txt
## Usage
### Run On CPU Using OPENVINO 
1. Export pytorch model (ex: yolov8n.pt or yolov5n.pt)  to openvino model
    ```bash
        python export_model_to_openvino.py --model yolov8n.pt --half 
2. Run model to test tracking, 
Markup :
* --model="yolov5nu_openvino_model" ( or "yolov8_openvino_model") : the path of model want to run, 
* --show flags: to show result (may decrease fps) 
* --save flags: to save output.
    ```bash
    python tracking.py --model "yolov8_openvino_model" --save




