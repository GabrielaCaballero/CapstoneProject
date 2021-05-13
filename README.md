# NYU Tandon - MOT Capston Project 2021 - Object Detection and Counting

## Getting Starter

### 1. Clone the repo
### 2. Install the Requirements TensorFlow GPU
```bash
pip install -r requirements-gpu.txtCancel changes
```
### 3. Downloading Official Pre-trained Weights
```bash
YOLOv4 comes pre-trained and able to detect 80 classes. We use the pre-trained weights. Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.
```
### 4. To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 
```
### 5.To run the detection on your video and count the objects you need to run the following function:
```bash
!python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.mp4 --count
```
Change the video path to the video path that you want to run and detect and the name of the resulting detected video as the one of your preferene


