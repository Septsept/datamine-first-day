# **First day : WebCam-Based Real-Time AI with YOLOv8 and Segformer**
## **Description**
This project demonstrates two real-time Artificial Intelligence applications using webcam input:
1. **Real-Time Object Detection with YOLOv8**: Detect and track objects live from your webcam.
2. **Facial Segmentation with Segformer**: Perform precise facial segmentation using the `face-parsing` model.

The project showcases the power of modern AI frameworks applied to both object detection and semantic segmentation in real time.
## **Features**
### YOLOv8 (Object Detection)
- Detects objects in real time using webcam input.
- Visualizes detection results with bounding boxes and labels.
- Powered by the `YOLOv8` model from [Ultralytics](), optimized for speed and performance.

### Segformer (Facial Segmentation)
- Performs semantic segmentation on facial features (eyes, nose, lips, hair, etc.) and displays color-coded masks.
- Overlays segmentation results on the live webcam feed.
- Leverages the `Segformer` model pre-trained for fine-grained "face-parsing."

### **Sample Outputs**
- **YOLOv8:**
A video stream with detected objects highlighted, each framed by a bounding box along with its label.
- **Segformer:**
A video stream with segmented facial features, represented as color-coded overlays. Includes a **legend** indicating the segmented classes (e.g., face, nose, hair, etc.).

## **Installation**
### **Prerequisites**
Ensure you have the following tools installed:
1. **Python** (version 3.8 or higher)
2. **Pip** (compatible with your Python version)
3. **Webcam** connected to your device.

### **Install Required Libraries**
1. Clone this repository:
``` bash
   git clone https://github.com/Septsept/datamine-first-day.git
   cd datamine-first-day
```
1. Install the Python dependencies:
``` bash
   pip install -r requirements.txt
```
The required libraries include:
- `ultralytics` (for YOLOv8)
- `opencv-python` (for video capture and display)
- `torch`, `transformers` (for Segformer-based facial segmentation)
- `Pillow`, `numpy`, `torchvision` (for general image processing tasks)

## **Usage**
### Run the Scripts:
1. **Real-Time Object Detection with YOLOv8**
    - Run the `yolo.py` script:
``` bash
     python yolo.py
```
- The webcam will activate, and you'll start seeing annotated objects in the live video feed.

1. **Facial Segmentation with Segformer**
    - Run the `segmentation.py` script:
``` bash
     python segmentation.py
```
- A window will open showing the segmented facial regions with overlaid colors. The top-left corner will display a **legend** for each segment class.

## **Technical Features**
### **YOLOv8 (Object Detection)**
- **YOLOv8 Model Loading:**
The model uses a pre-trained network (`yolov8n-oiv7.pt`) optimized for real-time object detection.
- **Continuous Live Feed Process:**
    - Captures frames from the webcam using OpenCV.
    - Processes each frame with `model.predict()`.
    - Visualizes results on the live feed with annotations for detected objects.

### **Segformer (Facial Segmentation)**
- **Pre-Trained Model**:
Utilizes the `jonathandinu/face-parsing` model for pixel-level classification of facial features.
- **Segmentation and Color Overlay:**
    - Each pixel is mapped to a specific class (e.g., face, lips, eyes) using pre-defined color palettes.
    - Results are blended with the original live feed for better visualization.

## **Project Structure**
``` 
datamine-first-day/
├── yolo.py                  # Script for YOLOv8 object detection
├── segmentation.py          # Script for facial segmentation with Segformer
├── requirements.txt         # File containing required Python libraries
├── README.md                # Project documentation
```
## **Common Issues**
1. **Error: "Unable to open webcam"**
Ensure your webcam is functional and not being used by another application.
2. **ImportError: Module not found**
Run the following command to install missing libraries:
``` bash
   pip install -r requirements.txt
```
1. **Slow performance or GPU crashes**
    - Ensure that your system has enough GPU memory.
    - If using the GPU leads to issues, force execution on the CPU by modifying the `device` variable in the scripts.
