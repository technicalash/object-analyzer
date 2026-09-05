# Live Object Analyzer

A real-time computer vision project that detects objects from a webcam using YOLO and displays their bounding boxes, class names, and confidence scores.

## Current Features

- Real-time webcam input using OpenCV
- YOLO-based object detection
- Bounding box visualization
- Object class identification
- Confidence score display
- Confidence threshold filtering

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO

## Current Pipeline

```text
Webcam
   ↓
OpenCV
   ↓
YOLO Object Detection
   ↓
Confidence Filtering
   ↓
Bounding Boxes
   ↓
Class Name + Confidence
```

## Current Status

This project is being developed step-by-step as a learning project in computer vision and AI.

The current version focuses on real-time YOLO object detection. Future stages will explore object tracking, trajectory analysis, movement analysis, and eventually more advanced computer vision applications.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The YOLO model weights are downloaded automatically by Ultralytics when required.

Press `q` to exit the application.
