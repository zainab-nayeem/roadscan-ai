# roadscan-ai
# RoadScan AI
Automated Road Surface Damage Severity and Risk Classification Using Computer Vision

## Tech Stack
- YOLOv8n trained on RDD2022 (47,420 images, 50 epochs)
- ONNX Runtime — CPU inference, no GPU required
- Flask REST API backend
- HTML/JavaScript frontend with voice assistant

## Results
- mAP@50: 0.571
- Inference: ~320ms per image on CPU
- Model size: ~6MB

## How to Run
1. Place yolov8_road.onnx in backend/models/
2. pip install flask flask-cors pillow numpy onnxruntime opencv-python-headless
3. cd backend && python app.py
4. Open http://localhost:5000

## By
- Zainab Nayeem (202300069)


SMIT, SMU — Department of AI&DS
