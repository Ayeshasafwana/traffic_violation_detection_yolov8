# 🚦 Traffic Violation Detection Using YOLOv8

## Project Overview
Traffic Violation Detection Using YOLOv8 is an AI-powered web application designed to assist traffic monitoring by detecting vehicles, helmets and number plates from traffic images and videos.

The system uses Deep Learning and Computer Vision to identify different vehicle categories and display preliminary traffic violation alerts.

## Objectives
- Detect vehicles from traffic images and videos.
- Identify different types of vehicles.
- Detect helmet and no-helmet cases.
- Detect and recognise vehicle number plates.
- Display possible traffic violation alerts.
- Provide a user-friendly web dashboard.

## Technologies Used
- Python
- YOLOv8
- OpenCV
- Flask
- Streamlit
- EasyOCR
- Ultralytics

## Vehicle Categories
The custom YOLOv8 model detects six vehicle categories:
1. Car
2. Motorcycle
3. Bus
4. Truck
5. Bicycle
6. Auto-rickshaw

## Main Features
- Vehicle detection
- Helmet detection
- Number plate detection and OCR
- Traffic image and video analysis
- Preliminary violation alerts
- Dashboard and violation records

## Model Evaluation
The custom vehicle detection model was trained for 30 epochs.

Best validation results were obtained at Epoch 25:

| Metric | Result |
|---|---:|
| Precision | 65.58% |
| Recall | 46.67% |
| mAP@50 | 53.02% |
| mAP@50-95 | 37.34% |

## Project Structure
```text
traffic_violation_detection_yolov8/
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── detection.py
│   ├── number_plate.py
│   ├── violation.py
│   └── model files
├── frontend/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Flask backend
```bash
cd backend
python app.py
```

### 3. Start Streamlit frontend in a separate terminal
```bash
cd frontend
streamlit run app.py
```

### 4. Open the application
```text
http://localhost:8501
```

## Limitations
- Traffic signal state is selected manually.
- Helmet detection is not linked to individual motorcycle riders.
- Number plate recognition depends on image quality.
- Violation alerts are preliminary and are not official traffic challans.

## Future Scope
- Automatic traffic signal recognition.
- Improved vehicle detection using larger datasets.
- Rider-level helmet detection.
- Integration with real-time traffic cameras.
- Improved number plate recognition.

## Author
**Ayeshath Safuvana**

Artificial Intelligence and Machine Learning Engineering
