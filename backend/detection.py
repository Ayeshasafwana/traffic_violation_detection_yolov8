
from ultralytics import YOLO
import cv2
from pathlib import Path

# ---------------- LOAD MODELS ----------------

BASE_DIR = Path(__file__).resolve().parent

# Vehicle detection model
vehicle_model = YOLO(str(BASE_DIR / "custom_vehicle.pt"))

# Helmet detection model
helmet_model = YOLO(str(BASE_DIR / "helmet.pt"))


def detect_objects_with_boxes(image):

    annotated_frame = image.copy()
    detections = []

    # ---------------- VEHICLE DETECTION ----------------

    vehicle_results = vehicle_model.predict(
        source=image,
        conf=0.40,
        iou=0.50,
        imgsz=640,
        verbose=False
    )

    for result in vehicle_results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = vehicle_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Preserve existing vehicle class IDs
            detections.append(cls)

            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ---------------- HELMET DETECTION ----------------

    helmet_results = helmet_model.predict(
        source=image,
        conf=0.35,
        iou=0.50,
        imgsz=640,
        verbose=False
    )

    for result in helmet_results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = helmet_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_name} {conf:.2f}"

            # Different colors for helmet classes
            if class_name.lower() == "helmet":
                color = (255, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return detections, annotated_frame
