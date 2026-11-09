
import cv2
import easyocr
from ultralytics import YOLO
from pathlib import Path

# Load the license plate detection model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "license_plate.pt"

plate_model = YOLO(str(MODEL_PATH))

# Load EasyOCR only once
reader = easyocr.Reader(['en'], gpu=False)


def detect_number_plate(image):
    """
    Detect a number plate using YOLO
    and read its characters using EasyOCR.
    """

    results = plate_model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    detected_plates = []

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep coordinates inside the image
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Read number plate characters
            ocr_results = reader.readtext(plate_crop)

            for item in ocr_results:
                text = item[1].strip().upper()

                if text:
                    detected_plates.append(text)

    if detected_plates:
        return max(detected_plates, key=len)

    return "Not Detected"
