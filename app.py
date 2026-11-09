from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from detection import detect_objects_with_boxes
from violation import check_violation
from database import save_log
from number_plate import detect_number_plate

app = Flask(__name__)

# ---------------- CONFIG ----------------
EVIDENCE_FOLDER = "../data/evidence"
VIDEO_FOLDER = "../data/processed_videos"

os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)


# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return "Smart Traffic AI Backend is Running Successfully!"


# ---------------- IMAGE DETECTION ROUTE ----------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image provided"
            }), 400

        file = request.files["image"]
        signal = request.form.get("signal", "RED")

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({
                "status": "error",
                "message": "Invalid image"
            }), 400

        # AI DETECTION
        detections, output_img = detect_objects_with_boxes(img)

        # VIOLATION CHECK
        violations = check_violation(detections, signal)

        # NUMBER PLATE
        plate = detect_number_plate(img)

        # SAVE EVIDENCE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{plate}_{timestamp}.jpg"
        image_path = os.path.join(EVIDENCE_FOLDER, image_filename)

        cv2.imwrite(image_path, output_img)

        # SAVE LOG
        log_data = {
            "plate": plate,
            "violations": violations,
            "signal": signal,
            "evidence_image": image_filename,
            "timestamp": timestamp
        }

        save_log(log_data)

        # ENCODE IMAGE
        success, buffer = cv2.imencode(".jpg", output_img)

        if not success:
            raise Exception("Could not encode output image")

        return jsonify({
            "status": "success",
            "plate": plate,
            "violations": violations,
            "signal": signal,
            "timestamp": timestamp,
            "evidence_image": image_filename,
            "image": buffer.tobytes().hex()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------- VIDEO DETECTION ROUTE ----------------
@app.route("/detect_video", methods=["POST"])
def detect_video():
    cap = None
    writer = None

    try:
        # CHECK VIDEO UPLOAD
        if "video" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No video uploaded"
            }), 400

        file = request.files["video"]

        if not file.filename:
            return jsonify({
                "status": "error",
                "message": "Please select a video"
            }), 400

        # ALLOWED VIDEO FORMATS
        filename = secure_filename(file.filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension not in [".mp4", ".avi", ".mov"]:
            return jsonify({
                "status": "error",
                "message": "Only MP4, AVI and MOV videos are supported"
            }), 400

        signal = request.form.get("signal", "RED")

        # SAVE TEMPORARY INPUT VIDEO
        video_id = uuid.uuid4().hex
        input_path = os.path.join(
            VIDEO_FOLDER, f"input_{video_id}{extension}"
        )
        output_filename = f"processed_{video_id}.mp4"
        output_path = os.path.join(VIDEO_FOLDER, output_filename)

        file.save(input_path)

        # OPEN VIDEO
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise Exception("Unable to open uploaded video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if width <= 0 or height <= 0:
            raise Exception("Invalid video dimensions")

        if not fps or fps <= 0:
            fps = 25.0

        # CREATE OUTPUT VIDEO
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise Exception("Unable to create processed MP4 video")

        # PROCESS VIDEO FRAME BY FRAME
        total_frames = 0
        all_violations = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # YOLO DETECTION
            detections, output_frame = detect_objects_with_boxes(frame)

            # VIOLATION CHECK
            frame_violations = check_violation(detections, signal)

            # COLLECT UNIQUE VIOLATION RESULTS
            if frame_violations:
                if isinstance(frame_violations, list):
                    current_violations = frame_violations
                else:
                    current_violations = [frame_violations]

                for violation in current_violations:
                    if violation not in all_violations:
                        all_violations.append(violation)

            # KEEP OUTPUT FRAME SIZE CONSISTENT
            if output_frame.shape[1] != width or output_frame.shape[0] != height:
                output_frame = cv2.resize(output_frame, (width, height))

            # WRITE ANNOTATED FRAME
            writer.write(output_frame)
            total_frames += 1

        # RELEASE VIDEO RESOURCES
        cap.release()
        writer.release()
        cap = None
        writer = None

        if total_frames == 0:
            raise Exception("No frames could be processed")

        # DELETE TEMPORARY INPUT
        if os.path.exists(input_path):
            os.remove(input_path)

        # RESPONSE
        return jsonify({
            "status": "success",
            "message": "Video processed successfully",
            "processed_video": output_filename,
            "video_url": f"/processed_video/{output_filename}",
            "total_frames": total_frames,
            "fps": fps,
            "violations": all_violations
        })

    except Exception as e:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------- SERVE PROCESSED VIDEO ----------------
@app.route("/processed_video/<filename>")
def get_processed_video(filename):
    return send_from_directory(
        VIDEO_FOLDER,
        filename,
        as_attachment=False
    )


# ---------------- HEALTH CHECK ----------------
@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "service": "Smart Traffic AI Backend"
    })


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
