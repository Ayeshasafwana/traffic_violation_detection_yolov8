import streamlit as st
import requests
import numpy as np
import cv2
import json
import pandas as pd
import html
from pathlib import Path
from datetime import datetime

# =========================================================
# SMART TRAFFIC AI - PREMIUM DARK NAVY FRONTEND
# Existing YOLO backend API and response format are preserved.
# =========================================================

st.set_page_config(
    page_title="Traffic Violation Detection | Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PATHS / BACKEND ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "data" / "logs.json"
EVIDENCE_DIR = BASE_DIR / "data" / "evidence"
API_URL = "http://127.0.0.1:5000"


# ---------------- DESIGN / CSS ----------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080f1e;
        --sidebar: #0c1729;
        --panel: #111f33;
        --panel2: #172942;
        --border: #263d58;
        --text: #e5edf7;
        --muted: #91a4bb;
        --cyan: #38bdf8;
        --blue: #0284c7;
        --green: #4ade80;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(14, 116, 144, .12), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 15, 30, .75);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div {
        background: var(--sidebar);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #c5d2e2;
    }

    .main .block-container {
        padding: 1.8rem 2.4rem 2.5rem;
        max-width: 1600px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    p, label {
        color: #c5d2e2;
    }

    .brand {
        font-size: 23px;
        line-height: 1.25;
        font-weight: 800;
        color: var(--cyan);
        letter-spacing: .3px;
        margin: 5px 0 5px;
    }

    .brand-sub {
        color: #71869f;
        font-size: 9px;
        letter-spacing: 1.5px;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    .sidebar-status {
        background: #102a2b;
        border: 1px solid #1d5147;
        border-radius: 10px;
        padding: 10px 12px;
        color: #86efac;
        font-size: 12px;
        font-weight: 700;
        margin-top: 14px;
    }

    .top-header {
        background: linear-gradient(115deg, #10213c 0%, #123653 55%, #07546a 100%);
        border: 1px solid #285575;
        border-radius: 18px;
        padding: 27px 30px;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, .18);
    }

    .top-header h1 {
        color: #ffffff !important;
        font-size: 28px !important;
        line-height: 1.25;
        margin: 0 0 8px 0;
    }

    .top-header p {
        color: #c1e9f8;
        font-size: 13px;
        margin: 0;
    }

    .eyebrow {
        color: #7dd3fc;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin-bottom: 9px;
    }

    .section-title {
        color: #e6eef8;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 25px 0 14px;
    }

    .metric-card {
        background: linear-gradient(145deg, #172942, #101d31);
        border: 1px solid #29435f;
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 142px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, .12);
        transition: border .2s ease, transform .2s ease;
    }

    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }

    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .metric-icon {
        font-size: 23px;
        background: rgba(56, 189, 248, .12);
        border: 1px solid rgba(56, 189, 248, .18);
        border-radius: 10px;
        padding: 7px 9px;
    }

    .metric-label {
        color: #91a4bb;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 15px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 800;
        margin-top: 4px;
    }

    .panel {
        background: linear-gradient(145deg, #13243a, #101c2e);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 19px;
        margin-bottom: 16px;
    }

    .panel-title {
        color: #f1f5f9;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 13px;
    }

    .muted {
        color: var(--muted);
        font-size: 12px;
    }

    .status-online {
        color: var(--green) !important;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(90deg, #0284c7, #0891b2);
        color: #ffffff;
        border: 1px solid #168bb7;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: .2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #0e7490);
        color: #ffffff;
        border-color: #38bdf8;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        background: #172942;
        color: #f8fafc;
        border-radius: 9px;
        border-color: #304965;
    }

    [data-testid="stFileUploader"] {
        background: #111f33;
        border: 1px dashed #38bdf8;
        border-radius: 13px;
        padding: 14px;
    }

    [data-testid="stMetric"] {
        background: #172942;
        border: 1px solid #29435f;
        border-radius: 12px;
        padding: 13px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #29435f;
        border-radius: 11px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: #111f33;
        border: 1px solid #29435f;
        border-radius: 11px;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stRadio"] > label {
        color: #91a4bb;
        font-size: 11px;
        font-weight: 700;
    }

    hr {
        border-color: #263d58;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    .footer {
        color: #71869f;
        text-align: center;
        font-size: 11px;
        padding: 18px 0 4px;
    }

    </style>
    """, unsafe_allow_html=True)


load_css()


# ---------------- HELPER FUNCTIONS ----------------
def load_logs():
    """Read traffic records from the existing JSON log file."""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        st.warning("The traffic log file could not be read.")
    return []


def metric_card(icon, label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-icon">{icon}</div>
        </div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow="SMART TRAFFIC AI  /  COMMAND CENTER"):
    st.markdown(f"""
    <div class="top-header">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_title(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def get_violation_summary(logs):
    counts = {}
    for log in logs:
        data = log.get("data", {})
        for violation in data.get("violations", []):
            if "No Violation" not in str(violation):
                counts[violation] = counts.get(violation, 0) + 1
    return counts


def call_detection_api(uploaded_file, signal):
    """Send an image to the existing Flask backend /detect endpoint."""
    response = requests.post(
        f"{API_URL}/detect",
        files={
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream"
            )
        },
        data={"signal": signal},
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def call_video_detection_api(uploaded_file, signal):
    """Upload a video to Flask and return its processing result."""
    response = requests.post(
        f"{API_URL}/detect_video",
        files={
            "video": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream"
            )
        },
        data={"signal": signal},
        timeout=3600
    )
    response.raise_for_status()
    return response.json()


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown('<div class="brand">🚦 TRAFFIC VIOLATION DETECTION</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">INTELLIGENT TRAFFIC MONITORING SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("##### MAIN MENU")

    menu = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "📸  Detection",
            "🎬  Video Upload",
            "🎥  Live Camera",
            "📋  Violation Logs",
            "🧾  E-Challan"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-status">● SYSTEM INTERFACE READY</div>',
        unsafe_allow_html=True
    )
    st.markdown("")
    st.caption("AI Traffic Monitoring Platform")
    st.caption("YOLOv8 Detection Engine")
    st.caption("Version 2.0")


# ---------------- LOAD DATA ----------------
logs = load_logs()
violation_counts = get_violation_summary(logs)
total_violations = sum(violation_counts.values())

helmet_count = sum(
    count for name, count in violation_counts.items()
    if "helmet" in name.lower()
)
signal_count = sum(
    count for name, count in violation_counts.items()
    if "signal" in name.lower()
)


# =========================================================
# DASHBOARD - SMART TRAFFIC AI PROJECT OVERVIEW
# =========================================================
if menu == "🏠  Dashboard":

    page_header(
        "Traffic Violation Detection System",
        "YOLOv8-based vehicle detection, helmet detection and number plate recognition.",
        "TRAFFIC MONITORING  /  PROJECT DASHBOARD"
    )

    # These figures are calculated only from records saved by this prototype.
    total_records = len(logs)
    plate_records = 0
    alert_records = 0
    evidence_records = 0

    for log in logs:
        data = log.get("data", {})
        plate_text = str(data.get("plate", "")).strip()
        if plate_text and plate_text.lower() not in {
            "not detected", "not detected.", "none", "unknown"
        }:
            plate_records += 1

        violations = data.get("violations", [])
        if not isinstance(violations, list):
            violations = [str(violations)]
        if any("no violation" not in str(item).lower() for item in violations):
            alert_records += 1

        if data.get("evidence_image"):
            evidence_records += 1

    section_title("PROJECT ACTIVITY")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("🗂️", "SAVED ANALYSIS RECORDS", total_records)
    with c2:
        metric_card("🔎", "PLATES READ IN SAVED RECORDS", plate_records)
    with c3:
        metric_card("⚠️", "RECORDS WITH POTENTIAL ALERTS", alert_records)
    with c4:
        metric_card("🖼️", "SAVED EVIDENCE IMAGES", evidence_records)

    st.markdown("""
    <div class="panel" style="margin-top:10px;">
        <div class="panel-title">ABOUT THIS PROJECT</div>
        <p class="muted" style="margin-bottom:0;">
            Traffic Violation Detection analyses traffic images and videos using computer vision.
            The dashboard summarises records saved by this prototype; it is not a
            live count of all vehicles on a road and does not independently verify
            a traffic offence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_title("DETECTION MODULES")
    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🚗</div>
            <div class="panel-title">Vehicle Detection</div>
            <p class="muted">Custom YOLOv8 model identifies cars, motorcycles,
            buses, trucks, bicycles and auto-rickshaws.</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🪖</div>
            <div class="panel-title">Helmet Detection</div>
            <p class="muted">A separate YOLO model detects helmet and no-helmet
            classes in the analysed image.</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🔠</div>
            <div class="panel-title">Number Plate Recognition</div>
            <p class="muted">A license-plate model locates plate regions and
            EasyOCR attempts to read the visible characters.</p>
        </div>
        """, unsafe_allow_html=True)

    section_title("WHAT THE AI SYSTEM DETECTS")
    st.markdown("""
    <div class="panel">
        <div class="panel-title">VEHICLE CATEGORIES — CUSTOM YOLOv8 MODEL</div>
        <p class="muted">The trained vehicle model is configured for these six classes:</p>
    </div>
    """, unsafe_allow_html=True)

    vehicle_cols = st.columns(6, gap="small")
    vehicle_items = [
        ("🚗", "Car"),
        ("🏍️", "Motorcycle"),
        ("🚌", "Bus"),
        ("🚚", "Truck"),
        ("🚲", "Bicycle"),
        ("🛺", "Auto-rickshaw")
    ]
    for col, (emoji, label) in zip(vehicle_cols, vehicle_items):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:105px;text-align:center;padding:14px 8px;">
                <div style="font-size:26px;">{emoji}</div>
                <div class="metric-label" style="margin-top:9px;">{label.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    detection_cols = st.columns(3, gap="large")
    with detection_cols[0]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🪖</div>
            <div class="panel-title">Helmet Detection</div>
            <p class="muted">A separate YOLO model identifies helmet and no-helmet classes in an image.</p>
        </div>
        """, unsafe_allow_html=True)
    with detection_cols[1]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🔠</div>
            <div class="panel-title">Number Plate Recognition</div>
            <p class="muted">A plate detector locates the number plate, and EasyOCR attempts to read its characters.</p>
        </div>
        """, unsafe_allow_html=True)
    with detection_cols[2]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:30px;">🚦</div>
            <div class="panel-title">Potential Violation Alert</div>
            <p class="muted">The prototype can show preliminary alerts based on its current rule logic. These are not verified legal violations.</p>
        </div>
        """, unsafe_allow_html=True)

    section_title("HOW TO USE THE SYSTEM")
    workflow_cols = st.columns(3, gap="large")
    with workflow_cols[0]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">📷</div>
            <div class="panel-title">1. Upload or Capture</div>
            <p class="muted">Choose a traffic image, upload a road video, or open Live Camera.</p>
        </div>
        """, unsafe_allow_html=True)
    with workflow_cols[1]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">🤖</div>
            <div class="panel-title">2. Run AI Detection</div>
            <p class="muted">The models process the input and display their available detection results.</p>
        </div>
        """, unsafe_allow_html=True)
    with workflow_cols[2]:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">📋</div>
            <div class="panel-title">3. Review Results</div>
            <p class="muted">Check the annotated output, number plate reading and saved records.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Prototype limitations: the helmet detector does not yet confirm which helmet "
        "belongs to which rider. Signal-jump alerts do not verify stop-line crossing. "
        "OCR readings may need manual checking."
    )

    section_title("RECENT DETECTION RECORDS")
    if logs:
        recent_rows = []
        for log in logs[-8:][::-1]:
            data = log.get("data", {})
            violations = data.get("violations", [])
            if not isinstance(violations, list):
                violations = [str(violations)]
            recent_rows.append({
                "Time": log.get("time", "N/A"),
                "Number Plate": data.get("plate", "Not detected"),
                "Potential Alerts": ", ".join(map(str, violations)) or "No alert recorded"
            })

        st.dataframe(
            pd.DataFrame(recent_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No saved records yet. Use Image Detection or Video Upload to begin.")

    section_title("PROJECT WORKFLOW")
    w1, w2, w3 = st.columns(3, gap="large")
    with w1:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">📷</div>
            <div class="panel-title">1. Provide Traffic Input</div>
            <p class="muted">Upload a road image or traffic video, or use the camera page.</p>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">🤖</div>
            <div class="panel-title">2. AI Analysis</div>
            <p class="muted">The detection models analyse vehicles, helmet classes and plate regions.</p>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="panel">
            <div style="font-size:27px;">📋</div>
            <div class="panel-title">3. Review Results</div>
            <p class="muted">Review the available output, potential alerts and saved evidence in the logs.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# IMAGE DETECTION
# =========================================================
elif menu == "📸  Detection":

    page_header(
        "Traffic Violation Detection",
        "Upload a traffic image and analyse it using the existing YOLOv8 detection engine.",
        "AI ANALYSIS  /  IMAGE DETECTION"
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">UPLOAD TRAFFIC IMAGE</div>',
                    unsafe_allow_html=True)

        signal = st.selectbox(
            "Traffic Signal Status",
            ["RED", "GREEN"]
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear road or traffic image."
        )

        if uploaded_file:
            st.image(
                uploaded_file,
                caption="Input traffic image",
                use_container_width=True
            )

            if st.button("🚨 ANALYSE TRAFFIC IMAGE", use_container_width=True):
                with st.spinner("YOLOv8 is analysing the image..."):
                    try:
                        result = call_detection_api(uploaded_file, signal)
                        st.session_state["detection_result"] = result
                        st.success("Image analysis completed.")
                    except requests.RequestException as error:
                        st.error(f"Backend connection error: {error}")
                    except (ValueError, KeyError) as error:
                        st.error(f"Could not read the backend response: {error}")

    with right:
        st.markdown('<div class="panel-title">DETECTION RESULTS</div>',
                    unsafe_allow_html=True)

        result = st.session_state.get("detection_result")

        if result:
            violations = result.get("violations", [])
            plate = result.get("plate", "Not detected")

            st.markdown("#### Vehicle information")
            st.info(f"Number Plate: {plate}")

            st.markdown("#### Violation status")
            actual_violations = [
                item for item in violations
                if "No Violation" not in str(item)
            ]

            if actual_violations:
                for violation in actual_violations:
                    st.error(f"⚠️ {violation}")
            else:
                st.info("No potential violation alert was reported by the current prototype.")

            if actual_violations:
                st.markdown("#### Preliminary violation review")
                for violation in actual_violations:
                    st.warning(f"Possible violation: {violation}")

                st.info(
                    "This is a preliminary AI alert, not a verified violation. "
                    "The current prototype does not confirm stop-line crossing. "
                    "No fine has been calculated and no official challan is issued."
                )
            else:
                st.info(
                    "The current prototype did not report a possible violation. "
                    "This does not certify that the scene is violation-free."
                )

            if result.get("image"):
                try:
                    image_bytes = bytes.fromhex(result["image"])
                    array = np.frombuffer(image_bytes, np.uint8)
                    processed_image = cv2.imdecode(array, cv2.IMREAD_COLOR)

                    if processed_image is not None:
                        st.markdown("#### Processed detection image")
                        st.image(
                            processed_image,
                            channels="BGR",
                            caption="YOLOv8 detection output",
                            use_container_width=True
                        )
                except (ValueError, TypeError):
                    st.warning("The processed image could not be displayed.")
        else:
            st.info("Upload an image and select Analyse to view detection results.")


# =========================================================
# VIDEO UPLOAD AND DETECTION
# =========================================================
elif menu == "🎬  Video Upload":

    page_header(
        "Traffic Video Detection",
        "Upload a road video and process it frame by frame using your YOLOv8 backend.",
        "AI ANALYSIS  /  VIDEO DETECTION"
    )

    st.info("Supported formats: MP4, AVI and MOV. Processing time depends on video length and your computer.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">UPLOAD TRAFFIC VIDEO</div>',
                    unsafe_allow_html=True)
        video_signal = st.selectbox(
            "Traffic Signal Status",
            ["RED", "GREEN"],
            key="video_signal"
        )
        uploaded_video = st.file_uploader(
            "Choose a traffic video",
            type=["mp4", "avi", "mov"],
            key="traffic_video",
            help="Choose a video from your downloaded traffic dataset."
        )

        if uploaded_video:
            st.video(uploaded_video)
            st.caption(f"Selected file: {uploaded_video.name} · {uploaded_video.size / (1024 * 1024):.2f} MB")

            if st.button("🚦 START VIDEO DETECTION", use_container_width=True):
                try:
                    with st.spinner("YOLOv8 is processing the video frame by frame. Please wait..."):
                        result = call_video_detection_api(uploaded_video, video_signal)
                    if result.get("status") == "success":
                        st.session_state["video_detection_result"] = result
                        st.success("Video processing completed successfully!")
                    else:
                        st.error(result.get("message", "Video processing failed."))
                except requests.Timeout:
                    st.error("Video processing took too long. Try a shorter video.")
                except requests.RequestException as error:
                    st.error(f"Backend connection error: {error}")
                except (ValueError, KeyError) as error:
                    st.error(f"Could not read the backend response: {error}")

    with right:
        st.markdown('<div class="panel-title">PROCESSED VIDEO RESULTS</div>',
                    unsafe_allow_html=True)
        video_result = st.session_state.get("video_detection_result")

        if video_result:
            st.success(video_result.get("message", "Processing completed."))
            st.metric("Frames processed", video_result.get("total_frames", 0))
            st.write("**Video FPS:**", round(video_result.get("fps", 0), 2))

            reported_violations = video_result.get("violations", [])
            st.markdown("**Violation types reported by the backend:**")
            if reported_violations:
                for item in reported_violations:
                    st.warning(str(item))
            else:
                st.info("The backend did not report any violation types for this video.")

            output_name = video_result.get("processed_video")
            if output_name:
                try:
                    safe_name = Path(output_name).name
                    video_response = requests.get(
                        f"{API_URL}/processed_video/{safe_name}",
                        timeout=300
                    )
                    video_response.raise_for_status()
                    st.markdown("**YOLOv8 processed video**")
                    st.video(video_response.content)
                    st.download_button(
                        "⬇️ DOWNLOAD PROCESSED VIDEO",
                        data=video_response.content,
                        file_name=safe_name,
                        mime="video/mp4",
                        use_container_width=True
                    )
                except requests.RequestException as error:
                    st.error(f"Could not load the processed video: {error}")
        else:
            st.info("Upload a video and click Start Video Detection to see the annotated result here.")


# =========================================================
# LIVE CAMERA
# =========================================================
elif menu == "🎥  Live Camera":

    page_header(
        "Live Traffic Monitoring",
        "Capture traffic through your connected webcam and send frames to the detection backend.",
        "LIVE MONITORING  /  CAMERA"
    )

    st.markdown('<div class="panel-title">CAMERA CONTROL</div>',
                unsafe_allow_html=True)
    st.info("Allow camera access if Windows or your browser requests permission.")

    run_camera = st.checkbox("🟢 Start Camera Monitoring")

    if run_camera:
        cap = cv2.VideoCapture(0)
        frame_window = st.empty()
        status_box = st.empty()

        if not cap.isOpened():
            st.error("Camera could not be opened. Check camera permissions and close other apps using it.")
        else:
            status_box.success("Camera connected. Processing frames...")
            try:
                while run_camera:
                    ret, frame = cap.read()

                    if not ret:
                        status_box.error("Unable to read a camera frame.")
                        break

                    encoded_ok, image_encoded = cv2.imencode(".jpg", frame)
                    if not encoded_ok:
                        status_box.error("Could not encode the camera frame.")
                        break

                    try:
                        response = requests.post(
                            f"{API_URL}/detect",
                            files={
                                "image": (
                                    "camera.jpg",
                                    image_encoded.tobytes(),
                                    "image/jpeg"
                                )
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        result = response.json()

                        if result.get("image"):
                            image_bytes = bytes.fromhex(result["image"])
                            array = np.frombuffer(image_bytes, np.uint8)
                            processed = cv2.imdecode(array, cv2.IMREAD_COLOR)

                            if processed is not None:
                                frame_window.image(
                                    processed,
                                    channels="BGR",
                                    use_container_width=True
                                )
                        else:
                            frame_window.image(
                                frame,
                                channels="BGR",
                                use_container_width=True
                            )

                    except requests.RequestException as error:
                        status_box.error(f"Backend not responding: {error}")
                        break
                    except (ValueError, KeyError):
                        status_box.error("Unexpected response from detection backend.")
                        break

            finally:
                cap.release()
                status_box.info("Camera released.")


# =========================================================
# VIOLATION LOGS
# =========================================================
elif menu == "📋  Violation Logs":

    page_header(
        "Traffic Violation Records",
        "Review recorded incidents, vehicle information and saved evidence.",
        "RECORDS  /  INCIDENT HISTORY"
    )

    if logs:
        section_title(f"RECORDED INCIDENTS  ·  {len(logs)}")

        search = st.text_input(
            "Search by vehicle number or violation",
            placeholder="Example: KA01AB1234 or Helmet"
        )

        filtered_logs = logs[::-1]

        if search.strip():
            query = search.strip().lower()
            filtered_logs = [
                log for log in filtered_logs
                if query in str(log.get("data", {}).get("plate", "")).lower()
                or query in " ".join(
                    log.get("data", {}).get("violations", [])
                ).lower()
            ]

        if not filtered_logs:
            st.info("No matching records found.")
        else:
            for index, log in enumerate(filtered_logs, start=1):
                data = log.get("data", {})
                plate = data.get("plate", "Unknown Vehicle")
                violations = data.get("violations", [])

                with st.expander(f"🚗 Incident {index}  |  {plate}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Time:**", log.get("time", "N/A"))
                        st.write("**Vehicle:**", plate)
                    with col2:
                        st.write("**Violations:**")
                        if violations:
                            for violation in violations:
                                st.write(f"• {violation}")
                        else:
                            st.write("No violation details available.")

                    evidence_file = data.get("evidence_image")
                    if evidence_file:
                        evidence_path = EVIDENCE_DIR / Path(evidence_file).name
                        if evidence_path.exists():
                            st.image(
                                str(evidence_path),
                                caption="Violation evidence",
                                use_container_width=True
                            )
                        else:
                            st.caption("Evidence image file is not available.")
    else:
        st.info("No violation logs are available yet. Run image detection to create records.")


# =========================================================
# E-CHALLAN
# =========================================================
elif menu == "🧾  E-Challan":

    page_header(
        "E-Challan Management",
        "Create a manual demonstration preview only. This prototype cannot issue official challans.",
        "TRAFFIC ENFORCEMENT  /  E-CHALLAN"
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="panel-title">DEMO CHALLAN INFORMATION</div>',
                    unsafe_allow_html=True)
        st.warning(
            "Demo only: vehicle ownership, violation evidence and legal fine are not "
            "verified by this prototype. Do not use this preview as an official challan."
        )

        with st.form("challan_form"):
            owner_name = st.text_input("Vehicle Owner Name")
            vehicle_number = st.text_input(
                "Vehicle Registration Number",
                placeholder="Example: KA 19 AB 1234"
            )

            violation_type = st.selectbox(
                "Violation Type",
                ["No Helmet", "Signal Jump"]
            )

            fine_amount = st.selectbox(
                "Fine Amount",
                ["₹500", "₹1000", "₹2000"]
            )

            submitted = st.form_submit_button(
                "GENERATE E-CHALLAN",
                use_container_width=True
            )

        if submitted:
            if owner_name.strip() and vehicle_number.strip():
                st.session_state["challan"] = {
                    "name": owner_name.strip(),
                    "vehicle": vehicle_number.strip().upper(),
                    "violation": violation_type,
                    "fine": fine_amount
                }
                st.success("Challan preview generated.")
            else:
                st.warning("Please enter both the owner name and vehicle number.")

    with col2:
        st.markdown('<div class="panel-title">CHALLAN PREVIEW</div>',
                    unsafe_allow_html=True)

        challan = st.session_state.get("challan")

        if challan:
            safe_name = html.escape(challan["name"])
            safe_vehicle = html.escape(challan["vehicle"])
            safe_violation = html.escape(challan["violation"])
            safe_fine = html.escape(challan["fine"])

            st.markdown(f"""
            <div class="panel">
                <div style="text-align:center;">
                    <div style="font-size:35px;">🚓</div>
                    <h2 style="color:#38bdf8 !important; margin-bottom:4px;">
                        TRAFFIC VIOLATION DETECTION — DEMO
                    </h2>
                    <p style="color:#fbbf24;">SAMPLE CHALLAN PREVIEW — NOT OFFICIAL</p>
                </div>
                <hr>
                <p><b>Owner Name:</b> {safe_name}</p>
                <p><b>Vehicle Number:</b> {safe_vehicle}</p>
                <p><b>Violation:</b> {safe_violation}</p>
                <hr>
                <div style="text-align:center;">
                    <p style="color:#91a4bb;">FINE AMOUNT</p>
                    <h1 style="color:#4ade80 !important;">{safe_fine}</h1>
                </div>
                <p style="text-align:center; color:#fbbf24; font-size:12px; font-weight:700;">
                    DEMONSTRATION ONLY — NOT A LEGAL OR OFFICIAL CHALLAN
                </p>
                <p style="text-align:center; color:#91a4bb; font-size:11px;">
                    Generated by Traffic Violation Detection prototype
                </p>
            </div>
            """, unsafe_allow_html=True)

            challan_text = (
                "TRAFFIC VIOLATION DETECTION — DEMO\n"
                "SAMPLE CHALLAN PREVIEW — NOT OFFICIAL\n\n"
                f"Owner Name: {challan['name']}\n"
                f"Vehicle Number: {challan['vehicle']}\n"
                f"Violation: {challan['violation']}\n"
                f"Fine: {challan['fine']}\n"
            )

            st.download_button(
                "⬇️ DOWNLOAD CHALLAN DETAILS",
                data=challan_text,
                file_name="traffic_challan.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Enter challan information and generate a preview.")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
    TRAFFIC VIOLATION DETECTION · INTELLIGENT TRAFFIC MONITORING SYSTEM<br>
    AI-Powered Vehicle & Traffic Violation Detection
</div>
""", unsafe_allow_html=True)
