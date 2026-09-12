"""Face Mask Detection Web App

Run locally:
    python -m streamlit run src/app.py

Project structure:
    face-mask-detection/
    ├─ models/
    │  ├─ saved_model_face_mask/
    │  └─ class_mapping.json
    ├─ outputs/
    │  ├─ images/
    │  ├─ videos/
    │  ├─ logs/
    │  └─ predictions.db
    └─ src/
       ├─ predictor.py
       ├─ utils.py
       ├─ db_utils.py
       └─ app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from db_utils import (
    init_db,
    save_prediction,
    get_recent_predictions,
    get_prediction_stats,
    get_all_predictions,
)
from predictor import FaceMaskPredictor

# =========================
# Environment
# =========================
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = OUTPUTS_DIR / "logs"
LOG_FILE = LOGS_DIR / "prediction_logs.log"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"
ROC_OUTPUT_PATH = OUTPUTS_DIR / "roc_curve.png"
CAMERA_STOP_FLAG = BASE_DIR / "camera_stop.flag"

for folder in [OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .sub-title {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: white;
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .hint {
            color: #6b7280;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Cached predictor
# =========================
@st.cache_resource(show_spinner=True)
def load_predictor() -> FaceMaskPredictor:
    return FaceMaskPredictor(
        model_path="models/saved_model_face_mask",
        class_map_path="models/class_mapping.json",
        use_mediapipe=True,
        smoothing_window=5,
        pad_ratio=0.05,
        min_confidence_to_show=0.0,
    )


predictor = load_predictor()

# Initialize database once
init_db()

# =========================
# Session state
# =========================
if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_file_key" not in st.session_state:
    st.session_state.video_file_key = None

# =========================
# Helpers
# =========================
def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bytes_to_bgr_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")
    return img_bgr


def read_uploaded_video_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    timestamp = int(time.time() * 1000)
    video_path = VIDEOS_DIR / f"uploaded_{timestamp}{suffix}"

    uploaded_file.seek(0)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return video_path


def summarize_detections(detections) -> dict:
    counts = {"With Mask": 0, "Without Mask": 0, "Incorrect Mask": 0}
    confidences = []

    for det in detections:
        label = det.prediction.label
        if label in counts:
            counts[label] += 1
        confidences.append(det.prediction.confidence)

    avg_conf = float(np.mean(confidences) * 100.0) if confidences else 0.0
    return {
        "counts": counts,
        "avg_confidence": avg_conf,
        "faces": len(detections),
    }


def record_prediction(source_type: str, label: str, confidence: float) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "label": label,
        "confidence": float(confidence),
    }

    st.session_state.recent_predictions.insert(0, entry)
    st.session_state.recent_predictions = st.session_state.recent_predictions[:10]

    try:
        save_prediction(
            source_type=source_type,
            label=label,
            confidence=confidence,
        )
    except Exception as exc:
        print(f"Database save failed: {exc}")


def load_metrics(metrics_file: Path) -> dict | None:
    if not metrics_file.exists():
        return None
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_recent_logs(log_file: Path, n: int = 20) -> list[str]:
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


def count_files(folder: Path, exts: tuple[str, ...]) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def plot_class_distribution(counts: dict) -> None:
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Prediction Distribution")
    st.pyplot(fig, clear_figure=True)


def plot_confusion_matrix(cm, class_names) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def reencode_video_python(input_path: Path, output_path: Path) -> bool:
    """
    Re-encode video using OpenCV فقط (بدون ffmpeg).
    بيحول الفيديو لـ codec مناسب للمتصفح.
    """
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # جرب avc1 الأول (H.264 في بعض البيئات)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            # fallback لـ mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()
        return output_path.exists() and output_path.stat().st_size > 0

    except Exception:
        return False


def launch_camera_process(camera_index: int) -> None:
    """Launch OpenCV camera in a separate Python process to avoid Streamlit rerun issues."""
    if CAMERA_STOP_FLAG.exists():
        try:
            CAMERA_STOP_FLAG.unlink()
        except Exception:
            pass

    camera_script = f"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, {repr(str(BASE_DIR / "src"))})

from predictor import FaceMaskPredictor

stop_flag = Path({repr(str(CAMERA_STOP_FLAG))})

predictor = FaceMaskPredictor(
    model_path="models/saved_model_face_mask",
    class_map_path="models/class_mapping.json",
    use_mediapipe=True,
    smoothing_window=5,
    pad_ratio=0.05,
    min_confidence_to_show=0.0,
)

cap = cv2.VideoCapture({int(camera_index)}, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture({int(camera_index)})

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

prev_time = time.time()

try:
    while True:
        if stop_flag.exists():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        annotated, detections = predictor.predict_frame(
            frame,
            smooth=True,
            source_type="camera",
        )

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {{fps:.1f}}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Face Mask Detection System", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        if stop_flag.exists():
            stop_flag.unlink()
    except Exception:
        pass
"""

    temp_script = BASE_DIR / "temp_camera_runner.py"
    temp_script.write_text(camera_script, encoding="utf-8")

    subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# =========================
# Sidebar
# =========================
st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Choose mode",
    ["Dashboard", "Image Upload", "Video Upload", "Live Camera"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Model path")
st.sidebar.code("models/saved_model_face_mask", language="text")

st.sidebar.caption("Classes")
for idx, (label, _) in predictor.classes.items():
    st.sidebar.write(f"{idx}: {label}")

st.sidebar.markdown("---")
st.sidebar.caption("Outputs")
st.sidebar.write(f"Images: `{IMAGES_DIR}`")
st.sidebar.write(f"Videos: `{VIDEOS_DIR}`")
st.sidebar.write(f"Logs: `{LOG_FILE}`")
st.sidebar.write(f"Metrics: `{METRICS_FILE}`")
st.sidebar.write(f"ROC: `{ROC_OUTPUT_PATH}`")

# =========================
# Header
# =========================
st.markdown('<div class="main-title">Face Mask Detection System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">Production-style AI dashboard for image, video, and live camera inference.</div>',
    unsafe_allow_html=True,
)

# =========================
# Dashboard data
# =========================
total_images_saved = count_files(IMAGES_DIR, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
total_videos_saved = count_files(VIDEOS_DIR, (".mp4", ".avi", ".mov", ".mkv", ".webm"))
recent_logs = read_recent_logs(LOG_FILE, 10)
metrics_data = load_metrics(METRICS_FILE)
db_stats = get_prediction_stats()
db_recent = get_recent_predictions(limit=10)

all_counts = {"With Mask": 0, "Without Mask": 0, "Incorrect Mask": 0}
for item in st.session_state.recent_predictions:
    if item["label"] in all_counts:
        all_counts[item["label"]] += 1

avg_conf = (
    float(np.mean([x["confidence"] for x in st.session_state.recent_predictions]))
    if st.session_state.recent_predictions
    else 0.0
)

if mode == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Saved Images", total_images_saved)
    c2.metric("Saved Videos", total_videos_saved)
    c3.metric(
        "Recent Predictions",
        len(st.session_state.recent_predictions),
    )
    c4.metric(
        "Avg Confidence",
        f"{avg_conf:.1f}%",
    )

    st.markdown(
        '<div class="section-title">Model Metrics</div>',
        unsafe_allow_html=True,
    )

    if metrics_data:
        metric_cols = st.columns(4)

        metric_cols[0].metric(
            "Accuracy",
            f"{metrics_data.get('accuracy', 0):.4f}",
        )

        metric_cols[1].metric(
            "Precision",
            f"{metrics_data.get('precision', 0):.4f}",
        )

        metric_cols[2].metric(
            "Recall",
            f"{metrics_data.get('recall', 0):.4f}",
        )

        metric_cols[3].metric(
            "F1 Score",
            f"{metrics_data.get('f1_score', 0):.4f}",
        )

        left, right = st.columns([1, 1])

        with left:
            st.markdown(
                '<div class="section-title">Confusion Matrix</div>',
                unsafe_allow_html=True,
            )

            cm = np.array(metrics_data.get("confusion_matrix", []))

            if cm.size > 0:
                class_names = [
                    label
                    for _, (label, _)
                    in sorted(predictor.classes.items())
                ]

                plot_confusion_matrix(cm, class_names)
            else:
                st.info("Confusion matrix not available.")

        with right:
            st.markdown(
                '<div class="section-title">Classification Report</div>',
                unsafe_allow_html=True,
            )

            report = metrics_data.get("classification_report", {})

            if report:
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            else:
                st.info("Classification report not available.")

        st.markdown(
            '<div class="section-title">ROC Curve</div>',
            unsafe_allow_html=True,
        )

        if ROC_OUTPUT_PATH.exists():
            st.image(str(ROC_OUTPUT_PATH), use_container_width=True)
        else:
            st.info("ROC curve image not found yet.")
    else:
        st.info("metrics.json not found yet.")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            '<div class="section-title">Prediction Distribution</div>',
            unsafe_allow_html=True,
        )

        if sum(all_counts.values()) > 0:
            plot_class_distribution(all_counts)
        else:
            st.info("No recent prediction history yet.")

    with right:
        st.markdown(
            '<div class="section-title">Recent Predictions</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.recent_predictions:
            df = pd.DataFrame(st.session_state.recent_predictions)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Run predictions first.")

    st.markdown(
        '<div class="section-title">Database Predictions</div>',
        unsafe_allow_html=True,
    )

    if db_recent:
        db_df = pd.DataFrame(
            db_recent,
            columns=["time", "source", "label", "confidence"],
        )
        st.dataframe(db_df, use_container_width=True, hide_index=True)
    else:
        st.info("No database records yet.")

    st.markdown(
        '<div class="section-title">Database Stats</div>',
        unsafe_allow_html=True,
    )

    db1, db2, db3 = st.columns(3)

    db1.metric("Total DB Predictions", db_stats["total_predictions"])
    db2.metric("Top Label", db_stats["top_label"] or "N/A")
    db3.metric("Avg Confidence (DB)", f"{db_stats['avg_confidence']:.1f}%")

    st.markdown(
        '<div class="section-title">All Database Records</div>',
        unsafe_allow_html=True,
    )

    all_db_rows = get_all_predictions()

    if all_db_rows:
        db_df_full = pd.DataFrame(
            all_db_rows,
            columns=["id", "timestamp", "source_type", "label", "confidence"],
        )
        st.dataframe(db_df_full, use_container_width=True, hide_index=True)
    else:
        st.info("No database records found yet.")

    st.markdown(
        '<div class="section-title">Logs Viewer</div>',
        unsafe_allow_html=True,
    )

    if recent_logs:
        st.code("\n".join(recent_logs), language="text")
    else:
        st.info("No logs found yet.")

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)

    s1.metric("With Mask", all_counts["With Mask"])
    s2.metric("Without Mask", all_counts["Without Mask"])
    s3.metric("Incorrect Mask", all_counts["Incorrect Mask"])

    st.stop()

# =========================
# Image mode
# =========================
if mode == "Image Upload":
    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

    if uploaded_image is None:
        st.info("Upload an image to start prediction.")
    else:
        try:
            image_bgr = bytes_to_bgr_image(uploaded_image.read())
            annotated, detections = predictor.predict_image(image_bgr)
            summary = summarize_detections(detections)

            with col1:
                st.subheader("Original")
                st.image(bgr_to_rgb(image_bgr), use_container_width=True)

            with col2:
                st.subheader("Prediction")
                st.image(bgr_to_rgb(annotated), use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faces Detected", summary["faces"])
            m2.metric("With Mask", summary["counts"]["With Mask"])
            m3.metric("Without Mask", summary["counts"]["Without Mask"])
            m4.metric("Incorrect Mask", summary["counts"]["Incorrect Mask"])

            st.caption(f"Average confidence: {summary['avg_confidence']:.1f}%")

            if detections:
                top = detections[0].prediction
                record_prediction("image", top.label, top.confidence * 100.0)
                st.success(f"Top result: {top.label} ({top.confidence * 100:.1f}%)")
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

# =========================
# Video mode
# =========================
elif mode == "Video Upload":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )

    if uploaded_video is not None:
        current_video_key = f"{uploaded_video.name}_{uploaded_video.size}"

        if st.session_state.video_file_key != current_video_key:
            st.session_state.video_path = read_uploaded_video_to_temp(uploaded_video)
            st.session_state.video_file_key = current_video_key

        video_path = st.session_state.video_path

        if video_path is None or not Path(video_path).exists():
            st.error("Video file could not be prepared.")
        else:
            st.subheader("Original Video")
            st.video(str(video_path))

            process_btn = st.button(
                "Process Video",
                type="primary",
                disabled=False,
            )

            if process_btn:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("Processing video... this may take a while."):
                        status_text.info("Reading video...")
                        progress_bar.progress(20)

                        video_path = st.session_state.video_path
                        if video_path is None or not Path(video_path).exists():
                            st.error("No valid video found in session state.")
                            st.stop()

                        output_path = VIDEOS_DIR / f"annotated_output_{int(time.time() * 1000)}.mp4"

                        predictor.reset_smoothing()

                        status_text.info("Running AI detection...")
                        progress_bar.progress(60)

                        predictor.predict_video(
                            video_path,
                            output_path=output_path,
                            show=False,
                        )

                        time.sleep(1)

                        progress_bar.progress(85)
                        status_text.info("Preparing video for browser...")

                        if not output_path.exists():
                            st.error("Output video was not created.")
                            st.stop()

                        file_size = os.path.getsize(output_path)

                        if file_size == 0:
                            st.error("Output video is empty (0 bytes).")
                            st.stop()

                        # ======================================================
                        # محاولة re-encode بـ ffmpeg الأول
                        # ======= python -m streamlit run src/app.py===============================================
                        import shutil

                        # جرب ffmpeg الخارجي الأول، وبعدين imageio_ffmpeg المدمج
                        ffmpeg_path = shutil.which("ffmpeg")
                        if not ffmpeg_path:
                            try:
                                import imageio_ffmpeg
                                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                            except Exception:
                                  ffmpeg_path = None
                        browser_ready_path = VIDEOS_DIR / f"browser_{int(time.time() * 1000)}.mp4"
                        encode_success = False

                        if ffmpeg_path:
                            try:
                                subprocess.run(
                                    [
                                        ffmpeg_path, "-y",
                                        "-i", str(output_path),
                                        "-c:v", "libx264",
                                        "-pix_fmt", "yuv420p",
                                        "-movflags", "+faststart",
                                        "-preset", "fast",
                                        str(browser_ready_path),
                                    ],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                )
                                if browser_ready_path.exists() and browser_ready_path.stat().st_size > 0:
                                    encode_success = True
                            except Exception:
                                encode_success = False

                        # ======================================================
                        # Fallback: re-encode بـ OpenCV بدون ffmpeg
                        # ======================================================
                        if not encode_success:
                            encode_success = reencode_video_python(output_path, browser_ready_path)

                        # اختار الفيديو النهائي للعرض
                        display_path = browser_ready_path if encode_success else output_path
                        final_size = os.path.getsize(display_path)

                        progress_bar.progress(100)
                        status_text.success("Processing completed.")

                        st.success("✅ Video processed successfully.")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Original Video")
                            st.video(str(video_path))

                        with col2:
                            st.subheader("Processed Video")

                            # اقرأ الـ bytes وجرب تعرضه
                            video_bytes = display_path.read_bytes()

                            try:
                                st.video(video_bytes)
                            except Exception:
                                st.warning("⚠️ Preview unavailable in browser.")

                            st.caption(f"Output size: {final_size / (1024 * 1024):.2f} MB")

                            # Download button دايمًا موجود كـ backup
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=video_bytes,
                                file_name="annotated_output.mp4",
                                mime="video/mp4",
                                type="primary",
                            )

                            if not encode_success:
                                st.info(
                                    "💡 تلميح: ثبّت ffmpeg على جهازك وأعد تشغيل الـ app "
                                    "علشان الفيديو يشتغل مباشرة في المتصفح."
                                )

                        record_prediction(
                            "video",
                            "Video Processed",
                            100.0,
                        )

                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")
    else:
        st.session_state.video_path = None
        st.session_state.video_file_key = None
        st.info("Upload a video to start prediction.")

# =========================
# Live camera mode
# =========================
else:
    st.warning("Live camera mode opens a real-time OpenCV window.")

    camera_index = st.number_input(
        "Camera index",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )

    st.markdown(
        """
        <div class="metric-card">
        <b>Instructions:</b><br>
        1) Click Start Camera<br>
        2) Real-time detection window opens<br>
        3) Press ESC to close camera
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Live Camera", type="primary"):
            launch_camera_process(int(camera_index))
            st.success("Camera started successfully.")

    with col2:
        if st.button("⛔ Stop Camera"):
            try:
                CAMERA_STOP_FLAG.write_text("stop", encoding="utf-8")
                st.info("Stop signal sent to camera.")
            except Exception as exc:
                st.error(f"Could not stop camera: {exc}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built for face mask detection using TensorFlow SavedModel + OpenCV + MediaPipe fallback.")