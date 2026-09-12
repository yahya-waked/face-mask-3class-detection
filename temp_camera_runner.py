
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, 'C:\\Users\\Ashmawy^_^\\Desktop\\face-mask-detection\\src')

from predictor import FaceMaskPredictor

stop_flag = Path('C:\\Users\\Ashmawy^_^\\Desktop\\face-mask-detection\\camera_stop.flag')

predictor = FaceMaskPredictor(
    model_path="models/saved_model_face_mask",
    class_map_path="models/class_mapping.json",
    use_mediapipe=True,
    smoothing_window=5,
    pad_ratio=0.05,
    min_confidence_to_show=0.0,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

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
            f"FPS: {fps:.1f}",
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
