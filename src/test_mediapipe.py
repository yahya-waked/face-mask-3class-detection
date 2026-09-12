import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from pathlib import Path
from collections import deque, Counter
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# 1. Config & Paths
# =========================
IMG_SIZE = 224
MODEL_PATH = "models/saved_model_face_mask"   # لازم يكون فولدر SavedModel
CLASS_MAP_PATH = "models/class_mapping.json"

# تأكد من وجود الموديل قبل البدء
if not os.path.exists(MODEL_PATH):
    print(f"خطأ: فولدر الموديل غير موجود في المسار: {MODEL_PATH}")
    print("تأكد أن saved_model_face_mask هو فولدر كامل وليس ملف zip.")
    exit()

# =========================
# 2. Load SavedModel
# =========================
print("جاري تحميل الموديل... قد يستغرق ذلك ثواني...")
saved_model = tf.saved_model.load(MODEL_PATH)

if "serving_default" in saved_model.signatures:
    infer = saved_model.signatures["serving_default"]
else:
    infer = next(iter(saved_model.signatures.values()))

print("تم تحميل الموديل بنجاح!")

# =========================
# 3. Class Labels
# =========================
classes = {
    0: ("With Mask", (0, 200, 0)),
    1: ("Without Mask", (0, 0, 255)),
    2: ("Incorrect Mask", (0, 165, 255)),
}

if Path(CLASS_MAP_PATH).exists():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_to_label = json.load(f)

    # class_to_label expected like: {"with_mask": 0, "without_mask": 1, "incorrect_mask": 2}
    for class_name, idx in class_to_label.items():
        class_name = str(class_name).lower()
        idx = int(idx)

        if "without" in class_name:
            classes[idx] = ("Without Mask", (0, 0, 255))
        elif "incorrect" in class_name:
            classes[idx] = ("Incorrect Mask", (0, 165, 255))
        else:
            classes[idx] = ("With Mask", (0, 200, 0))

# =========================
# 4. MediaPipe & Camera
# =========================
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

label_history = deque(maxlen=5)

print("جاري فتح الكاميرا...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("فشل فتح الكاميرا. تأكد من عدم استخدامها في برنامج آخر.")
    exit()

prev_time = time.time()

def run_inference(face_input):
    """
    face_input: shape (1, 224, 224, 3)
    returns: 1D numpy array of probabilities
    """
    # جرّب أولًا بشكل مباشر
    try:
        output = infer(face_input)
    except Exception:
        # لو signature محتاج اسم input
        input_keys = list(getattr(infer, "structured_input_signature", (None, {}))[1].keys())
        if input_keys:
            output = infer(**{input_keys[0]: face_input})
        else:
            raise

    if isinstance(output, dict):
        preds = next(iter(output.values())).numpy()
    else:
        preds = output.numpy()

    if preds.ndim == 2:
        preds = preds[0]

    return preds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # FPS Calculation
    current_time = time.time()
    fps = 1 / max((current_time - prev_time), 1e-6)
    prev_time = current_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    # UI Header
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Face Mask Detection | FPS: {fps:.1f}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    found_face = False

    if results.detections:
        for detection in results.detections:
            found_face = True
            bbox = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            # Padding & Cropping
            pad = int(0.15 * max(bw, bh))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocessing (same as training)
            face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            face_img = preprocess_input(face_img.astype(np.float32))
            face_input = np.expand_dims(face_img, axis=0)
            face_input = tf.convert_to_tensor(face_input, dtype=tf.float32)

            # Prediction with SavedModel
            pred = run_inference(face_input)
            idx = int(np.argmax(pred))

            label_history.append(idx)
            stable_idx = Counter(label_history).most_common(1)[0][0]
            label, color = classes.get(stable_idx, ("Unknown", (255, 255, 255)))

            confidence = float(pred[stable_idx]) * 100.0

            # Drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"{label}: {confidence:.1f}%",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    if not found_face:
        cv2.putText(
            frame,
            "Searching for faces...",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow("Mask Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()