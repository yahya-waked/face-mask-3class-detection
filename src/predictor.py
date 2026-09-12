"""Face mask inference engine.

This module is GUI-agnostic and can be reused by Streamlit, Flask, FastAPI,
or a plain OpenCV script.

It loads a TensorFlow SavedModel once and exposes helpers for:
- single image prediction
- annotated frame prediction
- video processing
- optional webcam loop

Expected project files:
- models/saved_model_face_mask/   (TensorFlow SavedModel folder)
- models/class_mapping.json       (optional)
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import shutil
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.utils import (
    save_image,
    generate_video_output_path,
    log_prediction,
    log_error,
)

Color = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PredictionResult:
    label: str
    class_id: int
    confidence: float
    probabilities: np.ndarray
    color: Color


@dataclass(frozen=True)
class DetectionResult:
    bbox: BBox
    prediction: PredictionResult


class FaceMaskPredictor:
    """Reusable inference engine for face-mask classification."""

    def __init__(
        self,
        model_path: str | Path = "models/saved_model_face_mask",
        class_map_path: str | Path = "models/class_mapping.json",
        img_size: int = 224,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.6,
        smoothing_window: int = 5,
        pad_ratio: float = 0.05,
        min_confidence_to_show: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.img_size = img_size
        self.use_mediapipe = use_mediapipe
        self.min_detection_confidence = min_detection_confidence
        self.label_history: Deque[int] = deque(maxlen=max(1, smoothing_window))
        self.pad_ratio = pad_ratio
        self.min_confidence_to_show = min_confidence_to_show

        self.classes: Dict[int, Tuple[str, Color]] = {
            0: ("With Mask", (0, 200, 0)),
            1: ("Without Mask", (0, 0, 255)),
            2: ("Incorrect Mask", (0, 165, 255)),
        }

        self._saved_model = None
        self._infer = None
        self._face_detection = None
        self._cascade = None

        self._validate_model_path()
        self._load_class_mapping()
        self._load_model()
        self._init_face_detector()

    def _validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SavedModel folder not found: {self.model_path}. "
                "Copy the whole folder (saved_model.pb + variables/), not a zip file."
            )

    def _load_class_mapping(self) -> None:
        if not self.class_map_path.exists():
            return

        with open(self.class_map_path, "r", encoding="utf-8") as f:
            class_to_label = json.load(f)

        for class_name, idx in class_to_label.items():
            class_name = str(class_name).lower()
            idx = int(idx)

            if "without" in class_name:
                self.classes[idx] = ("Without Mask", (0, 0, 255))
            elif "incorrect" in class_name:
                self.classes[idx] = ("Incorrect Mask", (0, 165, 255))
            else:
                self.classes[idx] = ("With Mask", (0, 200, 0))

    def _load_model(self) -> None:
        self._saved_model = tf.saved_model.load(str(self.model_path))

        if hasattr(self._saved_model, "signatures") and self._saved_model.signatures:
            if "serving_default" in self._saved_model.signatures:
                self._infer = self._saved_model.signatures["serving_default"]
            else:
                self._infer = next(iter(self._saved_model.signatures.values()))
        else:
            raise RuntimeError("Loaded model has no signatures.")

    def _init_face_detector(self) -> None:
        """Initialize MediaPipe if available; otherwise fall back to Haar cascade."""
        if self.use_mediapipe:
            try:
                import mediapipe as mp  # local import to avoid hard dependency issues

                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_detection_confidence,
                )
                return
            except Exception:
                self._face_detection = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            self._cascade = cv2.CascadeClassifier(str(cascade_path))

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop received.")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (self.img_size, self.img_size))
        face_rgb = face_rgb.astype(np.float32)
        face_rgb = preprocess_input(face_rgb)
        face_rgb = np.expand_dims(face_rgb, axis=0)
        return face_rgb

    def _signature_input_name(self) -> Optional[str]:
        try:
            structured = self._infer.structured_input_signature  # type: ignore[attr-defined]
            kwargs = structured[1]
            if kwargs:
                return next(iter(kwargs.keys()))
        except Exception:
            pass
        return None

    def _run_inference(self, face_input: np.ndarray | tf.Tensor) -> np.ndarray:
        tensor = tf.convert_to_tensor(face_input, dtype=tf.float32)

        try:
            output = self._infer(tensor)
        except Exception:
            input_name = self._signature_input_name()
            if input_name:
                output = self._infer(**{input_name: tensor})
            else:
                raise

        if isinstance(output, dict):
            preds = next(iter(output.values())).numpy()
        else:
            preds = output.numpy()

        if preds.ndim == 2:
            preds = preds[0]

        return preds

    def predict_face(self, face_bgr: np.ndarray) -> PredictionResult:
        face_input = self.preprocess_face(face_bgr)
        probs = self._run_inference(face_input)
        class_id = int(np.argmax(probs))
        label, color = self.classes.get(class_id, ("Unknown", (255, 255, 255)))
        confidence = float(probs[class_id])

        return PredictionResult(
            label=label,
            class_id=class_id,
            confidence=confidence,
            probabilities=probs,
            color=color,
        )

    def _detect_faces_mediapipe(self, frame_bgr: np.ndarray) -> List[BBox]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)  # type: ignore[union-attr]
        boxes: List[BBox] = []

        if results and results.detections:
            h, w = frame_bgr.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))

        return boxes

    def _detect_faces_haar(self, frame_bgr: np.ndarray) -> List[BBox]:
        if self._cascade is None:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def detect_faces(self, frame_bgr: np.ndarray) -> List[BBox]:
        if self._face_detection is not None:
            return self._detect_faces_mediapipe(frame_bgr)
        return self._detect_faces_haar(frame_bgr)

    def _expand_bbox(self, x: int, y: int, bw: int, bh: int, w: int, h: int) -> BBox:
        pad = int(self.pad_ratio * max(bw, bh))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        return x1, y1, x2, y2

    def _annotate_detection(
        self,
        frame: np.ndarray,
        bbox: BBox,
        prediction: PredictionResult,
    ) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), prediction.color, 3)
        cv2.putText(
            frame,
            f"{prediction.label}: {prediction.confidence * 100:.1f}%",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            prediction.color,
            2,
        )

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        smooth: bool = True,
        source_type: str = "frame",
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        annotated = frame_bgr.copy()
        detections: List[DetectionResult] = []
        h, w = annotated.shape[:2]

        faces = self.detect_faces(annotated)
        for x, y, bw, bh in faces:
            x1, y1, x2, y2 = self._expand_bbox(x, y, bw, bh, w, h)
            face = annotated[y1:y2, x1:x2]
            if face.size == 0:
                continue

            pred = self.predict_face(face)
            if pred.confidence < self.min_confidence_to_show:
                pred = PredictionResult(
                    label="Uncertain",
                    class_id=pred.class_id,
                    confidence=pred.confidence,
                    probabilities=pred.probabilities,
                    color=(255, 255, 0),
                )

            if smooth:
                self.label_history.append(pred.class_id)
                stable_idx = Counter(self.label_history).most_common(1)[0][0]
                stable_label, stable_color = self.classes.get(stable_idx, (pred.label, pred.color))
                stable_conf = float(pred.probabilities[stable_idx])
                final_pred = PredictionResult(
                    label=stable_label,
                    class_id=stable_idx,
                    confidence=stable_conf,
                    probabilities=pred.probabilities,
                    color=stable_color,
                )
            else:
                final_pred = pred

            self._annotate_detection(annotated, (x1, y1, x2, y2), final_pred)
            detections.append(
                DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    prediction=final_pred,
                )
            )

            log_prediction(
                source_type=source_type,
                label=final_pred.label,
                confidence=final_pred.confidence * 100.0,
            )

        return annotated, detections

    def predict_image(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Predict a still image without using smoothing history."""
        try:
            annotated, detections = self.predict_frame(
                image_bgr,
                smooth=False,
                source_type="image",
            )
            save_path = save_image(annotated, prefix="image_prediction")
            log_prediction(
                source_type="image_saved",
                label=str(save_path),
                confidence=100.0,
            )
            return annotated, detections
        except Exception as exc:
            log_error(f"IMAGE_PREDICTION_ERROR | {exc}")
            raise

    def predict_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Optional[Path]:
        """
        Predict a video and save an annotated copy.

        The final output is written as a web-friendly MP4 using ffmpeg (external
        or bundled via imageio_ffmpeg). If neither is available, the raw MP4 is kept.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_path is None:
            output_path = generate_video_output_path(prefix="video_prediction")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        raw_output_path = output_path.parent / f"{output_path.stem}_raw{output_path.suffix}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError("Could not read first frame from video.")
            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot open video writer.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, _ = self.predict_frame(frame, smooth=True, source_type="video")
                writer.write(annotated)

                if show:
                    cv2.imshow("Face Mask Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            log_prediction(
                source_type="video_saved",
                label=str(output_path),
                confidence=100.0,
            )

        except Exception as exc:
            log_error(f"VIDEO_PREDICTION_ERROR | {exc}")
            raise

        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

        # =============================================================
        # جرب ffmpeg الخارجي الأول، وبعدين imageio_ffmpeg المدمج
        # =============================================================
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            try:
                import imageio_ffmpeg
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_path = None

        if ffmpeg_path:
            converted_path = output_path.parent / f"{output_path.stem}_converted{output_path.suffix}"

            try:
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        str(raw_output_path),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(converted_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if output_path.exists():
                    output_path.unlink()

                converted_path.replace(output_path)

                if raw_output_path.exists():
                    raw_output_path.unlink()

            except Exception as conv_exc:
                log_error(f"VIDEO_REENCODE_ERROR | {conv_exc}")
                if raw_output_path.exists():
                    if output_path.exists():
                        output_path.unlink()
                    raw_output_path.replace(output_path)
        else:
            if output_path.exists():
                output_path.unlink()
            raw_output_path.replace(output_path)

        return output_path

    def reset_smoothing(self) -> None:
        self.label_history.clear()

    def camera_loop(self, camera_index: int = 0, show_fps: bool = True) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        prev_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, _ = self.predict_frame(frame, smooth=True, source_type="camera")

                if show_fps:
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

                cv2.imshow("Mask Detection System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as exc:
            log_error(f"CAMERA_ERROR | {exc}")
            raise
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    predictor = FaceMaskPredictor()
    predictor.camera_loop()