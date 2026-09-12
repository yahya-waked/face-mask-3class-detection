"""
FastAPI Backend
Run:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.predictor import FaceMaskPredictor

# =========================
# App
# =========================
app = FastAPI(
    title="Face Mask Detection API",
    description="Production-ready AI API for face mask detection.",
    version="1.0.0",
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "saved_model_face_mask"
CLASS_MAP_PATH = BASE_DIR / "models" / "class_mapping.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

# =========================
# Load predictor
# =========================
predictor = FaceMaskPredictor(
    model_path=str(MODEL_PATH),
    class_map_path=str(CLASS_MAP_PATH),
    use_mediapipe=True,
    smoothing_window=5,
)

# =========================
# Helpers
# =========================
def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image.")

    return image


# =========================
# Root
# =========================
@app.get("/")
def root():
    return {
        "message": "Face Mask Detection API is running."
    }


# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


# =========================
# Predict image
# =========================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        image = decode_image(contents)

        annotated, detections = predictor.predict_image(image)

        results = []

        for det in detections:
            results.append(
                {
                    "label": det.prediction.label,
                    "confidence": round(det.prediction.confidence * 100, 2),
                    "bbox": {
                        "x1": int(det.bbox[0]),
                        "y1": int(det.bbox[1]),
                        "x2": int(det.bbox[2]),
                        "y2": int(det.bbox[3]),
                    },
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "faces_detected": len(results),
                "detections": results,
            }
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


# =========================
# Metrics endpoint
# =========================
@app.get("/metrics")
def metrics():

    if not METRICS_PATH.exists():
        return {
            "success": False,
            "message": "metrics.json not found."
        }

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    return {
        "success": True,
        "metrics": metrics_data,
    }