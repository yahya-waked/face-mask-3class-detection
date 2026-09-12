"""
utils.py
Production utilities for:
- Logging
- Saving outputs
- Timestamp helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
import cv2


# =========================
# Project paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create folders automatically
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Logger setup
# =========================
LOG_FILE = LOGS_DIR / "prediction_logs.log"

logger = logging.getLogger("FaceMaskLogger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# =========================
# Time helpers
# =========================
def get_timestamp() -> str:
    """
    Returns timestamp string.

    Example:
        20260512_043522
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# Save image
# =========================
def save_image(image, prefix: str = "prediction") -> Path:
    """
    Save image automatically.

    Returns:
        Saved image path.
    """

    filename = f"{prefix}_{get_timestamp()}.jpg"
    output_path = IMAGES_DIR / filename

    cv2.imwrite(str(output_path), image)

    logger.info(f"IMAGE_SAVED | {output_path}")

    return output_path


# =========================
# Save video path helper
# =========================
def generate_video_output_path(prefix: str = "video") -> Path:
    """
    Generate output video path automatically.
    """

    filename = f"{prefix}_{get_timestamp()}.mp4"
    return VIDEOS_DIR / filename


# =========================
# Prediction logging
# =========================
def log_prediction(
    source_type: str,
    label: str,
    confidence: float,
):
    """
    Log prediction result.

    Example:
        IMAGE | Without Mask | 97.22%
    """

    logger.info(
        f"{source_type.upper()} | "
        f"{label} | "
        f"{confidence:.2f}%"
    )


# =========================
# Error logging
# =========================
def log_error(error_message: str):
    """
    Log errors.
    """

    logger.error(error_message)