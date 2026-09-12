from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUTS_DIR / "predictions.db"


# =========================
# Connection
# =========================
def get_connection():
    return sqlite3.connect(DB_PATH)


# =========================
# Initialize database
# =========================
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


# =========================
# Insert prediction
# =========================
def save_prediction(source_type: str, label: str, confidence: float):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (timestamp, source_type, label, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_type,
            label,
            float(confidence),
        ),
    )

    conn.commit()
    conn.close()


# =========================
# Get recent predictions
# =========================
def get_recent_predictions(limit: int = 10):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, source_type, label, confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


# =========================
# Get all predictions
# =========================
def get_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, timestamp, source_type, label, confidence
        FROM predictions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


# =========================
# Get basic stats
# =========================
def get_prediction_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT label, COUNT(*) as cnt
        FROM predictions
        GROUP BY label
        ORDER BY cnt DESC
        LIMIT 1
        """
    )
    top_row = cursor.fetchone()

    top_label = top_row[0] if top_row else None
    top_count = top_row[1] if top_row else 0

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT source_type, COUNT(*) as cnt
        FROM predictions
        GROUP BY source_type
        ORDER BY cnt DESC
        LIMIT 1
        """
    )
    source_row = cursor.fetchone()

    top_source = source_row[0] if source_row else None
    top_source_count = source_row[1] if source_row else 0

    conn.close()

    return {
        "total_predictions": total,
        "top_label": top_label,
        "top_count": top_count,
        "top_source": top_source,
        "top_source_count": top_source_count,
        "avg_confidence": float(avg_conf) if avg_conf is not None else 0.0,
    }


# =========================
# Clear all predictions
# =========================
def clear_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()