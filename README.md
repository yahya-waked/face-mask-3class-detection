# 😷 Face Mask 3-Class Detection

**An end-to-end, production-minded Computer Vision system that goes beyond binary "mask / no mask" classification — detecting three real-world compliance states in real time.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

---

## 💡 Why 3 Classes Instead of 2

Most face-mask detection projects stop at a binary classification: mask or no mask. In real high-risk work environments, that's not enough — a mask worn incorrectly (below the nose, loose, etc.) is a compliance failure too, and it looks nothing like "no mask" to a naive model.

This project treats it as a genuine multi-class problem:

| Class | Meaning |
|---|---|
| ✅ **Mask** | Mask worn correctly |
| ❌ **No Mask** | No mask detected |
| ⚠️ **Incorrect Mask** | Mask worn incorrectly (e.g. below the nose) — the case that actually matters most for compliance |

## ✨ Key Features

- 🎥 **Real-time detection** — live classification directly from a webcam feed
- 🎬 **Video processing** — run the model against a full pre-recorded video and extract frame-by-frame results
- 🖼️ **Robustness** — tuned to hold up under noise, poor lighting, and a range of ages/face types, not just clean, well-lit training-style images
- 📊 **Metrics dashboard** — a built-in dashboard surfaces the confusion matrix and full evaluation metrics, so a decision-maker can actually judge whether the model is production-ready rather than trusting a single accuracy number

## 🛠 Tech Stack

`Python` · `TensorFlow` · `OpenCV` · `MediaPipe` · `Computer Vision` · `Deep Learning`

## 🏗 Project Structure

```
face-mask-3class-detection/
├── src/
│   ├── app.py               # Main dashboard / application entry point
│   ├── api.py                # API layer
│   ├── predictor.py          # Inference logic
│   ├── train.py               # Model training pipeline
│   ├── evaluate_metrics.py   # Confusion matrix & metrics generation
│   ├── db_utils.py           # Data/results persistence helpers
│   └── utils.py
├── models/
│   ├── saved_model_face_mask/  # Trained TensorFlow SavedModel
│   ├── class_mapping.json
│   └── metrics.json
├── run_app.py                 # Packaged/standalone entry point
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A webcam (for real-time mode)

### Installation
```bash
git clone https://github.com/yahya-waked/face-mask-3class-detection.git
cd face-mask-3class-detection
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the app
```bash
python src/app.py
```
This launches the main dashboard, including real-time detection and access to the evaluation metrics view.

## 🎥 Demo

**[Watch the demo on LinkedIn](https://www.linkedin.com/posts/yahya-waked_computervision-ai-deeplearning-activity-7479162999786872832-w1vC)**

---

*Built to explore what it actually takes to deploy Computer Vision in a way that reflects the messiness of real-world environments — not just a clean benchmark dataset.*