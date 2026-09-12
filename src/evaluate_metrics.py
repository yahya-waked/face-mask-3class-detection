import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# Reproducibility
# =========================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw" / "FMD_DATASET"

MODEL_PATH = BASE_DIR / "models" / "saved_model_face_mask"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_METRICS = BASE_DIR / "models" / "metrics.json"
ROC_OUTPUT_PATH = OUTPUT_DIR / "roc_curve.png"

CLASS_NAMES = [
    "with_mask",
    "without_mask",
    "incorrect_mask",
]

CLASS_TO_LABEL = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IMG_SIZE = 224
BATCH_SIZE = 32

ALLOWED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================
# Collect image paths
# =========================
def collect_image_paths(data_dir, class_names):

    rows = []

    for class_name in class_names:

        class_folder = data_dir / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {class_folder}"
            )

        for file_path in class_folder.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in ALLOWED_EXTS
            ):

                rows.append(
                    {
                        "filepath": str(file_path),
                        "label_name": class_name,
                        "label": CLASS_TO_LABEL[class_name],
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No images found in: {data_dir}"
        )

    return df


# =========================
# Load + preprocess
# =========================
def load_and_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
    )

    image = tf.cast(image, tf.float32)

    image = preprocess_input(image)

    return image, label


def make_dataset(dataframe):

    paths = dataframe["filepath"].values
    labels = dataframe["label"].values

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)

    return ds


def plot_roc_curve(y_true, y_scores, output_path):
    """
    Draw ROC curve for each class and save it.
    """
    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(len(CLASS_NAMES)),
    )

    plt.figure(figsize=(8, 6))
    roc_auc_scores = {}

    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_scores[:, i],
        )
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[class_name] = float(roc_auc)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {roc_auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Guess",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close()

    return roc_auc_scores


# =========================
# Main
# =========================
def main():

    print("=" * 60)
    print("Face Mask Detection Metrics Evaluation")
    print("=" * 60)

    # =========================
    # Collect dataset
    # =========================
    df = collect_image_paths(
        DATA_DIR,
        CLASS_NAMES,
    )

    print(f"\nTotal images: {len(df)}")

    # =========================
    # Recreate SAME split
    # =========================
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    print(f"Test samples: {len(test_df)}")

    # =========================
    # Build test dataset
    # =========================
    test_ds = make_dataset(test_df)

    # =========================
    # Load trained model
    # =========================
    print("\nLoading trained model...")

    loaded_model = tf.saved_model.load(
        str(MODEL_PATH)
    )

    infer = loaded_model.signatures[
        "serving_default"
    ]

    print("Model loaded successfully!")

    # =========================
    # Predict
    # =========================
    y_true = []
    y_pred = []
    y_scores = []

    print("\nGenerating predictions...")

    for images, labels in test_ds:

        preds = infer(
            tf.convert_to_tensor(images)
        )

        preds = list(
            preds.values()
        )[0].numpy()

        pred_classes = np.argmax(
            preds,
            axis=1,
        )

        y_scores.extend(preds)

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            pred_classes
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # =========================
    # Metrics
    # =========================
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    # =========================
    # ROC Curve + AUC
    # =========================
    roc_auc_scores = plot_roc_curve(
        y_true,
        y_scores,
        ROC_OUTPUT_PATH,
    )

    # =========================
    # Print results
    # =========================
    print("\n========== METRICS ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nROC AUC per class:")
    for class_name, score in roc_auc_scores.items():
        print(f"{class_name}: {score:.4f}")

    print("\nROC curve saved successfully!")
    print(ROC_OUTPUT_PATH)

    # =========================
    # Save metrics.json
    # =========================
    metrics_data = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc_scores,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(
        OUTPUT_METRICS,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_data,
            f,
            indent=4,
        )

    print("\nmetrics.json saved successfully!")
    print(OUTPUT_METRICS)


# =========================
# Run
# =========================
if __name__ == "__main__":
    main()