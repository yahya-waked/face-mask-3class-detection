import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
DATA_DIR = Path("data/raw/FMD_DATASET")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["with_mask", "without_mask", "incorrect_mask"]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# Collect image paths
# =========================
def collect_image_paths(data_dir: Path, class_names: list[str]) -> pd.DataFrame:
    rows = []

    for class_name in class_names:
        class_folder = data_dir / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        for file_path in class_folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTS:
                rows.append(
                    {
                        "filepath": str(file_path),
                        "label_name": class_name,
                        "label": CLASS_TO_LABEL[class_name],
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found in: {data_dir}")

    return df


df = collect_image_paths(DATA_DIR, CLASS_NAMES)

print("Total images:", len(df))
print("Class counts:\n", df["label_name"].value_counts())
print("Class mapping:", CLASS_TO_LABEL)

# =========================
# Split: train / val / test
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

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# =========================
# Class weights
# =========================
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"],
)
class_weights = {int(i): float(w) for i, w in zip(np.unique(train_df["label"]), cw)}
print("Class weights:", class_weights)

# =========================
# tf.data pipeline
# =========================
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label


def make_dataset(dataframe, training=False):
    paths = dataframe["filepath"].values
    labels = dataframe["label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=min(len(dataframe), 2048), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)
test_ds = make_dataset(test_df, training=False)

# =========================
# Model: MobileNetV2 transfer learning
# =========================
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.05)(x)
x = layers.RandomZoom(0.10)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = models.Model(inputs, outputs, name="face_mask_mobilenetv2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =========================
# Callbacks
# =========================
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(MODEL_DIR / "MyTrainingModel.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# =========================
# Train stage 1
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# =========================
# Fine-tuning stage 2
# =========================
base_model.trainable = True

# Freeze early layers and fine-tune the top part only
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS + fine_tune_epochs,
    initial_epoch=history.epoch[-1] + 1 if history.epoch else 0,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# =========================
# Evaluate
# =========================
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# =========================
# Save final model + class mapping
# =========================
final_model_path = MODEL_DIR / "MyTrainingModel.keras"
model.save(final_model_path)

with open(MODEL_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(CLASS_TO_LABEL, f, ensure_ascii=False, indent=4)

print("Model trained and saved successfully!")
print("Saved model:", final_model_path)
print("Saved class mapping:", MODEL_DIR / "class_mapping.json")