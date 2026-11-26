import json
import random
import shutil
import sys
from pathlib import Path

# Make the project root importable when running this file directly.
ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

from configs.config import (
    ROOT_DIR,
    MANIFEST_DIR,
    MODEL_DIR,
    REPORT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAST_LAYERS,
    INITIAL_LR,
    FINE_TUNE_LR,
    MODEL_NAME,
)


# --------------------------------------------------
# 1. Reproducibility and GPU configuration
# --------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU memory-growth setting:", exc)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus if gpus else "None detected; training will use CPU")

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------
# 2. Load and validate CSV manifests
# --------------------------------------------------
def load_manifest(split):
    csv_path = MANIFEST_DIR / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run: python src/prepare_data.py"
        )

    df = pd.read_csv(csv_path)
    required = {"image_path", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain {required}")

    if not (df["split"] == split).all():
        raise ValueError(f"Unexpected split value in {csv_path}")

    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {csv_path}")

    df["absolute_path"] = df["image_path"].map(
        lambda p: str((ROOT_DIR / p).resolve())
    )

    missing = [
        p for p in df["absolute_path"]
        if not Path(p).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing. First: {missing[0]}"
        )

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"{split} must contain both classes.")

    print(f"\n{split.upper()} images: {len(df)}")
    print(df["label"].map({
        0: "non_fractured",
        1: "fractured"
    }).value_counts().to_dict())

    return df


train_df = load_manifest("train")
valid_df = load_manifest("valid")
test_df = load_manifest("test")

# Prevent image paths from appearing in multiple splits.
split_path_sets = [
    set(df["absolute_path"]) for df in (train_df, valid_df, test_df)
]
if (
    split_path_sets[0] & split_path_sets[1]
    or split_path_sets[0] & split_path_sets[2]
    or split_path_sets[1] & split_path_sets[2]
):
    raise ValueError("Image-path overlap detected between splits.")


# --------------------------------------------------
# 3. Efficient tf.data input pipeline
# --------------------------------------------------
def decode_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(df, training=False):
    paths = df["absolute_path"].to_numpy()
    labels = df["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
valid_ds = make_dataset(valid_df)
test_ds = make_dataset(test_df)


# --------------------------------------------------
# 4. Class weights for imbalanced data
# --------------------------------------------------
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy(),
)
class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# --------------------------------------------------
# 5. Build EfficientNetB0 transfer-learning model
# --------------------------------------------------
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.03, seed=SEED),
        tf.keras.layers.RandomZoom(0.08, seed=SEED + 1),
    ],
    name="training_augmentation",
)

backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
backbone.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="xray_image")
x = augmentation(inputs)
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid", dtype="float32", name="fracture_probability"
)(x)

model = tf.keras.Model(inputs, outputs, name="bone_fracture_efficientnetb0")


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def callbacks_for(checkpoint_path, log_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path)),
    ]


stage1_path = MODEL_DIR / "efficientnetb0_stage1_best.keras"
stage2_path = MODEL_DIR / "efficientnetb0_finetune_best.keras"

print("\n=== STAGE 1: TRAIN CLASSIFIER ===")
compile_model(INITIAL_LR)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_for(
        stage1_path, REPORT_DIR / "training_stage1.csv"
    ),
)

# Reload the best checkpoint before fine-tuning.
model = tf.keras.models.load_model(stage1_path)
backbone = model.get_layer("efficientnetb0")

print("\n=== STAGE 2: FINE-TUNE BACKBONE ===")
backbone.trainable = True

# Freeze most layers; fine-tune only the last few layers.
cutoff = max(0, len(backbone.layers) - FINE_TUNE_LAST_LAYERS)
for layer in backbone.layers[:cutoff]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stability with small batches.
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

compile_model(FINE_TUNE_LR)

history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    initial_epoch=0,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_for(
        stage2_path, REPORT_DIR / "training_finetune.csv"
    ),
)


# --------------------------------------------------
# 6. Select the model with the best validation ROC-AUC
# --------------------------------------------------
def validation_auc(model_path):
    candidate = tf.keras.models.load_model(model_path)
    labels = valid_df["label"].to_numpy()
    probabilities = candidate.predict(valid_ds, verbose=0).ravel()
    return float(roc_auc_score(labels, probabilities))


auc1 = validation_auc(stage1_path)
auc2 = validation_auc(stage2_path)

if auc2 >= auc1:
    best_path = stage2_path
    best_auc = auc2
else:
    best_path = stage1_path
    best_auc = auc1

final_path = MODEL_DIR / MODEL_NAME
shutil.copy2(best_path, final_path)

print("\n=== TRAINING COMPLETE ===")
print(f"Stage 1 validation ROC-AUC: {auc1:.4f}")
print(f"Stage 2 validation ROC-AUC: {auc2:.4f}")
print(f"Selected model: {best_path.name}")
print(f"Best validation ROC-AUC: {best_auc:.4f}")
print(f"Final model saved: {final_path}")


# --------------------------------------------------
# 7. Save history and plot loss/AUC curves
# --------------------------------------------------
def history_frame(history, stage):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.insert(1, "stage", stage)
    return frame


combined_history = pd.concat(
    [
        history_frame(history1, "classifier"),
        history_frame(history2, "fine_tune"),
    ],
    ignore_index=True,
)
combined_history.to_csv(
    REPORT_DIR / "training_history.csv", index=False
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for history, label in [(history1, "Classifier"), (history2, "Fine-tune")]:
    axes[0].plot(history.history["loss"], label=f"{label} train")
    axes[0].plot(history.history["val_loss"], "--", label=f"{label} val")
    axes[1].plot(history.history["auc"], label=f"{label} train")
    axes[1].plot(history.history["val_auc"], "--", label=f"{label} val")

axes[0].set_title("Training and validation loss")
axes[1].set_title("Training and validation ROC-AUC")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(REPORT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

summary = {
    "architecture": "EfficientNetB0",
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "class_weights": class_weights,
    "stage1_validation_roc_auc": auc1,
    "stage2_validation_roc_auc": auc2,
    "selected_model": best_path.name,
    "selected_validation_roc_auc": best_auc,
    "final_model": str(final_path.relative_to(ROOT_DIR)),
    "test_set_used_for_model_selection": False,
}

with (REPORT_DIR / "training_summary.json").open("w") as file:
    json.dump(summary, file, indent=2)

print("History: reports/training_history.csv")
print("Curves: reports/training_curves.png")
print("Summary: reports/training_summary.json")
print("\nThe test set was not used for model selection.")
