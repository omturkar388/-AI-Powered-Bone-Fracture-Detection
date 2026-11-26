import json
import sys
from pathlib import Path

# Allow imports from the project root when run as a script.
ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from configs.config import (
    ROOT_DIR,
    MANIFEST_DIR,
    MODEL_DIR,
    REPORT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_NAME,
)

TEST_CSV = MANIFEST_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / MODEL_NAME


def main():
    print("TensorFlow:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"Test manifest not found: {TEST_CSV}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(TEST_CSV)

    required = {"label", "image_path", "class_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing test CSV columns: {sorted(missing)}")

    if not (df["split"] == "test").all():
        raise ValueError("The manifest contains rows that are not in the test split.")

    df["label"] = df["label"].astype(int)

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Test set must contain both classes: 0 and 1.")

    paths = []
    for value in df["image_path"]:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Test image not found: {path}")

        paths.append(str(path))

    labels = df["label"].to_numpy(dtype=np.int32)

    print(f"\nTest images: {len(df)}")
    print("Class counts:")
    print(df["class_name"].value_counts().to_string())
    print(f"\nLoading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        # Keep pixel values in [0, 255], matching training input.
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("\nGenerating predictions...")
    raw_predictions = model.predict(dataset, verbose=1)
    probabilities = np.asarray(raw_predictions).reshape(-1)

    if len(probabilities) != len(labels):
        raise RuntimeError("Prediction count does not match test labels.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite prediction values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Expected sigmoid probabilities in [0, 1]. "
            "Check the model output layer before continuing."
        )

    # Fixed threshold; do not tune it on the test set.
    threshold = 0.25  # Selected using validation-set analysis
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "model": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "test_images": int(len(labels)),
        "threshold": threshold,
        "positive_class": "fractured (label 1)",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "note": (
            "Threshold fixed at 0.5. Metrics are for this held-out test split; "
            "they do not establish clinical validity."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_df = df.copy()
    predictions_df["fracture_probability"] = probabilities
    predictions_df["predicted_label"] = predictions
    predictions_df["predicted_class"] = np.where(
        predictions == 1, "fractured", "non_fractured"
    )
    predictions_df["correct"] = predictions == labels

    predictions_path = REPORT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-fractured", "Fractured"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Test Set Confusion Matrix (threshold = 0.5)")
    fig.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(labels, probabilities, ax=ax)
    ax.set_title(f"Test ROC Curve (AUC = {roc_auc:.4f})")
    fig.tight_layout()
    roc_path = REPORT_DIR / "test_roc_curve.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["non_fractured", "fractured"],
        zero_division=0,
        digits=4,
    )

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall/Sensitivity:{recall: .4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Average precision: {average_precision:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    print("=== CONFUSION MATRIX ===")
    print("Rows = actual; columns = predicted")
    print("Labels: 0 = non-fractured, 1 = fractured")
    print(cm)

    print("\n=== SAVED REPORTS ===")
    print(metrics_path)
    print(predictions_path)
    print(cm_path)
    print(roc_path)


if __name__ == "__main__":
    main()
