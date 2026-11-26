import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from configs.config import (
    MANIFEST_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_NAME,
)

def main():
    csv_path = MANIFEST_DIR / "valid.csv"
    model_path = MODEL_DIR / MODEL_NAME

    df = pd.read_csv(csv_path)

    if not (df["split"] == "valid").all():
        raise ValueError("Expected validation-only manifest.")

    labels = df["label"].to_numpy(dtype=np.int32)
    paths = []

    for value in df["image_path"]:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path.resolve()))

    model = tf.keras.models.load_model(model_path, compile=False)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    probabilities = model.predict(ds, verbose=1).reshape(-1)

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite model probabilities.")

    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(labels, pred),
            "precision": precision_score(
                labels, pred, zero_division=0
            ),
            "recall": recall_score(
                labels, pred, zero_division=0
            ),
            "f1": f1_score(labels, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "false_negatives": int(fn),
            "false_positives": int(fp),
        })

    results = pd.DataFrame(rows)
    out = ROOT / "reports" / "validation_threshold_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    print("\n=== VALIDATION THRESHOLD COMPARISON ===")
    print(results.round(3).to_string(index=False))

    # Illustrative target for research: recall >= 0.80.
    eligible = results[results["recall"] >= 0.80]

    print("\n=== THRESHOLD CANDIDATE ===")
    if eligible.empty:
        print(
            "No tested threshold achieved recall >= 0.80. "
            "Do not force a threshold; inspect model performance first."
        )
    else:
        # Prefer the highest threshold meeting the recall target.
        candidate = eligible.sort_values(
            "threshold", ascending=False
        ).iloc[0]
        print(candidate.to_string())
        print(
            "\nThis is a validation-only candidate, not a clinical "
            "threshold or a guarantee of future performance."
        )

    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
