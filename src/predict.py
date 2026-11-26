import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import IMAGE_SIZE, MODEL_DIR, MODEL_NAME

DEFAULT_THRESHOLD = 0.25
CLASS_NAMES = ["non_fractured", "fractured"]


def load_model():
    model_path = MODEL_DIR / MODEL_NAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the model first using: python src/train.py"
        )

    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


def predict_image(model, image_path, threshold):
    image_array = preprocess_image(image_path)
    output = model.predict(image_array, verbose=0)

    # This project uses a binary sigmoid output.
    probability = float(np.asarray(output).reshape(-1)[0])
    predicted_class = (
        "fractured" if probability >= threshold else "non_fractured"
    )

    return probability, predicted_class


def main():
    parser = argparse.ArgumentParser(
        description="Predict fracture probability for an X-ray image."
    )
    parser.add_argument(
        "image",
        help="Path to an X-ray image, e.g. data/raw/example.jpg",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Experimental decision threshold (default: 0.25).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    model = load_model()
    probability, predicted_class = predict_image(
        model, args.image, args.threshold
    )

    print("\n=== BONE FRACTURE DETECTION ===")
    print(f"Image: {Path(args.image).resolve()}")
    print(f"Fracture probability: {probability:.4f} ({probability:.2%})")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Predicted class: {predicted_class}")
    print("\nResearch prototype only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()
