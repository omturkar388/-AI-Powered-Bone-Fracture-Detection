import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import IMAGE_SIZE, MODEL_DIR, MODEL_NAME, REPORT_DIR

DEFAULT_THRESHOLD = 0.25


def load_xray(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original = image.copy()
        resized = image.resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32)

    return original, np.expand_dims(array, axis=0)


def create_gradcam(model, image_batch):
    # EfficientNetB0 is nested inside the outer classification model.
    augmentation = model.get_layer("training_augmentation")
    backbone = model.get_layer("efficientnetb0")
    pooling = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("fracture_probability")

    conv_layer = backbone.get_layer("top_conv")

    # Build a feature extractor from EfficientNet input to its final
    # convolutional feature map and backbone output.
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        # Keep augmentation in inference mode.
        x = augmentation(image_batch, training=False)
        conv_outputs, backbone_output = feature_model(x, training=False)
        tape.watch(conv_outputs)

        pooled = pooling(backbone_output)
        dropped = dropout(pooled, training=False)
        probability = classifier(dropped)

        # Positive class is "fractured".
        score = probability[:, 0]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients. "
            "Check the model architecture and target layer."
        )

    # Average gradients across spatial dimensions.
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)

    # Weighted sum of feature maps, followed by ReLU.
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    cam = tf.nn.relu(cam)

    maximum = tf.reduce_max(cam)
    cam = tf.math.divide_no_nan(cam, maximum)
    heatmap = cam.numpy()

    return float(probability.numpy().reshape(-1)[0]), heatmap


def save_visualizations(original, heatmap, probability, threshold, image_path):
    output_dir = REPORT_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    comparison_path = output_dir / f"{stem}_comparison.png"

    # Resize the heatmap to match the original image.
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(original.size, Image.Resampling.BILINEAR)
    heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    original_array = np.asarray(original, dtype=np.uint8)
    overlay = np.uint8(
        0.55 * original_array + 0.45 * colored_heatmap
    )

    Image.fromarray(colored_heatmap).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)

    predicted_class = (
        "fractured" if probability >= threshold else "non_fractured"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_array)
    axes[0].set_title("Original X-ray")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Prediction: {predicted_class} | "
        f"Score: {probability:.2%} | Threshold: {threshold:.2f}"
    )
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return heatmap_path, overlay_path, comparison_path, predicted_class


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for an X-ray."
    )
    parser.add_argument("image", help="Path to an X-ray image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Experimental threshold (default: 0.25)",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    model_path = MODEL_DIR / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    original, image_batch = load_xray(args.image)
    probability, heatmap = create_gradcam(model, image_batch)

    heatmap_path, overlay_path, comparison_path, predicted_class = (
        save_visualizations(
            original, heatmap, probability, args.threshold, args.image
        )
    )

    print("\n=== GRAD-CAM EXPLANATION ===")
    print(f"Image: {Path(args.image).resolve()}")
    print(f"Fracture probability: {probability:.4f} ({probability:.2%})")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Predicted class: {predicted_class}")
    print(f"\nHeatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Comparison: {comparison_path}")
    print("\nGrad-CAM is an exploratory visualization, not a confirmed")
    print("fracture localization or a clinical diagnosis.")


if __name__ == "__main__":
    main()
