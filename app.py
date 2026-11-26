import sys
import uuid
from pathlib import Path

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import MODEL_DIR, MODEL_NAME, REPORT_DIR
from src.gradcam import load_xray, create_gradcam, save_visualizations

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB maximum

UPLOAD_DIR = ROOT_DIR / "static" / "uploads"
GRADCAM_DIR = REPORT_DIR / "gradcam"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
THRESHOLD = 0.25

MODEL_PATH = MODEL_DIR / MODEL_NAME
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# Load the model once when the Flask process starts.
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("xray")

        if not uploaded or not uploaded.filename:
            error = "Please select an X-ray image first."
        elif not allowed_file(uploaded.filename):
            error = "Please upload a PNG, JPG, or JPEG image."
        else:
            unique_name = f"{uuid.uuid4().hex}.jpg"
            image_path = UPLOAD_DIR / unique_name

            try:
                # Verify actual image contents, not only the extension.
                uploaded.stream.seek(0)
                with Image.open(uploaded.stream) as image:
                    image.verify()

                uploaded.stream.seek(0)
                with Image.open(uploaded.stream) as image:
                    image.convert("RGB").save(image_path, format="JPEG")

                original, image_batch = load_xray(image_path)
                probability, heatmap = create_gradcam(model, image_batch)

                heatmap_path, overlay_path, comparison_path, predicted_class = (
                    save_visualizations(
                        original,
                        heatmap,
                        probability,
                        THRESHOLD,
                        str(image_path),
                    )
                )

                result = {
                    "filename": secure_filename(uploaded.filename),
                    "image_url": f"/static/uploads/{unique_name}",
                    "probability": round(probability * 100, 2),
                    "predicted_class": predicted_class,
                    "threshold": THRESHOLD,
                    "heatmap_url": f"/gradcam/{heatmap_path.name}",
                    "overlay_url": f"/gradcam/{overlay_path.name}",
                    "comparison_url": f"/gradcam/{comparison_path.name}",
                }

            except (UnidentifiedImageError, OSError, ValueError):
                image_path.unlink(missing_ok=True)
                error = "The uploaded file is not a valid readable image."
            except Exception:
                app.logger.exception("Prediction or Grad-CAM failed")
                image_path.unlink(missing_ok=True)
                error = (
                    "Could not process this image. Check the terminal "
                    "for the error details and try another image."
                )

    return render_template("index.html", result=result, error=error)


@app.route("/gradcam/<path:filename>")
def gradcam_file(filename):
    # Only generated visualization files are served from this directory.
    return send_from_directory(GRADCAM_DIR, filename)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Maximum upload size is 10 MB.",
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
