import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import CLASS_NAMES, TrafficSignPredictor, image_to_data_url


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "best_model.pth"))

app = Flask(__name__)
predictor = TrafficSignPredictor(MODEL_PATH)


@app.get("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        model_name=MODEL_PATH.name,
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "classes": len(CLASS_NAMES),
    })


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read this image file."}), 400

    crop_box = None
    crop_raw = request.form.get("crop")
    if crop_raw:
        try:
            crop = json.loads(crop_raw)
            x = max(0, int(round(crop["x"])))
            y = max(0, int(round(crop["y"])))
            width = max(1, int(round(crop["width"])))
            height = max(1, int(round(crop["height"])))
            crop_box = (
                x,
                y,
                min(image.width, x + width),
                min(image.height, y + height),
            )
        except Exception:
            return jsonify({"error": "Invalid crop data."}), 400

    try:
        predictions = predictor.predict(image, crop_box=crop_box, topk=5)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    preview = image.crop(crop_box) if crop_box else image
    return jsonify({
        "prediction": predictions[0],
        "top5": predictions,
        "crop_used": crop_box is not None,
        "preview": image_to_data_url(preview.resize((224, 224))),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
