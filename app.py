from pathlib import Path
import io

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

# Resolve the folder where this script lives
FOLDER = Path(__file__).resolve().parent

# Flask app – serve static files (including index.html) from the folder
app = Flask(__name__, static_folder=str(FOLDER), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MiB upload limit

# Load YOLO model (ensure yolo11n.pt is in the same folder or give full path)
model = YOLO("yolo11n.pt")
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}

@app.get("/")
def home():
    """Serve the main UI page (index.html)."""
    return send_from_directory(FOLDER, "index.html")

@app.post("/api/detect")
def detect():
    """Receive an uploaded image, run tiled YOLO detection, and return JSON results.
    The tiling logic mirrors the original script – it creates up to 4×4 tiles
    with a 20 % overlap to catch objects on tile borders.
    """
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(error="Choose a photo first."), 400

    try:
        # Load image with Pillow and ensure RGB mode (YOLO expects 3 channels)
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception:
        return jsonify(error="Could not read that image."), 400

    width, height = image.size
    # Determine number of columns/rows (max 4, min 1) based on image size
    columns = min(4, max(1, (width + 299) // 300))
    rows = min(4, max(1, (height + 299) // 300))
    # Compute tile size with ~20 % overlap between adjacent tiles
    tile_width = min(width, round(width / (columns - 0.2 * (columns - 1))))
    tile_height = min(height, round(height / (rows - 0.2 * (rows - 1))))
    detections = []

    for row in range(rows):
        y = 0 if rows == 1 else round(row * (height - tile_height) / (rows - 1))
        for column in range(columns):
            x = 0 if columns == 1 else round(column * (width - tile_width) / (columns - 1))
            crop = image.crop((x, y, x + tile_width, y + tile_height))
            # Run YOLO on the cropped tile
            result = model.predict(crop, imgsz=960, conf=0.12, iou=0.5, verbose=False)[0]
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                if label not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidate = {
                    "class": label,
                    "bbox": [x1 + x, y1 + y, x2 - x1, y2 - y1],
                }
                # De‑duplicate overlapping boxes (IoU > 0.65 considered duplicate)
                bx, by, bw, bh = candidate["bbox"]
                duplicate = False
                for old in detections:
                    ax, ay, aw, ah = old["bbox"]
                    overlap = (
                        max(0, min(ax + aw, bx + bw) - max(ax, bx))
                        * max(0, min(ay + ah, by + bh) - max(ay, by))
                    )
                    if overlap / max(1, min(aw * ah, bw * bh)) > 0.65:
                        duplicate = True
                        break
                if not duplicate:
                    detections.append(candidate)

    return jsonify(width=width, height=height, detections=detections)

import os

if __name__ == "__main__":
    # Render provides the listening port in the $PORT env var.
    # Fallback to 8787 for local testing.
    port = int(os.getenv("PORT", 8787))
    app.run(host="0.0.0.0", port=port, debug=False)
