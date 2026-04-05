import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# Use YOLO segmentation model instead of detection-only
model = YOLO("yolov8n-seg.pt")

def crop_to_object(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    results = model(image_path)
    boxes = results[0].boxes
    masks = results[0].masks

    if boxes is None or len(boxes) == 0 or masks is None:
        print(f"No object found in {image_path}")
        return

    # Pick the highest-confidence detection
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes[best_idx].xyxy[0])
    label = model.names[int(boxes[best_idx].cls)]
    conf = float(boxes[best_idx].conf)

    # Get the segmentation mask and resize to image dimensions
    mask = masks[best_idx].data[0].cpu().numpy()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Apply mask as alpha channel
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    # Crop to bounding box
    cropped = bgra[y1:y2, x1:x2]

    out_path = f"{name}_cropped.png"
    cv2.imwrite(out_path, cropped)
    print(f"Saved cropped '{label}' ({conf:.2f}) to {out_path}")

# Usage: python3 crop.py ball.png cube.jpg pyramid.png
for path in sys.argv[1:]:
    crop_to_object(path)
