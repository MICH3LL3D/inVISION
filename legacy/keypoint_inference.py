import cv2
from inference import get_model

MODEL_ID = "yolov8n-pose-640"
CAMERA_INDEX = 0

print("Loading model...")
model = get_model(model_id=MODEL_ID)
print("Model loaded.")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    results = model.infer(frame)
    result = results[0] if isinstance(results, list) else results

    predictions = getattr(result, "predictions", [])

    for pred in predictions:
        x = getattr(pred, "x", None)
        y = getattr(pred, "y", None)
        w = getattr(pred, "width", None)
        h = getattr(pred, "height", None)
        conf = getattr(pred, "confidence", None)
        cls = getattr(pred, "class_name", "person")

        if all(v is not None for v in [x, y, w, h]):
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls} {conf:.2f}" if conf is not None else str(cls)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        keypoints = getattr(pred, "keypoints", [])
        for kp in keypoints:
            kx = getattr(kp, "x", None)
            ky = getattr(kp, "y", None)

            if kx is not None and ky is not None:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

    cv2.imshow("Roboflow Pose Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()