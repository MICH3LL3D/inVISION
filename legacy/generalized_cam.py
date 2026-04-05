import cv2
from ultralytics import YOLO
import torch

# Pick the best available device for Mac
if torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon GPU
else:
    device = "cpu"   # Intel Mac, or fallback if MPS isn't available

print("loading model...")
model = YOLO("yolo11n.pt")
model.to(device)

print(f"inference running on {device}")

# On macOS, CAP_AVFOUNDATION often works more reliably for the webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error opening camera")
    raise SystemExit

print("showing frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Run inference
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x_avg = (x1 + x2) // 2
            y_avg = (y1 + y2) // 2

            if label == "person":
                color = (0, 255, 0)
                text = "Person"
            elif label == "dog":
                color = (255, 0, 0)
                text = "zook"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x_avg, y_avg), 7, (0, 0, 255), -1)
            cv2.putText(
                frame,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2
            )

    cv2.imshow("Frame", frame)

    # 1 ms is better for live camera than 25 ms
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()