import cv2
import time
import os
from ultralytics import YOLO

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")

# === LOAD MODEL ===
model = YOLO("yolov8n.pt")

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Không mở được video nguồn: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] FPS gốc: {fps:.2f}")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hết video hoặc lỗi đọc frame")
        break

    frame_count += 1

    # --- Detect ---
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label != "person":  # chỉ giữ người
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("STEP2 - YOLOv8 Person Detection", frame)

    key = cv2.waitKey(int(1000 / (fps if fps > 0 else 30)))
    if key == 27:  # ESC để thoát
        break

elapsed = time.time() - start_time
print(f"[INFO] {frame_count} frames, trung bình {frame_count / elapsed:.2f} FPS")

cap.release()
cv2.destroyAllWindows()
