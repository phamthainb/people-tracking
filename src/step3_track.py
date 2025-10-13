import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")

# Khởi tạo YOLO
model = YOLO("yolov8n.pt")

# Khởi tạo DeepSORT
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.3,
)

# Đọc video đầu vào
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Không mở được video nguồn")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # B1: phát hiện người
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    boxes = []
    confidences = []

    for *xyxy, conf, cls in detections:
        if conf < 0.4:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(conf)

    # B2: DeepSORT tracking
    detections = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        detections.append(([x1, y1, x2, y2], conf, "person"))  # tuple đúng chuẩn

    tracks = tracker.update_tracks(detections, frame=frame)

    # B3: Vẽ kết quả
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{track_id}",
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
