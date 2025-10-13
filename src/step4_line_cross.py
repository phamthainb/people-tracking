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

# === Line setup (đếm ngang khung hình) ===
line_y = 300  # toạ độ đường
offset = 10  # ngưỡng cho phép lệch
counter_in = 0
counter_out = 0
memory = {}  # lưu vị trí trước đó của từng track ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # B1: phát hiện người (dùng code từ step3)
    results = model(frame)
    
    boxes = []
    confidences = []

    # YOLOv8 API - results is a list of Results objects
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # Only process person class (class 0 in COCO dataset)
                if cls == 0 and conf > 0.4:
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(conf))

    # B2: DeepSORT tracking (dùng code từ step3)
    detections = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        detections.append(([x1, y1, x2, y2], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    # B3: Vẽ kết quả và line crossing detection
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        # Tính center point để tracking line crossing
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Line crossing detection
        prev_y = memory.get(track_id, cy)
        memory[track_id] = cy

        if prev_y < line_y - offset and cy >= line_y + offset:
            counter_in += 1
            print(f"[IN] Track {track_id} qua line tại y={cy}")
        elif prev_y > line_y + offset and cy <= line_y - offset:
            counter_out += 1
            print(f"[OUT] Track {track_id} qua line tại y={cy}")

        # Vẽ box và text giống step3
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
        # Vẽ center point để dễ quan sát line crossing
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Vẽ line đếm
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    text = f"IN: {counter_in} | OUT: {counter_out}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    padding = 8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 30, 40

    x1 = max(x - padding, 0)
    y1 = max(y - text_h - baseline - padding, 0)
    x2 = min(x + text_w + padding, frame.shape[1])
    y2 = min(y + padding, frame.shape[0])

    # Draw black background rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    # Draw white text
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)

    cv2.imshow("Step 4 - Line Crossing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
