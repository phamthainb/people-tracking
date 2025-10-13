import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Init YOLO + DeepSORT ===
model = YOLO("yolov5s.pt")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.3,
)

# === Video input ===
cap = cv2.VideoCapture("videos/sample.mp4")
if not cap.isOpened():
    raise RuntimeError("Không mở được video nguồn")

# === Line setup (đếm ngang khung hình) ===
line_y = 300  # toạ độ đường
offset = 10  # ngưỡng cho phép lệch
counter_in = 0
counter_out = 0
memory = {}  # lưu vị trí trước đó của từng track ID

# === Loop xử lý từng frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for box, conf, cls in zip(
        results.boxes.xyxy, results.boxes.conf, results.boxes.cls
    ):
        if int(cls) == 0:  # chỉ lấy người (person)
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2, y2], float(conf), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Thu nhỏ box để hiển thị gọn
        w, h = x2 - x1, y2 - y1
        shrink = 0.1
        x1 += int(w * shrink / 2)
        y1 += int(h * shrink / 2)
        x2 -= int(w * shrink / 2)
        y2 -= int(h * shrink / 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        prev_y = memory.get(track_id, cy)
        memory[track_id] = cy

        if prev_y < line_y - offset and cy >= line_y + offset:
            counter_in += 1
            print(f"[IN] Track {track_id} qua line tại y={cy}")
        elif prev_y > line_y + offset and cy <= line_y - offset:
            counter_out += 1
            print(f"[OUT] Track {track_id} qua line tại y={cy}")

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # vẽ box person
        # cv2.putText(
        #     frame,
        #     f"ID {track_id}",
        #     (x1, y1 - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (255, 255, 0),
        #     2,
        # )
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{track_id}",
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        print(memory)

    # Vẽ line đếm
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"IN: {counter_in} | OUT: {counter_out}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Step 4 - Line Crossing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
