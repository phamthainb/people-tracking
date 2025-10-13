import cv2
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")

# Khởi tạo YOLO
model = YOLO("yolov8n.pt")
model.fuse()  # tăng tốc inference
model.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")  # sử dụng GPU nếu có
# model.to("mps")  # sử dụng GPU nếu có

# Khởi tạo DeepSORT
tracker = DeepSort(
    max_age=10,
    n_init=1,
    nn_budget=10,
    max_cosine_distance=0.3,
)

# Đọc video đầu vào với tối ưu hóa
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Không mở được video nguồn")

# Tối ưu hóa video playback
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # lấy FPS gốc hoặc default 30
frame_delay = int(1000 / fps)  # delay giữa các frame (ms)
print(f"Video FPS: {fps}, Frame delay: {frame_delay}ms")

# Tối ưu buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# === Line setup (dùng 2 điểm) ===
line_point1 = (719, 258)  # điểm đầu
line_point2 = (652, 429)  # điểm cuối
offset = 15  # ngưỡng cho phép lệch
in_tracks = set()  # lưu track ID của những người đã in
out_tracks = set()  # lưu track ID của những người đã out
memory = {}  # lưu vị trí trước đó của từng track ID

# Tính toán các tham số đường thẳng ax + by + c = 0
def calculate_line_params(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Phương trình đường thẳng: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1
    return a, b, c

# Tính khoảng cách từ điểm đến đường thẳng
def point_to_line_distance(point, line_params):
    x, y = point
    a, b, c = line_params
    return abs(a * x + b * y + c) / (a**2 + b**2)**0.5

# Xác định phía của điểm so với đường thẳng
def point_side(point, line_params):
    x, y = point
    a, b, c = line_params
    return a * x + b * y + c

line_params = calculate_line_params(line_point1, line_point2)

# Hàm vẽ thước đo trục tọa độ
def draw_rulers(frame):
    height, width = frame.shape[:2]
    
    # Vẽ thước trục X (dưới cùng)
    ruler_y = height - 30
    cv2.line(frame, (0, ruler_y), (width, ruler_y), (200, 200, 200), 1)
    
    # Vẽ các vạch chia trục X (mỗi 50px)
    for x in range(0, width, 50):
        cv2.line(frame, (x, ruler_y - 5), (x, ruler_y + 5), (200, 200, 200), 1)
        if x % 100 == 0:  # số lớn mỗi 100px
            cv2.putText(frame, str(x), (x - 10, ruler_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    # Vẽ thước trục Y (bên trái)
    ruler_x = 30
    cv2.line(frame, (ruler_x, 0), (ruler_x, height), (200, 200, 200), 1)
    
    # Vẽ các vạch chia trục Y (mỗi 50px)
    for y in range(0, height, 50):
        cv2.line(frame, (ruler_x - 5, y), (ruler_x + 5, y), (200, 200, 200), 1)
        if y % 100 == 0 and y > 0:  # số lớn mỗi 100px
            cv2.putText(frame, str(y), (5, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

# Thêm FPS counter để monitor performance
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Tính FPS thực tế
    if frame_count % 30 == 0:  # cập nhật mỗi 30 frames
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"Processing FPS: {actual_fps:.1f}")
    
    # Resize frame để xử lý nhanh hơn (tùy chọn)
    # frame = cv2.resize(frame, None, fx=0.8, fy=0.8)  # giảm 20% kích thước

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
        
        # Line crossing detection với 2D tracking
        prev_pos = memory.get(track_id, (cx, cy))
        memory[track_id] = (cx, cy)
        
        prev_x, prev_y = prev_pos

        # Xác định hướng di chuyển 2D
        direction = None
        arrow_color = (128, 128, 128)  # default gray
        
        if (cx != prev_x) or (cy != prev_y):  # có di chuyển
            # Tính vector di chuyển
            dx = cx - prev_x
            dy = cy - prev_y
            
            # Xác định hướng chính
            if abs(dx) > abs(dy):  # di chuyển ngang nhiều hơn
                if dx > 0:
                    direction = "RIGHT"
                    arrow_color = (255, 0, 255)  # magenta
                else:
                    direction = "LEFT"
                    arrow_color = (255, 255, 0)  # cyan
            else:  # di chuyển dọc nhiều hơn
                if dy > 0:
                    direction = "DOWN"
                    arrow_color = (255, 0, 0)  # blue
                else:
                    direction = "UP"
                    arrow_color = (0, 0, 255)  # red
            
            # Thêm thông tin hướng phụ nếu có di chuyển xiên đáng kể
            secondary_threshold = max(abs(dx), abs(dy)) * 0.3  # 30% của hướng chính
            if abs(dx) > secondary_threshold and abs(dy) > secondary_threshold:
                if dy > 0:
                    direction += "-DOWN"
                else:
                    direction += "-UP"

        # Tính khoảng cách đến đường thẳng
        current_distance = point_to_line_distance((cx, cy), line_params)
        prev_distance = point_to_line_distance((prev_x, prev_y), line_params)
        
        # Xác định phía của điểm (dương/âm)
        current_side = point_side((cx, cy), line_params)
        prev_side = point_side((prev_x, prev_y), line_params)
        
        # Debug thông tin để kiểm tra
        if track_id == 1:  # chỉ debug track ID 1
            print(f"Track {track_id}: prev=({prev_x},{prev_y}) cur=({cx},{cy}) dir={direction}")
            print(f"  prev_side={prev_side:.1f}, cur_side={current_side:.1f}, dist={current_distance:.1f}")
        
        # Logic line crossing với khoảng cách đến đường thẳng
        # Kiểm tra nếu có chuyển phía và đủ gần đường thẳng
        if abs(current_distance) < offset and abs(prev_distance) < offset:
            # Nếu cả hai điểm đều gần đường thẳng, kiểm tra chuyển phía
            if (prev_side > 0 and current_side <= 0) and track_id not in out_tracks:
                # Từ phía dương sang phía âm = OUT
                out_tracks.add(track_id)
                print(f"[OUT] Track {track_id}: ({prev_x},{prev_y}) → ({cx},{cy}) [{direction}]")
            elif (prev_side < 0 and current_side >= 0) and track_id not in in_tracks:
                # Từ phía âm sang phía dương = IN
                in_tracks.add(track_id)
                print(f"[IN] Track {track_id}: ({prev_x},{prev_y}) → ({cx},{cy}) [{direction}]")

        # Chọn màu box dựa trên trạng thái
        if track_id in out_tracks:
            box_color = (0, 255, 255)  # Vàng cho người đã out
            text_color = (0, 255, 255)
        else:
            box_color = (0, 255, 0)  # Xanh lá cho người bình thường
            text_color = (0, 255, 0)

        # Vẽ box và text với màu phù hợp
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame,
            f"{track_id}",
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        
        # Visualization: Hiển thị hướng di chuyển 2D
        if direction and (cx != prev_x or cy != prev_y):
            # Vẽ mũi tên theo vector thực tế
            arrow_start = (cx, cy)
            
            # Tính vector di chuyển và scale để hiển thị
            dx = cx - prev_x
            dy = cy - prev_y
            length = (dx**2 + dy**2)**0.5
            
            if length > 0:
                # Normalize và scale vector
                scale = 20  # độ dài mũi tên
                norm_dx = (dx / length) * scale
                norm_dy = (dy / length) * scale
                arrow_end = (int(cx + norm_dx), int(cy + norm_dy))
                
                cv2.arrowedLine(frame, arrow_start, arrow_end, arrow_color, 2, tipLength=0.3)
                
                # Hiển thị text direction với tốc độ
                speed = int(length)
                direction_text = f"{direction} ({speed}px)"
                cv2.putText(
                    frame,
                    direction_text,
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    arrow_color,
                    1,
                )
        
        # Vẽ center point để dễ quan sát line crossing
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Vẽ visualization cho line zones
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Vẽ thước đo trục tọa độ
    draw_rulers(frame)
    
    # Vẽ đường thẳng chính (từ 2 điểm)
    cv2.line(frame, line_point1, line_point2, (0, 0, 255), 3)
    
    # Vẽ các điểm đầu cuối của đường thẳng
    cv2.circle(frame, line_point1, 8, (0, 255, 0), -1)  # điểm đầu màu xanh
    cv2.circle(frame, line_point2, 8, (255, 0, 0), -1)  # điểm cuối màu đỏ
    
    # Vẽ tọa độ của các điểm
    cv2.putText(frame, f"P1({line_point1[0]},{line_point1[1]})", 
                (line_point1[0] + 10, line_point1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"P2({line_point2[0]},{line_point2[1]})", 
                (line_point2[0] + 10, line_point2[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Vẽ text labels cho các vùng
    mid_x = (line_point1[0] + line_point2[0]) // 2
    mid_y = (line_point1[1] + line_point2[1]) // 2
    
    cv2.putText(frame, f"CROSSING LINE (offset: {offset}px)", 
                (mid_x - 80, mid_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    text = f"IN: {len(in_tracks)} | OUT: {len(out_tracks)}"
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
    
    # Hiển thị FPS thực tế
    if frame_count > 0:
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        fps_text = f"FPS: {actual_fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Step 4 - Line Crossing", frame)
    
    # Tối ưu frame rate - sử dụng delay phù hợp với FPS gốc
    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # space để pause/resume
        cv2.waitKey(0)  # chờ đến khi nhấn phím bất kỳ
    elif key == ord("s"):  # 's' để slow motion
        frame_delay = min(frame_delay * 2, 1000)  # tăng delay (chậm lại)
        print(f"Slow motion: {frame_delay}ms delay")
    elif key == ord("f"):  # 'f' để fast forward
        frame_delay = max(frame_delay // 2, 1)  # giảm delay (nhanh lên)
        print(f"Fast forward: {frame_delay}ms delay")

cap.release()
cv2.destroyAllWindows()
