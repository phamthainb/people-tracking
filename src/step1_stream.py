import cv2
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Không mở được video nguồn")

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
    cv2.imshow("STEP1 - Camera Stream", frame)

    # delay theo fps (nếu có)
    key = cv2.waitKey(int(1000 / (fps if fps > 0 else 30)))
    if key == 27:  # ESC để thoát
        break

# Thống kê đơn giản
elapsed = time.time() - start_time
print(f"[INFO] Tổng {frame_count} frames, trung bình {frame_count/elapsed:.2f} FPS")

cap.release()
cv2.destroyAllWindows()
