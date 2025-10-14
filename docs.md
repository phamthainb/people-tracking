# Flow
Camera stream
     ↓
YOLOv8 (Detect person)
     ↓
DeepSort (Track ID)
SORT – thuật toán theo dõi dựa trên Kalman Filter + Hungarian Algorithm.
Deep – thêm embedding (đặc trưng khuôn mặt/cơ thể) để phân biệt người giống nhau khi họ đi gần nhau.
     ↓
Line-crossing logic
     ↓
Counter (In/Out + Timestamp)
     ↓
UI

# Các model dùng
- YOLOv8 -> dùng để detect người
- DeepSort -> dùng để tracking người