# Đề tài
Giám sát và đếm số người ra vào bằng AI

Mục tiêu: Xây dựng hệ thống dùng camera để phát hiện, theo dõi và đếm số người ra vào tòa nhà nhằm phục vụ quản lý an ninh và tối ưu hóa vận hành.

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

# Yêu cầu cho bài tập cuối kỳ
1. Báo cáo 20-30 pages 
2. Source code 
3. Presentation 3-5 slides tập trung bài toán - giải pháp - Kết quả