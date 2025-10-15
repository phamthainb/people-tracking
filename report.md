### **1. Giới thiệu đề tài (2–3 trang)**

* **1.1. Lý do chọn đề tài**

  * Nhu cầu thực tế trong quản lý tòa nhà, an ninh, tối ưu nhân sự.
  * So sánh với phương pháp truyền thống (bảo vệ thủ công, cảm biến hồng ngoại…).
  * Mục tiêu hướng đến tự động hóa và AI-based surveillance.

* **1.2. Mục tiêu và phạm vi đề tài**

  * Mục tiêu tổng thể: Đếm số người ra/vào bằng camera AI.
  * Các mục tiêu cụ thể:

    * Phát hiện người (YOLOv8).
    * Theo dõi người (DeepSort).
    * Nhận biết hướng di chuyển (line-crossing).
    * Ghi log In/Out + thời gian.
    * Hiển thị giao diện trực quan (UI).

* **1.3. Phạm vi ứng dụng**

  * Văn phòng, tòa nhà, trung tâm thương mại, trường học…
  * Giới hạn: Không xử lý khuôn mặt, không nhận diện danh tính.


### **2. Cơ sở lý thuyết (4–5 trang)**

* **2.1. Giới thiệu về Thị giác máy tính (Computer Vision)**

  * Khái niệm, vai trò, ứng dụng thực tế.

* **2.2. Object Detection và Tracking**

  * Phân biệt giữa detection và tracking.
  * Vấn đề “multi-object tracking”.

* **2.3. YOLOv8**

  * Giới thiệu mô hình YOLO và tiến hóa từ v1 đến v8.
  * Cấu trúc mạng YOLOv8 (Backbone – Neck – Head).
  * Ưu điểm (real-time, chính xác cao).
  * Cách sử dụng pre-trained model.

* **2.4. DeepSort Algorithm**

  * Cơ chế Kalman Filter.
  * Hungarian Algorithm cho assignment.
  * Thêm Deep Embedding giúp nhận dạng người khi bị che khuất hoặc gần nhau.

* **2.5. Line-crossing & Counting Logic**

  * Phương pháp xác định hướng di chuyển (vector, vị trí tương đối với line).
  * Cách xử lý In/Out và cập nhật bộ đếm.


### **3. Phân tích bài toán (3–4 trang)**

* **3.1. Input & Output của hệ thống**

  * Input: luồng video từ camera (hoặc file test).
  * Output:

    * Số lượng người đi vào / đi ra.
    * Log thời gian.
    * Giao diện hiển thị trực quan.

* **3.2. Luồng xử lý tổng thể (Pipeline)**

  * Sơ đồ:

    ```
    Camera Stream → YOLOv8 (detect) → DeepSort (track) → Line-cross → Counter → UI
    ```

* **3.3. Vấn đề gặp phải trong thực tế**

  * Che khuất (occlusion).
  * Ánh sáng thay đổi.
  * Nhiều người di chuyển cùng lúc.
  * Camera góc xiên hoặc rung.


### **4. Thiết kế hệ thống (4–5 trang)**

* **4.1. Kiến trúc tổng thể**

  * Mô hình hệ thống (Hình vẽ: camera – server xử lý – dashboard).

* **4.2. Mô tả các thành phần**

  * **Module Detection:** YOLOv8 model load sẵn từ `ultralytics`.
  * **Module Tracking:** DeepSort dùng `deepsort_realtime` hoặc custom embedding.
  * **Module Counting:** Logic line-cross.
  * **Module Logging:** Ghi In/Out + timestamp vào file CSV hoặc database.
  * **Module UI:** Dùng OpenCV / Streamlit / Flask hiển thị kết quả real-time.

* **4.3. Lưu đồ xử lý chi tiết**

  * Flowchart từng bước.

* **4.4. Cấu hình phần cứng – phần mềm**

  * Phần cứng: GPU (nếu có), CPU tối thiểu, camera specs.
  * Phần mềm: Python, OpenCV, Ultralytics YOLO, DeepSort, NumPy, Flask/Streamlit.


### **5. Cài đặt và triển khai (4–5 trang)**

* **5.1. Môi trường phát triển**

  * Python 3.x, CUDA/cuDNN (nếu có).
  * Các thư viện cần cài: ultralytics supervision opencv-python deep-sort-realtime pyqt5

* **5.2. Mô tả từng module chính**

  * Input video reader.
  * YOLO detection pipeline.
  * DeepSort tracker.
  * Counting logic.
  * UI và logging.

* **5.3. Giao diện người dùng (UI)**

  * Mô tả hiển thị video + bounding box + ID + In/Out count.
  * Có thể minh họa bằng ảnh chụp màn hình.


### **6. Kết quả và đánh giá (3–4 trang)**

* **6.1. Kết quả thử nghiệm**

  * Hình ảnh minh họa:

    * Phát hiện người.
    * Theo dõi ID.
    * Line crossing count.
  * Bảng thống kê số người vào/ra trong các video test.

* **6.2. Đánh giá hiệu năng**

  * FPS trung bình.
  * Độ chính xác (Accuracy, Precision, Recall).
  * Thời gian xử lý/frame.

* **6.3. Hạn chế và nguyên nhân**

  * Sai lệch khi người đứng lâu trên line.
  * Che khuất nặng → mất tracking.
  * Hạn chế về góc nhìn camera.


### **7. Kết luận và hướng phát triển (2–3 trang)**

* **7.1. Kết luận**

  * Tổng kết hệ thống đã đạt được mục tiêu đề ra.
  * Ứng dụng thực tế và khả năng mở rộng.

* **7.2. Hướng phát triển**

  * Nhận diện khuôn mặt để tránh đếm trùng.
  * Kết hợp nhiều camera (multi-view).
  * Lưu log vào database và dashboard (Grafana, Kibana).
  * Cảnh báo real-time khi vượt ngưỡng số người.


### **8. Tài liệu tham khảo (1 trang)**

* Liệt kê các nguồn:

  * Ultralytics YOLOv8 Docs.
  * DeepSort Paper (Wojke et al., 2017).
  * OpenCV documentation.
  * Một vài tài liệu nghiên cứu và blog kỹ thuật liên quan.
