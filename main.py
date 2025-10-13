import sys
import cv2
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QListWidget, QAction, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv


class PeopleTrackAI(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Window setup ---
        self.setWindowTitle("People Tracking - Team 2")
        self.setGeometry(100, 100, 1024, 800)
        self.setMinimumSize(800, 600)

        # --- Core components ---
        self.model = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=30)
        self.line_zone = sv.LineZone(start=sv.Point(100, 200), end=sv.Point(500, 200))

        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2)
        self.count_in = 0
        self.count_out = 0

        self.cap = None
        self.running = False
        self.video_path = 0  # default camera

        # --- UI ---
        self.init_ui()

    def init_ui(self):
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Source")

        open_file_action = QAction("Chọn Video File", self)
        open_file_action.triggered.connect(self.choose_video_file)
        file_menu.addAction(open_file_action)

        open_cam_action = QAction("Sử dụng Camera", self)
        open_cam_action.triggered.connect(self.use_camera)
        file_menu.addAction(open_cam_action)

        # Video label
        self.video_label = QLabel("Video Preview")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #ccc; border: 1px solid #444;")

        # Object list + counter
        self.list_widget = QListWidget()
        self.counter_label = QLabel("In: 0 | Out: 0")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-weight: bold; padding: 6px;")

        # Control buttons
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_tracking)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_tracking)

        # Layout setup
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Đối tượng đang theo dõi:"))
        right_layout.addWidget(self.list_widget)
        right_layout.addWidget(self.counter_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.start_btn)
        bottom_layout.addWidget(self.stop_btn)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, 2)
        main_layout.addLayout(right_layout, 1)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(final_layout)
        self.setCentralWidget(container)

    # --- Video control ---
    def choose_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            QMessageBox.information(self, "Nguồn", f"Đã chọn video:\n{file_path}")

    def use_camera(self):
        self.video_path = 0
        QMessageBox.information(self, "Nguồn", "Đang dùng camera mặc định.")

    def start_tracking(self):
        if self.running:
            return
        self.running = True
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def stop_tracking(self):
        self.running = False

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Lỗi", "Không thể mở nguồn video.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracks = self.tracker.update_tracks(detections.xyxy, frame=frame)

            annotated_frame = frame.copy()
            self.line_annotator.annotate(annotated_frame, line_counter=self.line_zone)

            current_boxes = []
            for track in tracks:
                if not track.is_confirmed() or track.state != 2:
                    continue

                box = track.to_tlbr()
                x1, y1, x2, y2 = map(int, box)
                id = track.track_id

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID {id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # check direction crossing
                cy = (y1 + y2) // 2
                if cy < self.line_zone.start[1] - 10:
                    direction = "up"
                elif cy > self.line_zone.start[1] + 10:
                    direction = "down"
                else:
                    direction = None

                if direction == "down":
                    self.count_in += 1
                elif direction == "up":
                    self.count_out += 1

                current_boxes.append(f"ID {id} - ({x1},{y1},{x2},{y2})")

            self.update_ui_frame(annotated_frame, current_boxes)

        self.cap.release()
        self.running = False

    def update_ui_frame(self, frame, boxes):
        # update image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

        # update list
        self.list_widget.clear()
        for box in boxes:
            self.list_widget.addItem(box)

        # update counter
        self.counter_label.setText(f"In: {self.count_in} | Out: {self.count_out}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeopleTrackAI()
    window.show()
    sys.exit(app.exec_())
