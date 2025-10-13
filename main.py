"""
Chạy:
  python main.py --source 0  # webcam
  python main.py --source rtsp://user:pass@camera-ip:554/stream
  python main.py --source videos/sample.mp4

"""

import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=0, help='camera index, file path or RTSP URL')
    p.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model (n/s/m/l/x)')
    p.add_argument('--conf', type=float, default=0.4, help='detection confidence threshold')
    p.add_argument('--show', action='store_true', help='show window (default False)')
    return p.parse_args()


class LineCounter:
    def __init__(self, line_position, direction='horizontal'):
        # line_position: y coordinate for horizontal line or x for vertical line
        self.line_pos = line_position
        self.direction = direction
        self.count_in = 0
        self.count_out = 0
        # track_id -> last side ('in'/'out')
        self.last_side = dict()

    def get_side(self, centroid):
        x, y = centroid
        if self.direction == 'horizontal':
            return 'in' if y > self.line_pos else 'out'
        else:
            return 'in' if x > self.line_pos else 'out'

    def update(self, track_id, centroid):
        side = self.get_side(centroid)
        prev = self.last_side.get(track_id, None)
        if prev is None:
            self.last_side[track_id] = side
            return False
        if prev != side:
            # crossing happened
            # define convention: moving from 'out' -> 'in' increments count_in
            if prev == 'out' and side == 'in':
                self.count_in += 1
            elif prev == 'in' and side == 'out':
                self.count_out += 1
            self.last_side[track_id] = side
            return True
        return False


def draw_ui(frame, dets, tracks, counter: LineCounter, fps):
    h, w = frame.shape[:2]
    # draw counting line
    if counter.direction == 'horizontal':
        cv2.line(frame, (0, counter.line_pos), (w, counter.line_pos), (0, 255, 255), 2)
    else:
        cv2.line(frame, (counter.line_pos, 0), (counter.line_pos, h), (0, 255, 255), 2)

    # draw tracks
    for t in tracks:
        track_id = t.track_id
        l, t_, r, b = map(int, t.to_tlbr())
        cx = int((l + r) / 2)
        cy = int((t_ + b) / 2)
        # bbox
        cv2.rectangle(frame, (l, t_), (r, b), (0, 255, 0), 2)
        # id
        cv2.putText(frame, f'ID {track_id}', (l, t_ - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # centroid
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # overlays: counts, FPS
    cv2.rectangle(frame, (5, 5), (260, 75), (0, 0, 0), -1)
    cv2.putText(frame, f'In: {counter.count_in}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Out: {counter.count_out}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    args = parse_args()

    # init video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit('Cannot open source: ' + str(args.source))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video resolution: {width}x{height}')

    # initialize model
    model = YOLO(args.model)

    # initialize DeepSort from deep_sort_realtime
    tracker = DeepSort(max_age=30)

    # counting line at center horizontally
    line_y = height // 2
    counter = LineCounter(line_y, direction='horizontal')

    prev_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run YOLOv8 inference (batch size 1)
        # model returns detections in xyxy pixel format by default
        results = model.predict(source=frame, conf=args.conf, classes=[0], verbose=False)  # class 0 = person in COCO
        # results is a list with one element
        r = results[0]

        detections_for_tracker = []  # list of (tlbr, confidence, class, feature) -- feature optional

        if r.boxes is not None and len(r.boxes) > 0:
            for box, prob in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                # clip to frame
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                detections_for_tracker.append(((x1, y1, x2, y2), float(prob), 'person'))

        # update tracker
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # update counting logic
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            l, t_, r, b = t.to_tlbr()
            cx = int((l + r) / 2)
            cy = int((t_ + b) / 2)
            counter.update(track_id, (cx, cy))

        # compute FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        # draw UI
        out_frame = draw_ui(frame, detections_for_tracker, tracks, counter, fps)

        # show
        if args.show:
            cv2.imshow('YOLOv8 + DeepSORT Counting', out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
