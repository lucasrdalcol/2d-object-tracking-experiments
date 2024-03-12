#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
sys.path.append(os.getenv("TWODOBJECTTRACKING_ROOT"))
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import yolov8_bytetrack_ultralytics.config.infer_config as cfg
from tqdm import tqdm

track_history = defaultdict(lambda: [])
model = YOLO(cfg.MODEL_PATH)
names = model.model.names

video_capture = cv2.VideoCapture(cfg.SOURCE_VIDEO_PATH)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
assert video_capture.isOpened(), f"Failed to open {cfg.SOURCE_VIDEO_PATH}"

width, height, fps = (
    int(video_capture.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    cfg.TARGET_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

progress_bar = iter(tqdm(range(total_frames)))
while video_capture.isOpened():
    success, frame = video_capture.read()
    if success:
        next(progress_bar)
        results = model.track(source=frame, persist=True, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu()

        if results.boxes.id is not None:
            # Extract predictions results
            classes = results.boxes.cls.cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.float().cpu().tolist()

            # Annotator instance
            annotator = Annotator(im=frame, line_width=2)

            for box, classs, track_id, conf in zip(boxes, classes, track_ids, confs):
                annotator.box_label(
                    box=box, color=colors(int(classs), True), label=f"#{track_id} {names[int(classs)]} {conf:0.2f}"
                )

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)
                
                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(classs), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(classs), True), thickness=2)

        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

video_writer.release()
video_capture.release()
cv2.destroyAllWindows()

