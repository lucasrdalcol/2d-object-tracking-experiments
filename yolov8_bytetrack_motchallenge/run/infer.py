#!/usr/bin/env python3

import sys
import os
from tqdm import tqdm

sys.path.append(os.getenv("TWODOBJECTTRACKING_ROOT"))
import ultralytics
from ultralytics import YOLO

# from bytetracker import BYTETracker
from yolox.tracker.byte_tracker import BYTETracker, STrack
import supervision
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
import yolov8_bytetrack_motchallenge.config.infer_config as cfg
from yolov8_bytetrack_motchallenge.utils.tracking import *

ultralytics.checks()

model = YOLO(cfg.MODEL_PATH)
model.fuse()

CLASS_NAMES_DICT = model.model.names
CLASS_IDS = [0]  # Pedestrians only (class 0) or "all" for all classes

byte_tracker = BYTETracker(cfg.BYTETrackerArgs())

generator = get_video_frames_generator(cfg.SOURCE_VIDEO_PATH)  # Video frames generator

box_annotator = BoxAnnotator(
    ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5
)  # Annotation settings

video_info = VideoInfo.from_video_path(cfg.SOURCE_VIDEO_PATH)  # Video info
video_tqdm = tqdm(generator, total=video_info.total_frames)
with VideoSink(
    output_path=cfg.TARGET_VIDEO_PATH, video_info=video_info
) as sink:  # Video sink
    for frame in video_tqdm:  # Iterate over frames of the video
        results = model.track(source=frame, persist=True, verbose=False)[0]
        video_tqdm.set_postfix(fps=1 / (results.speed["inference"] / 1000))

        boxes = results.boxes.xyxy.cpu()  # Extract boxes in the format (x1, y1, x2, y2)

        if results.boxes.id is not None:  # If there are any detections
            if CLASS_IDS == "all":
                mask_classes_interest = np.ones(
                    results.boxes.cls.cpu().numpy().astype(int).shape, dtype=bool
                )
            else:
                mask_classes_interest = np.isin(
                    results.boxes.cls.cpu().numpy().astype(int), CLASS_IDS
                )
            detections = Detections(
                xyxy=results.boxes.xyxy[mask_classes_interest].cpu().numpy(),
                confidence=results.boxes.conf[mask_classes_interest].cpu().numpy(),
                class_id=results.boxes.cls[mask_classes_interest]
                .cpu()
                .numpy()
                .astype(int),
                tracker_id=results.boxes.id[mask_classes_interest]
                .cpu()
                .numpy()
                .astype(int),
            )  # Convert results to Detections roboflow framework in order to plot them

            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]  # Labels for each detection

            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )  # Annotate frame with detections and labels

        sink.write_frame(frame)  # Write frame to the video sink
