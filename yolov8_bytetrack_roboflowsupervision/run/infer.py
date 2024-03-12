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
import yolov8_bytetrack_roboflowsupervision.config.infer_config as cfg
from yolov8_bytetrack_roboflowsupervision.utils.tracking import *

ultralytics.checks()

model = YOLO(cfg.MODEL_PATH)
model.fuse()

class_names = model.model.names

byte_tracker = BYTETracker(cfg.BYTETrackerArgs())

generator = get_video_frames_generator(cfg.SOURCE_VIDEO_PATH)  # Video frames generator
# frame = next(iter(generator))  # Get one frame from the generator
# show_frame_in_notebook(frame)  # Show the frame in the notebook

line_counter = LineCounter(start=cfg.LINE_START, end=cfg.LINE_END)

box_annotator = BoxAnnotator(
    ColorPalette(), thickness=4, text_thickness=4, text_scale=2
)  # Annotation settings
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)  # LineAnnotation settings

video_info = VideoInfo.from_video_path(cfg.SOURCE_VIDEO_PATH)  # Video info
with VideoSink(
    output_path=cfg.TARGET_VIDEO_PATH, video_info=video_info
) as sink:  # Video sink
    for frame in tqdm(
        generator, total=video_info.total_frames
    ):  # Iterate over frames of the video
        results = model(frame, verbose=False)[0]  # Inference for one frame

        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int),
        )  # Convert results to Detections roboflow framework in order to plot them

        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape,
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        labels = [
            f"#{tracker_id} {class_names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id in detections
        ]  # Labels for each detection
        
        line_counter.update(detections=detections)

        frame = box_annotator.annotate(
            frame=frame, detections=detections, labels=labels
        )  # Annotate frame with detections and labels
        line_annotator.annotate(frame=frame, line_counter=line_counter)  # Annotate frame with line counter

        sink.write_frame(frame)  # Write frame to the video sink
