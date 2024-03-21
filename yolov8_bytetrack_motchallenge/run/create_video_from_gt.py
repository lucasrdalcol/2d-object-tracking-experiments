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
from yolov8_bytetrack_motchallenge.utils.common import *

generator = get_video_frames_generator(cfg.SOURCE_VIDEO_PATH)  # Video frames generator

class_id_to_label = {
    1: "pedestrian",
    2: "person on vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non motorized vehicle",
    7: "static person",
    8: "distractor",
    9: "occluder",
    10: "occluder on the ground",
    11: "occluder full",
    12: "reflection",
}

CLASS_IDS = [1]  # Pedestrians only (class 1) or "all" for all classes

box_annotator = BoxAnnotator(
    ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5
)  # Annotation settings

video_info = VideoInfo.from_video_path(cfg.SOURCE_VIDEO_PATH)  # Video info
video_tqdm = tqdm(generator, total=video_info.total_frames)

detections_by_frame = read_gt_from_file(cfg.SOURCE_DETECTIONS_FILE)

with VideoSink(
    output_path=cfg.TARGET_VIDEO_PATH, video_info=video_info
) as sink:  # Video sink
    for frame_idx, frame in enumerate(
        video_tqdm, start=1
    ):  # Iterate over frames of the video

        if frame_idx in detections_by_frame.keys():
            if detections_by_frame[frame_idx]:  # Check if there are any detections for the current frame

                xyhw_detections = np.array([detection[2:6] for detection in detections_by_frame[frame_idx]])
                xyxy_detections = xywh_to_xyxy(xyhw_detections)
                confidences = [detection[6] for detection in detections_by_frame[frame_idx]]
                class_ids = [detection[7] for detection in detections_by_frame[frame_idx]]
                trackers_id = [detection[1] for detection in detections_by_frame[frame_idx]]

                if CLASS_IDS == "all":
                    mask_classes_interest = np.ones(
                        class_ids.shape, dtype=bool
                    )
                else:
                    mask_classes_interest = np.isin(
                        class_ids, CLASS_IDS
                    )
                
                detections = Detections(
                    xyxy=np.array(xyxy_detections)[mask_classes_interest],
                    confidence=np.array(confidences)[mask_classes_interest],
                    class_id=np.array(class_ids)[mask_classes_interest],
                    tracker_id=np.array(trackers_id)[mask_classes_interest],
                )  # Convert results to Detections roboflow framework in order to plot them

                try:
                    labels = [
                        f"#{tracker_id} {class_id_to_label[class_id]} {confidence:0.2f}"
                        for _, confidence, class_id, tracker_id in detections
                    ]  # Labels for each detection
                except:
                    continue

                frame = box_annotator.annotate(
                    frame=frame, detections=detections, labels=labels
                )  # Annotate frame with detections and labels

        sink.write_frame(frame)  # Write frame to the video sink
