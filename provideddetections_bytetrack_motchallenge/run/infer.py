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
import provideddetections_bytetrack_motchallenge.config.infer_config as cfg
from provideddetections_bytetrack_motchallenge.utils.tracking import *
from provideddetections_bytetrack_motchallenge.utils.common import *

byte_tracker = BYTETracker(cfg.BYTETrackerArgs())

generator = get_video_frames_generator(cfg.SOURCE_VIDEO_PATH)  # Video frames generator

box_annotator = BoxAnnotator(
    ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5
)  # Annotation settings

video_info = VideoInfo.from_video_path(cfg.SOURCE_VIDEO_PATH)  # Video info
video_tqdm = tqdm(generator, total=video_info.total_frames)

detections_by_frame = read_detections_from_file(cfg.SOURCE_DETECTIONS_FILE)

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
                
                detections = Detections(
                    xyxy=np.array(xyxy_detections),
                    # confidence=np.array(confidences),
                    confidence=np.array([1] * len(xyxy_detections)),
                    class_id=np.array([0] * len(xyxy_detections)),
                )  # Convert results to Detections roboflow framework in order to plot them

                tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
                )
                trackers_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(trackers_id)

                # Take out the detections that are not associated with any tracker
                mask_none = np.logical_not(np.isnan(detections.tracker_id.astype(float)))

                detections.xyxy = detections.xyxy[mask_none]
                detections.confidence = detections.confidence[mask_none]
                detections.class_id = detections.class_id[mask_none]
                detections.tracker_id = detections.tracker_id[mask_none]
                xyhw_detections = xyhw_detections[mask_none]

                try:
                    labels = [
                        f"#{tracker_id} pedestrian {confidence:0.2f}"
                        for _, confidence, _, tracker_id in detections
                    ]  # Labels for each detection
                except:
                    continue

                # Get detections in MOT format and save them to a file
                detections_motformat = [
                    (
                        frame_idx,
                        tracker_id,
                        bb_x,
                        bb_y,
                        width,
                        height,
                        confidence,
                        x,
                        y,
                        z,
                    )
                    for (
                        frame_idx,
                        tracker_id,
                        (bb_x, bb_y, width, height),
                        confidence,
                        x,
                        y,
                        z,
                    ) in zip(
                        [frame_idx] * len(detections),
                        detections.tracker_id,
                        xyhw_detections,
                        [1] * len(detections),
                        [-1] * len(detections),
                        [-1] * len(detections),
                        [-1] * len(detections),
                    )
                ]

                write_mot_format(detections_motformat, cfg.DETECTIONS_OUTPUT_FILE_PATH)

                frame = box_annotator.annotate(
                    frame=frame, detections=detections, labels=labels
                )  # Annotate frame with detections and labels

        sink.write_frame(frame)  # Write frame to the video sink
