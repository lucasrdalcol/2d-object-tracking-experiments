# Imports
from dataclasses import dataclass
import torch
import os
from supervision.geometry.dataclasses import Point

# Hyperparameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "2d-object-tracking-experiments"
PROJECT_DIR = os.path.join(os.getenv("PHD_REPOSITORIES"), PROJECT_NAME)
DATASET_DIR = os.path.join(os.getenv("PHD_DATASETS"), PROJECT_NAME)
MODELS_DIR = os.path.join(os.getenv("PHD_MODELS"), PROJECT_NAME)
RESULTS_DIR = os.path.join(os.getenv("PHD_RESULTS"), PROJECT_NAME)
SEED = 123
SOURCE_VIDEO_PATH = os.path.join(DATASET_DIR, "MOT16/train/MOT16-02/video/MOT16-02.mp4")
SOURCE_DETECTIONS_FILE = os.path.join(DATASET_DIR, "MOT16/train/MOT16-13/det/det.txt")
TARGET_VIDEO_PATH = os.path.join(RESULTS_DIR, "motchallenge_mot16/train/MOT16-02/MOT16-02-results-yolov8bytetrackconfidence.mp4")
DETECTIONS_OUTPUT_FILE_PATH = os.path.join(RESULTS_DIR, "results_trackevalrepo/trackers/mot_challenge/MOT16-train/yolov8_bytetrack_confidence/data/MOT16-02.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8x.pt")

# Byte Tracker arguments inside a class
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False