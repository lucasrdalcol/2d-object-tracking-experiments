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
SOURCE_VIDEO_PATH = os.path.join(DATASET_DIR, "MOT16/train/MOT16-13/video/MOT16-13.mp4")
SOURCE_DETECTIONS_FILE = os.path.join(DATASET_DIR, "/media/lucasrdalcol/data/phd_research/datasets/2d-object-tracking-experiments/data_trackevalrepo/gt/mot_challenge/MOT16-train/MOT16-13/gt/gt.txt")
TARGET_VIDEO_PATH = os.path.join(RESULTS_DIR, "motchallenge_mot16/train/MOT16-13/MOT16-13-groundtruth_pedestrians.mp4")
DETECTIONS_OUTPUT_FILE_PATH = os.path.join(RESULTS_DIR, "results_trackevalrepo/trackers/mot_challenge/MOT16-train/yolov8_botsort/data/MOT16-13.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8x.pt")
TRACKER = "bytetrack.yaml"  # "bytetrack.yaml" or "botsort.yaml"