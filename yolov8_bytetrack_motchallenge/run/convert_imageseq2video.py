#!/usr/bin/env python3

import sys
import os

sys.path.append(os.getenv("TWODOBJECTTRACKING_ROOT"))
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Convert images to video")
parser.add_argument("--image_folder_path", type=str, help="Path to the folder containing images", default="/media/lucasrdalcol/data/phd_research/datasets/2d-object-tracking-experiments/MOT16/train/MOT16-13/img1")
parser.add_argument("--video_path", type=str, help="Path (with video extension) to save the output video", default="/media/lucasrdalcol/data/phd_research/datasets/2d-object-tracking-experiments/MOT16/train/MOT16-13/video/MOT16-13.mp4")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video (default: 30)")
parser.add_argument("--image_extension", type=str, default="jpg", help="Extension of the images (default: jpg)")
args = parser.parse_args()

frames = [img for img in sorted(os.listdir(args.image_folder_path)) if img.endswith(f".{args.image_extension}")]
img = cv2.imread(os.path.join(args.image_folder_path, frames[0]))
height, width, layers = img.shape

video_writer = cv2.VideoWriter(
    args.video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    args.fps,
    (width, height),
)

print(f"Creating video in {args.video_path} from image sequences in {args.image_folder_path}")
for frame in tqdm(frames):
    video_writer.write(cv2.imread(os.path.join(args.image_folder_path, frame)))

cv2.destroyAllWindows()
video_writer.release()
