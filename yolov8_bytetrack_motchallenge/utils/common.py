def write_mot_format(detections, output_file):
    with open(output_file, 'a') as f:
        for frame_idx, tracker_id, bb_x, bb_y, width, height, confidence, x, y, z in detections:
            line = f"{frame_idx},{tracker_id},{bb_x},{bb_y},{width},{height},{confidence},{x},{y},{z}\n"
            f.write(line)