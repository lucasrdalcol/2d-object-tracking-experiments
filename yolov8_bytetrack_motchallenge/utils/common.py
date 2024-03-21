def write_mot_format(detections, output_file):
    with open(output_file, 'a') as f:
        for frame_idx, tracker_id, bb_x, bb_y, width, height, confidence, x, y, z in detections:
            line = f"{frame_idx},{tracker_id},{bb_x},{bb_y},{width},{height},{confidence},{x},{y},{z}\n"
            f.write(line)

def read_detections_from_file(file_path):
    detections = {}
    with open(file_path, 'r') as file:
        for line in file:
            detection = line.strip().split(',')
            detection[0], detection[7], detection[8], detection[9] = int(detection[0]), int(detection[7]), int(detection[8]), int(detection[9])
            detection[1:7] = list(map(float, detection[1:7]))
            frame_number = detection[0]
            if frame_number not in detections.keys():
                detections[frame_number] = []
            detections[frame_number].append(detection)
    return detections

def xywh_to_xyxy(detections):
    """
    Convert a list of bounding box detections from (x1, y1, width, height) to (x1, y1, x2, y2) format.
    
    Args:
        detections (list): List of bounding box detections, each in the format (x1, y1, width, height).
        
    Returns:
        list: List of bounding box detections in (x1, y1, x2, y2) format.
    """
    converted_detections = []
    for bbox in detections:
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        converted_detections.append([x1, y1, x2, y2])
    return converted_detections