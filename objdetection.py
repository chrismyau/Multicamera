import torch
import ultralytics
ultralytics.checks()

def process_frame(frame, model):
    """
    Processes a single frame using the YOLO model.

    Args:
        frame: The video frame to process.
        model: The YOLO model to use for detection.

    Returns:
        results: The detection results from YOLO.
    """
    
    results = model(frame)
    return results
