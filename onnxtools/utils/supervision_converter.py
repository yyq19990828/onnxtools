"""Format conversion utilities for supervision integration."""

import numpy as np
import supervision as sv
from typing import List, Dict, Any, Union


def convert_to_supervision_detections(detections_array: List[List[List[float]]],
                                    class_names: Union[Dict[int, str], List[str]]) -> sv.Detections:
    """
    Convert current detection format to supervision.Detections format.

    Args:
        detections_array: List of detection batches, each containing detection tuples
                         Format: [[[x1, y1, x2, y2, confidence, class_id], ...], ...]
        class_names: Dict mapping class_id to class_name or list of class names

    Returns:
        supervision.Detections object
    """
    if not detections_array or len(detections_array) == 0 or len(detections_array[0]) == 0:
        return sv.Detections.empty()

    # Take first batch (assuming single image detection)
    all_detections = detections_array[0]

    if len(all_detections) == 0:
        return sv.Detections.empty()

    # Extract coordinates, confidence, and class_id
    xyxy = np.array([[float(det[0]), float(det[1]), float(det[2]), float(det[3])]
                     for det in all_detections], dtype=np.float32)
    confidence = np.array([float(det[4]) for det in all_detections], dtype=np.float32)
    class_id = np.array([int(det[5]) for det in all_detections], dtype=np.int32)

    # Handle class names mapping
    if isinstance(class_names, dict):
        class_names_list = []
        for cls_id in class_id:
            if cls_id in class_names:
                class_names_list.append(class_names[cls_id])
            else:
                class_names_list.append(f"unknown_{cls_id}")
    else:
        # class_names is a list
        class_names_list = []
        for cls_id in class_id:
            if 0 <= cls_id < len(class_names):
                class_names_list.append(class_names[cls_id])
            else:
                class_names_list.append(f"unknown_{cls_id}")

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={'class_name': class_names_list}
    )