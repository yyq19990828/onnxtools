"""Label generation functions for supervision annotators."""

from typing import Any, Dict, List, Optional, Union

import numpy as np


def create_confidence_labels(scores: np.ndarray) -> List[str]:
    """
    Create labels with confidence scores only.

    Args:
        scores: Confidence scores array [N]

    Returns:
        List of confidence strings like ["0.85", "0.92", ...]
    """
    return [f"{float(score):.2f}" for score in scores]


def create_ocr_labels(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    plate_results: List[Optional[Dict[str, Any]]],
    class_names: Union[Dict[int, str], List[str]]
) -> List[str]:
    """
    Create labels for detections including OCR information for plate class.

    Adapted for Result API - accepts separate arrays instead of combined detection list.

    Args:
        boxes: Detection boxes array [N, 4] in xyxy format
        scores: Confidence scores array [N]
        class_ids: Class ID array [N]
        plate_results: List of OCR results for each detection (None for non-plates)
        class_names: Dict mapping class_id to class_name or list of class names

    Returns:
        List of label strings for each detection

    Example:
        >>> result = detector(image)
        >>> labels = create_ocr_labels(
        ...     result.boxes, result.scores, result.class_ids,
        ...     plate_results, class_names
        ... )
    """
    labels = []
    n_detections = len(boxes)

    for i in range(n_detections):
        class_id = int(class_ids[i])
        confidence = float(scores[i])

        # Get class name
        if isinstance(class_names, dict):
            class_name = class_names.get(class_id, f"unknown_{class_id}")
        else:
            # class_names is a list
            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"unknown_{class_id}"

        # Base label
        base_label = f"{class_name} {confidence:.2f}"

        # Add OCR information if available
        if (i < len(plate_results) and plate_results[i] is not None
                and class_name == 'plate' and isinstance(plate_results[i], dict)):

            plate_result = plate_results[i]
            if plate_result.get("should_display_ocr", False):
                ocr_text = plate_result.get("plate_text", "")
                color = plate_result.get("color", "")
                layer = plate_result.get("layer", "")

                if ocr_text:
                    ocr_info = []
                    if color and color != "unknown":
                        ocr_info.append(color)
                    if layer and layer != "unknown":
                        ocr_info.append(layer)

                    # Create multi-line label (supervision RichLabelAnnotator supports \n)
                    ocr_line = f"{ocr_text}"
                    if ocr_info:
                        ocr_line += f"\n{' '.join(ocr_info)}"

                    labels.append(f"{base_label}\n{ocr_line}")
                else:
                    labels.append(base_label)
            else:
                labels.append(base_label)
        else:
            labels.append(base_label)

    return labels
