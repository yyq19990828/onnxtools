"""Label generation functions for supervision annotators."""

from typing import List, Dict, Any, Optional, Union


def create_ocr_labels(detections: List[List[float]],
                     plate_results: List[Optional[Dict[str, Any]]],
                     class_names: Union[Dict[int, str], List[str]]) -> List[str]:
    """
    Create labels for detections including OCR information for plate class.

    Args:
        detections: List of detection tuples [x1, y1, x2, y2, confidence, class_id]
        plate_results: List of OCR results for each detection (None for non-plates)
        class_names: Dict mapping class_id to class_name or list of class names

    Returns:
        List of label strings for each detection
    """
    labels = []

    for i, detection in enumerate(detections):
        if len(detection) < 6:
            labels.append("invalid_detection")
            continue

        class_id = int(detection[5])
        confidence = detection[4]

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
        if (i < len(plate_results) and plate_results[i] is not None and
            class_name == 'plate' and isinstance(plate_results[i], dict)):

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