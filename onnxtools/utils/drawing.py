import numpy as np
from typing import List, Dict, Any, Optional, Union

try:
    from .supervision_converter import convert_to_supervision_detections
    from .supervision_labels import create_ocr_labels
    from .supervision_config import create_box_annotator, create_rich_label_annotator
except ImportError as e:
    raise ImportError(
        "supervision library is required for drawing functionality. "
        "Install it with: pip install supervision>=0.16.0"
    ) from e

def draw_detections(image: np.ndarray,
                   detections: List[List[List[float]]],
                   class_names: Union[Dict[int, str], List[str]],
                   colors: List[tuple],
                   plate_results: Optional[List[Optional[Dict[str, Any]]]] = None,
                   font_path: str = "SourceHanSans-VF.ttf") -> np.ndarray:
    """
    Draw detection boxes using supervision library for enhanced performance and visuals.

    Args:
        image: Input image as BGR numpy array
        detections: Detection results in format [[[x1, y1, x2, y2, confidence, class_id], ...]]
        class_names: Dict mapping class_id to class_name or list of class names
        colors: List of colors for different classes (not used directly in supervision)
        plate_results: Optional OCR results for plate detections
        font_path: Path to font file for text rendering

    Returns:
        Annotated image as BGR numpy array
    """
    # Supervision is now a required dependency (ImportError raised at module level if not available)

    # Convert to supervision format
    sv_detections = convert_to_supervision_detections(detections, class_names)

    if len(sv_detections.xyxy) == 0:
        return image.copy()

    # Create annotators
    box_annotator = create_box_annotator(thickness=3)
    label_annotator = create_rich_label_annotator(font_path=font_path, font_size=16)

    # Start with copy of input image
    annotated_image = image.copy()

    # Draw bounding boxes
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=sv_detections
    )

    # Create labels with OCR information
    if len(detections) > 0 and len(detections[0]) > 0:
        # Extract separate arrays from detection format for create_ocr_labels
        det_array = np.array(detections[0])  # [N, 6] format
        boxes = det_array[:, :4]              # [N, 4] xyxy
        scores = det_array[:, 4]              # [N] confidence
        class_ids = det_array[:, 5].astype(int)  # [N] class_id

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results or [], class_names)

        # Draw labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=sv_detections,
            labels=labels
        )

    return annotated_image