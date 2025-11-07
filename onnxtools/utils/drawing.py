"""
Drawing utilities for visualization of detection results.

TODO: Future refactoring
    The current implementation depends on the supervision library for drawing functionality.
    Consider implementing a naive/lightweight version that only uses OpenCV (cv2) to reduce
    external dependencies. This would involve:
    - Direct cv2.rectangle() for bounding boxes
    - Direct cv2.putText() with proper Chinese font support for labels
    - Manual color management without supervision's ColorPalette
    - Custom logic for label positioning and overlap avoidance

    Benefits:
    - Reduced dependency footprint
    - Faster installation (no supervision required)
    - More control over rendering behavior
    - Easier to debug and customize

    Challenges:
    - Need to reimplement smart label positioning
    - Chinese font rendering requires careful handling
    - Color palette management needs custom implementation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import supervision as sv

try:
    from .supervision_labels import create_ocr_labels
    from .font_utils import get_fallback_font_path
except ImportError as e:
    raise ImportError(
        "Required utilities are not available. "
        "Ensure supervision library is installed: pip install supervision>=0.16.0"
    ) from e


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


def create_box_annotator(thickness: int = 1,
                        color_palette: Optional[sv.ColorPalette] = None) -> sv.BoxAnnotator:
    """
    Create BoxAnnotator with optimized settings for vehicle detection.

    Args:
        thickness: Border thickness for detection boxes
        color_palette: Color palette for different classes

    Returns:
        Configured BoxAnnotator instance
    """
    return sv.BoxAnnotator(
        color=color_palette or sv.ColorPalette.DEFAULT,
        thickness=thickness,
        color_lookup=sv.ColorLookup.CLASS
    )


def create_rich_label_annotator(font_path: Optional[str] = None,
                               font_size: int = 16,
                               text_padding: int = 10,
                               color_palette: Optional[sv.ColorPalette] = None) -> sv.RichLabelAnnotator:
    """
    Create RichLabelAnnotator with Chinese font support and optimized settings.

    Args:
        font_path: Path to font file (will use fallback if None or invalid)
        font_size: Font size for text rendering
        text_padding: Padding around text
        color_palette: Color palette for label backgrounds

    Returns:
        Configured RichLabelAnnotator instance
    """
    # Get valid font path with fallback support
    valid_font_path = get_fallback_font_path(font_path)

    return sv.RichLabelAnnotator(
        color=color_palette or sv.ColorPalette.DEFAULT,
        text_color=sv.Color.BLACK,
        font_path=valid_font_path,
        font_size=font_size,
        text_padding=text_padding,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=sv.ColorLookup.CLASS,
        border_radius=3,
        smart_position=True  # Intelligent positioning to avoid overlaps
    )

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