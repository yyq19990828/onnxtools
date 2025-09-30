import cv2
import numpy as np
import os
import logging
import time
from typing import List, Dict, Any, Optional, Union

try:
    import supervision as sv
    from .supervision_converter import convert_to_supervision_detections
    from .supervision_labels import create_ocr_labels
    from .supervision_config import create_box_annotator, create_rich_label_annotator
except ImportError as e:
    raise ImportError(
        "supervision library is required for drawing functionality. "
        "Install it with: pip install supervision>=0.16.0"
    ) from e

def draw_detections(image, detections, class_names, colors, plate_results=None, font_path="SourceHanSans-VF.ttf"):
    """
    Draws detection boxes on the image using supervision library.

    Args:
        image: Input image as numpy array (BGR)
        detections: Detection results
        class_names: Class name mapping
        colors: Colors for different classes
        plate_results: Optional OCR results for plates
        font_path: Path to font file

    Returns:
        Annotated image as numpy array (BGR)
    """
    return draw_detections_supervision(image, detections, class_names, colors, plate_results, font_path)


def draw_detections_supervision(image: np.ndarray,
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
        labels = create_ocr_labels(detections[0], plate_results or [], class_names)

        # Draw labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=sv_detections,
            labels=labels
        )

    return annotated_image


def benchmark_drawing_performance(image: np.ndarray,
                                 detections_data: List[List[List[float]]],
                                 iterations: int = 100,
                                 target_ms: float = 10.0) -> Dict[str, float]:
    """
    Benchmark drawing performance with supervision implementation.

    Args:
        image: Test image
        detections_data: Detection data for testing
        iterations: Number of test iterations
        target_ms: Performance target in milliseconds

    Returns:
        Dictionary with performance metrics: {'avg_time_ms': float, 'target_met': bool}
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    # Mock class names and colors for testing
    class_names = {0: "vehicle", 1: "plate"}
    colors = [(255, 0, 0), (0, 255, 0)]

    # Test supervision implementation (only)
    start_time = time.time()
    for _ in range(iterations):
        _ = draw_detections(image.copy(), detections_data, class_names, colors)
    avg_time = (time.time() - start_time) / iterations * 1000  # ms

    return {
        'avg_time_ms': avg_time,
        'target_met': avg_time < target_ms
    }