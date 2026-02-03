"""Integration tests for geometric marker annotators (Circle, Triangle, Ellipse)."""

import numpy as np
import pytest
import supervision as sv

from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType


@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    xyxy = np.array([[100, 100, 250, 200], [300, 150, 450, 280]], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87]),
        class_id=np.array([0, 1])
    )


class TestGeometricAnnotators:
    """Integration tests for Circle, Triangle, and Ellipse annotators."""

    def test_circle_annotator(self, test_image, test_detections):
        """Test CircleAnnotator rendering."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.CIRCLE,
            {'thickness': 2}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_triangle_annotator(self, test_image, test_detections):
        """Test TriangleAnnotator rendering."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.TRIANGLE,
            {'base': 20, 'height': 20, 'position': sv.Position.TOP_CENTER}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_ellipse_annotator(self, test_image, test_detections):
        """Test EllipseAnnotator rendering."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.ELLIPSE,
            {'thickness': 2, 'start_angle': 0, 'end_angle': 360}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_triangle_different_positions(self, test_image, test_detections):
        """Test triangle at different positions."""
        positions = [
            sv.Position.TOP_CENTER,
            sv.Position.BOTTOM_CENTER,
            sv.Position.CENTER
        ]
        for position in positions:
            annotator = AnnotatorFactory.create(
                AnnotatorType.TRIANGLE,
                {'base': 20, 'height': 20, 'position': position}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_ellipse_partial_arc(self, test_image, test_detections):
        """Test ellipse with partial arc."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.ELLIPSE,
            {'thickness': 2, 'start_angle': 0, 'end_angle': 180}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_circle_different_thickness(self, test_image, test_detections):
        """Test circle with different thickness values."""
        for thickness in [1, 3, 5]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.CIRCLE,
                {'thickness': thickness}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape
