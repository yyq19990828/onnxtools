"""Integration test for DotAnnotator."""

import pytest
import numpy as np
import supervision as sv
from utils.annotator_factory import AnnotatorFactory, AnnotatorType


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


class TestDotAnnotatorIntegration:
    """Integration tests for DotAnnotator."""

    def test_dot_basic_rendering(self, test_image, test_detections):
        """Test basic dot rendering at center."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.DOT,
            {'radius': 5, 'position': sv.Position.CENTER}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_dot_different_radii(self, test_image, test_detections):
        """Test different dot radius values."""
        for radius in [3, 5, 8, 10]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.DOT,
                {'radius': radius, 'position': sv.Position.CENTER}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_dot_different_positions(self, test_image, test_detections):
        """Test dot at different anchor positions."""
        positions = [
            sv.Position.CENTER,
            sv.Position.TOP_LEFT,
            sv.Position.BOTTOM_RIGHT
        ]
        for position in positions:
            annotator = AnnotatorFactory.create(
                AnnotatorType.DOT,
                {'radius': 5, 'position': position}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_dot_with_outline(self, test_image, test_detections):
        """Test dot with outline."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.DOT,
            {
                'radius': 5,
                'position': sv.Position.CENTER,
                'outline_thickness': 2,
                'outline_color': sv.Color.BLACK
            }
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape