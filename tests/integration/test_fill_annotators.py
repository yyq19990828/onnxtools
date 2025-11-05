"""Integration tests for fill annotators (Color, BackgroundOverlay)."""

import pytest
import numpy as np
import supervision as sv
from onnxtools.utils.annotator_factory import AnnotatorFactory, AnnotatorType


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


class TestFillAnnotators:
    """Integration tests for ColorAnnotator and BackgroundOverlayAnnotator."""

    def test_color_annotator_basic(self, test_image, test_detections):
        """Test ColorAnnotator with default opacity."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.COLOR,
            {'opacity': 0.3}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_color_annotator_different_opacity(self, test_image, test_detections):
        """Test different opacity values."""
        for opacity in [0.1, 0.3, 0.5, 0.7]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.COLOR,
                {'opacity': opacity}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_background_overlay_basic(self, test_image, test_detections):
        """Test BackgroundOverlayAnnotator."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BACKGROUND_OVERLAY,
            {'opacity': 0.5, 'color': sv.Color.BLACK}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_background_overlay_different_colors(self, test_image, test_detections):
        """Test background overlay with different colors."""
        colors = [sv.Color.BLACK, sv.Color.WHITE]
        for color in colors:
            annotator = AnnotatorFactory.create(
                AnnotatorType.BACKGROUND_OVERLAY,
                {'opacity': 0.5, 'color': color}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_color_annotator_with_palette(self, test_image, test_detections):
        """Test ColorAnnotator with custom palette."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.COLOR,
            {
                'opacity': 0.3,
                'color_palette': sv.ColorPalette.ROBOFLOW
            }
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape