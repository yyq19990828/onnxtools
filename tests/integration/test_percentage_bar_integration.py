"""Integration test for PercentageBarAnnotator."""

import numpy as np
import supervision as sv

from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType


class TestPercentageBarAnnotatorIntegration:
    """Integration tests for PercentageBarAnnotator."""

    def test_percentage_bar_basic(self, test_image, test_detections):
        """Test basic percentage bar rendering."""
        annotator = AnnotatorFactory.create(AnnotatorType.PERCENTAGE_BAR, {"height": 16, "width": 80})
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_percentage_bar_different_sizes(self, test_image, test_detections):
        """Test different bar sizes."""
        configs = [{"height": 10, "width": 60}, {"height": 20, "width": 100}, {"height": 30, "width": 120}]
        for config in configs:
            annotator = AnnotatorFactory.create(AnnotatorType.PERCENTAGE_BAR, config)
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_percentage_bar_positions(self, test_image, test_detections):
        """Test different position placements."""
        positions = [sv.Position.TOP_LEFT, sv.Position.TOP_CENTER, sv.Position.BOTTOM_LEFT]
        for position in positions:
            annotator = AnnotatorFactory.create(
                AnnotatorType.PERCENTAGE_BAR, {"height": 16, "width": 80, "position": position}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_percentage_bar_high_confidence(self, test_image):
        """Test with high confidence values."""
        high_conf_detections = sv.Detections(
            xyxy=np.array([[100, 100, 250, 200]], dtype=np.float32), confidence=np.array([0.99]), class_id=np.array([0])
        )
        annotator = AnnotatorFactory.create(AnnotatorType.PERCENTAGE_BAR, {"height": 16, "width": 80})
        result = annotator.annotate(test_image, high_conf_detections)
        assert result.shape == test_image.shape

    def test_percentage_bar_low_confidence(self, test_image):
        """Test with low confidence values."""
        low_conf_detections = sv.Detections(
            xyxy=np.array([[100, 100, 250, 200]], dtype=np.float32), confidence=np.array([0.15]), class_id=np.array([0])
        )
        annotator = AnnotatorFactory.create(AnnotatorType.PERCENTAGE_BAR, {"height": 16, "width": 80})
        result = annotator.annotate(test_image, low_conf_detections)
        assert result.shape == test_image.shape
