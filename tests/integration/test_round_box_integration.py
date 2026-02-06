"""
Integration test for RoundBoxAnnotator end-to-end functionality.
"""

import numpy as np
import supervision as sv

from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType


class TestRoundBoxAnnotatorIntegration:
    """Integration tests for RoundBoxAnnotator."""

    def test_round_box_basic_rendering(self, test_image, test_detections):
        """Test basic round box rendering."""
        annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3})

        result = annotator.annotate(test_image, test_detections)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype
        # Image should be modified (supervision may return same or modified)
        # Just verify it rendered successfully

    def test_round_box_different_roundness_values(self, test_image, test_detections):
        """Test different roundness parameter values."""
        roundness_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Note: roundness must be > 0

        results = []
        for roundness in roundness_values:
            annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": roundness})
            result = annotator.annotate(test_image.copy(), test_detections)
            results.append(result)

        # All should render successfully
        for result in results:
            assert result.shape == test_image.shape

        # Different roundness should produce different results (except maybe 0.0 and 1.0)
        # At least some should be different
        differences = []
        for i in range(len(results) - 1):
            differences.append(not np.array_equal(results[i], results[i + 1]))
        assert any(differences), "Different roundness values should produce some different results"

    def test_round_box_different_thickness(self, test_image, test_detections):
        """Test different thickness values."""
        thickness_values = [1, 2, 4, 6]

        for thickness in thickness_values:
            annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": thickness, "roundness": 0.3})
            result = annotator.annotate(test_image.copy(), test_detections)

            assert result.shape == test_image.shape

    def test_round_box_with_color_palette(self, test_image, test_detections):
        """Test with custom color palette."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.ROUND_BOX, {"thickness": 3, "roundness": 0.4, "color_palette": sv.ColorPalette.ROBOFLOW}
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_round_box_single_detection(self, test_image):
        """Test with single detection."""
        single_detection = sv.Detections(
            xyxy=np.array([[200, 200, 400, 400]], dtype=np.float32), confidence=np.array([0.9]), class_id=np.array([0])
        )

        annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3})

        result = annotator.annotate(test_image, single_detection)
        assert result.shape == test_image.shape

    def test_round_box_empty_detections(self, test_image):
        """Test with empty detections."""
        empty_detections = sv.Detections.empty()

        annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3})

        result = annotator.annotate(test_image, empty_detections)

        # Should return modified image (or copy)
        assert result.shape == test_image.shape

    def test_round_box_overlapping_boxes(self, test_image):
        """Test with overlapping detection boxes."""
        overlapping_detections = sv.Detections(
            xyxy=np.array([[100, 100, 300, 300], [200, 200, 400, 400], [150, 150, 350, 350]], dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.85]),
            class_id=np.array([0, 1, 0]),
        )

        annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3})

        result = annotator.annotate(test_image, overlapping_detections)
        assert result.shape == test_image.shape

    def test_round_box_extreme_roundness(self, test_image, test_detections):
        """Test with extreme roundness values (edge cases)."""
        # Minimum roundness (must be > 0)
        annotator_min = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.01})
        result_min = annotator_min.annotate(test_image.copy(), test_detections)
        assert result_min.shape == test_image.shape

        # Maximum roundness
        annotator_max = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 1.0})
        result_max = annotator_max.annotate(test_image.copy(), test_detections)
        assert result_max.shape == test_image.shape

    def test_round_box_small_boxes(self, test_image):
        """Test with very small detection boxes."""
        small_detections = sv.Detections(
            xyxy=np.array([[100, 100, 110, 110], [200, 200, 215, 215], [300, 300, 320, 320]], dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.85]),
            class_id=np.array([0, 1, 0]),
        )

        annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.5})

        result = annotator.annotate(test_image, small_detections)
        assert result.shape == test_image.shape

    def test_round_box_color_lookup_class(self, test_image, test_detections):
        """Test color lookup by class."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3, "color_lookup": sv.ColorLookup.CLASS}
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
