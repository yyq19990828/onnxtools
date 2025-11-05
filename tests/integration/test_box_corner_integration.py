"""
Integration test for BoxCornerAnnotator end-to-end functionality.
"""

import pytest
import numpy as np
import supervision as sv
from onnxtools.utils.annotator_factory import AnnotatorFactory, AnnotatorType


@pytest.fixture
def test_image():
    """Create 640x640 test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create test detections with 5 objects."""
    xyxy = np.array([
        [100, 100, 250, 200],
        [300, 150, 450, 280],
        [500, 100, 620, 220],
        [100, 350, 240, 480],
        [350, 400, 500, 550]
    ], dtype=np.float32)

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]),
        class_id=np.array([0, 1, 0, 1, 0])
    )


class TestBoxCornerAnnotatorIntegration:
    """Integration tests for BoxCornerAnnotator."""

    def test_box_corner_basic_rendering(self, test_image, test_detections):
        """Test basic box corner rendering."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {'thickness': 2, 'corner_length': 20}
        )

        result = annotator.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        # Just verify it rendered successfully

    def test_box_corner_different_corner_lengths(self, test_image, test_detections):
        """Test different corner length values."""
        corner_lengths = [10, 20, 30, 40]

        for corner_length in corner_lengths:
            annotator = AnnotatorFactory.create(
                AnnotatorType.BOX_CORNER,
                {'thickness': 2, 'corner_length': corner_length}
            )
            result = annotator.annotate(test_image.copy(), test_detections)

            assert result.shape == test_image.shape

    def test_box_corner_different_thickness(self, test_image, test_detections):
        """Test different thickness values."""
        thickness_values = [1, 2, 4, 6]

        for thickness in thickness_values:
            annotator = AnnotatorFactory.create(
                AnnotatorType.BOX_CORNER,
                {'thickness': thickness, 'corner_length': 20}
            )
            result = annotator.annotate(test_image.copy(), test_detections)

            assert result.shape == test_image.shape

    def test_box_corner_dense_detections(self, test_image):
        """Test with many close detections (corner overlap scenario)."""
        # Create grid of detections
        detections_list = []
        for x in range(50, 600, 100):
            for y in range(50, 600, 100):
                detections_list.append([x, y, x + 80, y + 80])

        dense_detections = sv.Detections(
            xyxy=np.array(detections_list, dtype=np.float32),
            confidence=np.ones(len(detections_list)) * 0.9,
            class_id=np.zeros(len(detections_list), dtype=int)
        )

        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {'thickness': 2, 'corner_length': 15}
        )

        result = annotator.annotate(test_image, dense_detections)
        assert result.shape == test_image.shape

    def test_box_corner_small_boxes(self, test_image):
        """Test with small boxes where corner length is significant."""
        small_detections = sv.Detections(
            xyxy=np.array([
                [100, 100, 130, 130],  # 30x30 box
                [200, 200, 220, 220],  # 20x20 box
                [300, 300, 340, 340]   # 40x40 box
            ], dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.85]),
            class_id=np.array([0, 1, 0])
        )

        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {'thickness': 2, 'corner_length': 15}
        )

        result = annotator.annotate(test_image, small_detections)
        assert result.shape == test_image.shape

    def test_box_corner_large_corner_length(self, test_image, test_detections):
        """Test with very large corner length."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {'thickness': 3, 'corner_length': 50}
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_box_corner_empty_detections(self, test_image):
        """Test with empty detections."""
        empty_detections = sv.Detections.empty()

        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {'thickness': 2, 'corner_length': 20}
        )

        result = annotator.annotate(test_image, empty_detections)
        assert result.shape == test_image.shape

    def test_box_corner_with_color_palette(self, test_image, test_detections):
        """Test with different color palettes."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {
                'thickness': 2,
                'corner_length': 20,
                'color_palette': sv.ColorPalette.ROBOFLOW
            }
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_box_corner_color_lookup_class(self, test_image, test_detections):
        """Test color lookup by class."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {
                'thickness': 2,
                'corner_length': 20,
                'color_lookup': sv.ColorLookup.CLASS
            }
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_box_corner_minimal_config(self, test_image, test_detections):
        """Test with minimal configuration (defaults)."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BOX_CORNER,
            {}
        )

        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape