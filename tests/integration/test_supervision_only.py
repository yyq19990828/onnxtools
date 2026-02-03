"""Integration tests for supervision-only drawing implementation.

These tests verify that the drawing module works correctly with only
the supervision library, without PIL fallback.
"""

import os
import sys

import cv2
import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_test_image(width, height, color=(255, 255, 255)):
    """Create a test image with specified dimensions and color."""
    return np.full((height, width, 3), color, dtype=np.uint8)


@pytest.mark.integration
class TestSupervisionOnlyRendering:
    """Integration tests for supervision-only drawing implementation."""

    def test_supervision_only_basic_rendering(self):
        """Test that supervision renders basic detections correctly."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(640, 480, (255, 255, 255))
        detections = [[[100, 100, 200, 200, 0.9, 0]]]
        class_names = {0: "vehicle"}
        colors = [(255, 0, 0)]

        result = draw_detections(image, detections, class_names, colors)

        assert result.shape == image.shape, "Output shape must match input"
        assert result.dtype == np.uint8, "Output dtype must be uint8"
        assert not np.array_equal(result, image), "Image must be modified with bounding box"

    def test_supervision_only_multiple_detections(self):
        """Test supervision rendering with multiple detections."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(640, 480)
        detections = [[
            [50, 50, 150, 150, 0.95, 0],  # vehicle
            [200, 200, 350, 280, 0.88, 1],  # plate
            [400, 100, 550, 250, 0.92, 0],  # another vehicle
        ]]
        class_names = {0: "vehicle", 1: "plate"}
        colors = [(255, 0, 0), (0, 255, 0)]

        result = draw_detections(image, detections, class_names, colors)

        assert result.shape == image.shape
        assert not np.array_equal(result, image), "Multiple boxes should be drawn"

    def test_supervision_only_chinese_rendering(self):
        """Test Chinese character rendering for plate OCR with supervision."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(640, 480)
        detections = [[[200, 150, 450, 220, 0.93, 1]]]
        class_names = {1: "plate"}
        colors = [(0, 255, 0)]

        plate_results = [{
            "plate_text": "苏A88888",  # Chinese + alphanumeric
            "plate_conf": 0.95,
            "color": "blue",
            "layer": "single",
            "should_display_ocr": True
        }]

        result = draw_detections(
            image, detections, class_names, colors,
            plate_results=plate_results,
            font_path="SourceHanSans-VF.ttf"
        )

        assert result.shape == image.shape
        assert not np.array_equal(result, image), "OCR text should be rendered"

    def test_supervision_only_empty_detections(self):
        """Test handling of empty detection list."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(640, 480)
        detections = [[]]  # Empty detection list
        class_names = {0: "vehicle"}
        colors = [(255, 0, 0)]

        result = draw_detections(image, detections, class_names, colors)

        assert result.shape == image.shape
        # Empty detections should return image (possibly slightly modified by supervision)
        assert isinstance(result, np.ndarray)

    def test_supervision_only_large_detection_count(self):
        """Test supervision performance with many detections (50+ boxes)."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(1920, 1080)

        # Generate 50 random detections
        detections_list = []
        for i in range(50):
            x1 = np.random.randint(0, 1800)
            y1 = np.random.randint(0, 1000)
            x2 = min(x1 + 100, 1920)
            y2 = min(y1 + 80, 1080)
            conf = np.random.uniform(0.7, 0.99)
            cls = i % 2
            detections_list.append([x1, y1, x2, y2, conf, cls])

        detections = [detections_list]
        class_names = {0: "vehicle", 1: "plate"}
        colors = [(255, 0, 0), (0, 255, 0)]

        result = draw_detections(image, detections, class_names, colors)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should handle large number of detections without errors

    def test_supervision_only_font_missing_fallback(self):
        """Test behavior when font file is missing (supervision should use fallback)."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(640, 480)
        detections = [[[100, 100, 200, 200, 0.9, 0]]]
        class_names = {0: "vehicle"}
        colors = [(255, 0, 0)]

        # Use non-existent font path
        result = draw_detections(
            image, detections, class_names, colors,
            font_path="non_existent_font.ttf"
        )

        # Should not crash, supervision will use fallback font
        assert result.shape == image.shape
        assert isinstance(result, np.ndarray)

    def test_supervision_only_with_plate_ocr_full(self):
        """Test full integration with vehicle detection and plate OCR."""
        from onnxtools.utils.drawing import draw_detections

        image = create_test_image(800, 600)
        detections = [[
            [100, 100, 300, 250, 0.92, 0],  # vehicle
            [150, 180, 250, 220, 0.88, 1],  # plate inside vehicle
        ]]
        class_names = {0: "vehicle", 1: "plate"}
        colors = [(255, 0, 0), (0, 255, 0)]

        plate_results = [
            None,  # vehicle has no OCR
            {
                "plate_text": "京B12345",
                "plate_conf": 0.93,
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            }
        ]

        result = draw_detections(
            image, detections, class_names, colors,
            plate_results=plate_results
        )

        assert result.shape == image.shape
        assert not np.array_equal(result, image)

    def test_supervision_library_available(self):
        """Test that supervision library is available (required dependency)."""
        try:
            import supervision as sv
            assert hasattr(sv, '__version__'), "Supervision must have version attribute"
            # Should be >= 0.16.0
            version_parts = sv.__version__.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            assert major > 0 or (major == 0 and minor >= 16), \
                f"Supervision version {sv.__version__} too old, need >= 0.16.0"
        except ImportError:
            pytest.fail("supervision library must be installed")

    def test_supervision_annotators_integration(self):
        """Test that supervision annotators are properly integrated."""
        try:
            from onnxtools.utils.drawing import (
                convert_to_supervision_detections,
                create_box_annotator,
                create_rich_label_annotator,
            )

            # Test annotator creation
            box_annotator = create_box_annotator(thickness=3)
            label_annotator = create_rich_label_annotator(font_size=16)

            assert box_annotator is not None, "Box annotator creation failed"
            assert label_annotator is not None, "Label annotator creation failed"

        except ImportError as e:
            pytest.fail(f"Supervision integration modules not available: {e}")
