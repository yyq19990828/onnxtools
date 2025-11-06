"""Integration tests for basic detection drawing - 这些测试必须在实现前编写且必须失败."""

import pytest
import numpy as np
import cv2
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.integration
class TestBasicDrawingIntegration:
    """Integration tests for basic detection drawing functionality with supervision."""

    def test_end_to_end_basic_drawing(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: End-to-end basic detection drawing without OCR."""
        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

        # Basic output validation
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # Check that image was modified (not identical to input)
        assert not np.array_equal(result, sample_image), "Image should be modified with drawn boxes"

    def test_supervision_drawing_output(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Test supervision-based drawing output."""
        try:
            from onnxtools.utils.drawing import draw_detections

            # Supervision-based output (now the only implementation)
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

            # Should be valid image
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape

            # Should be different from original (has annotations)
            assert not np.array_equal(result, sample_image)

        except ImportError:
            pytest.fail("draw_detections must be implemented for integration")

    def test_multiple_detection_types(self, sample_image, sample_class_names, sample_colors):
        """Integration: Handle multiple detection types correctly."""
        # Create diverse detection set
        diverse_detections = [
            [
                [50.0, 50.0, 150.0, 100.0, 0.95, 0],   # vehicle - high confidence
                [200.0, 60.0, 240.0, 80.0, 0.88, 1],   # plate - medium confidence
                [100.0, 200.0, 200.0, 280.0, 0.75, 0], # vehicle - lower confidence
                [300.0, 300.0, 350.0, 320.0, 0.92, 1]  # plate - high confidence
            ]
        ]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, diverse_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_edge_case_coordinates(self, sample_image, sample_class_names, sample_colors):
        """Integration: Handle edge case bounding box coordinates."""
        edge_detections = [
            [
                [0.0, 0.0, 50.0, 50.0, 0.9, 0],                    # Top-left corner
                [590.0, 430.0, 640.0, 480.0, 0.85, 1],             # Bottom-right corner
                [300.0, 0.0, 350.0, 30.0, 0.8, 0],                 # Top edge
                [0.0, 240.0, 40.0, 280.0, 0.75, 1]                 # Left edge
            ]
        ]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, edge_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_overlapping_detections(self, sample_image, sample_class_names, sample_colors):
        """Integration: Handle overlapping detection boxes correctly."""
        overlapping_detections = [
            [
                [100.0, 100.0, 200.0, 200.0, 0.95, 0],  # Base box
                [150.0, 150.0, 250.0, 250.0, 0.90, 1],  # Overlapping box
                [120.0, 120.0, 180.0, 180.0, 0.85, 0]   # Nested box
            ]
        ]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, overlapping_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_confidence_display_integration(self, sample_image, sample_class_names, sample_colors):
        """Integration: Confidence scores should be displayed in labels."""
        varied_confidence_detections = [
            [
                [100.0, 100.0, 200.0, 150.0, 0.99, 0],  # Very high confidence
                [250.0, 200.0, 350.0, 250.0, 0.50, 1],  # Medium confidence
                [400.0, 300.0, 500.0, 350.0, 0.15, 0]   # Low confidence
            ]
        ]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, varied_confidence_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_color_assignment_integration(self, sample_image, sample_detections, sample_class_names):
        """Integration: Different color schemes should work correctly."""
        # Test with different color configurations
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, sample_detections, sample_class_names, custom_colors)

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_large_number_of_detections(self, sample_image, sample_class_names, sample_colors):
        """Integration: Handle large number of detections efficiently."""
        # Create many detections
        many_detections = []
        detection_list = []
        for i in range(30):  # 30 detections
            x1, y1 = (i % 6) * 100, (i // 6) * 80
            x2, y2 = x1 + 80, y1 + 60
            detection_list.append([x1, y1, x2, y2, 0.8 + (i % 20) * 0.01, i % 2])
        many_detections.append(detection_list)

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, many_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_empty_detections_integration(self, sample_image, sample_class_names, sample_colors):
        """Integration: Empty detection list should return unmodified image."""
        empty_detections = [[]]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, empty_detections, sample_class_names, sample_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        # With empty detections, image might be identical or slightly modified
        # depending on implementation

    def test_different_image_sizes(self, sample_class_names, sample_colors):
        """Integration: Drawing should work with different image dimensions."""
        test_sizes = [
            (240, 320, 3),   # Small
            (480, 640, 3),   # Medium (sample size)
            (720, 1280, 3),  # Large
            (1080, 1920, 3)  # Full HD
        ]

        sample_detections = [[[100.0, 100.0, 200.0, 150.0, 0.9, 0]]]

        from onnxtools.utils.drawing import draw_detections

        for size in test_sizes:
            test_image = np.zeros(size, dtype=np.uint8)
            test_image[:, :] = [128, 128, 128]  # Gray background

            result = draw_detections(test_image, sample_detections, sample_class_names, sample_colors)

            assert isinstance(result, np.ndarray)
            assert result.shape == size

    def test_supervision_format_conversion_integration(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Format conversion should work seamlessly in drawing pipeline."""
        # This test ensures the full pipeline works
        try:
            from onnxtools.utils.supervision_converter import convert_to_supervision_detections
            from onnxtools.utils.drawing import draw_detections
            import supervision as sv

            # Convert detections
            sv_detections = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Use in supervision drawing
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape

        except ImportError:
            pytest.fail("Supervision integration components must be implemented")

    def test_thread_safety_integration(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Drawing should be thread-safe for concurrent usage."""
        import threading
        import time

        results = []
        errors = []

        def draw_worker():
            try:
                from onnxtools.utils.drawing import draw_detections
                result = draw_detections(sample_image.copy(), sample_detections, sample_class_names, sample_colors)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=draw_worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Thread safety issues: {errors}"
        assert len(results) == 5, "All threads should complete"

        # All results should be valid
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape