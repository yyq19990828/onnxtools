"""Contract tests for draw_detections API - 这些测试必须在实现前编写且必须失败."""

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.contract
class TestDrawDetectionsContract:
    """Contract tests to ensure draw_detections API remains stable during supervision integration."""

    def test_draw_detections_function_exists(self):
        """Contract: draw_detections function must exist in utils.drawing module."""
        from onnxtools.utils.drawing import draw_detections
        assert callable(draw_detections), "draw_detections must be a callable function"

    def test_draw_detections_signature_compatibility(self):
        """Contract: draw_detections must have supervision-only signature (use_supervision removed)."""
        import inspect

        from onnxtools.utils.drawing import draw_detections

        sig = inspect.signature(draw_detections)
        params = list(sig.parameters.keys())

        # New API signature after removing use_supervision parameter
        expected_params = ['image', 'detections', 'class_names', 'colors', 'plate_results', 'font_path']

        assert params == expected_params, f"Function signature changed. Expected {expected_params}, got {params}"
        assert 'use_supervision' not in params, "use_supervision parameter should be removed"
        assert len(params) == 6, f"Expected 6 params, got {len(params)}"

        # Check default values are preserved
        assert sig.parameters['plate_results'].default is None
        assert sig.parameters['font_path'].default == "SourceHanSans-VF.ttf"

    def test_draw_detections_input_validation(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Contract: draw_detections must validate input parameters and handle edge cases."""
        from onnxtools.utils.drawing import draw_detections

        # Test with valid inputs (should not raise)
        try:
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)
            assert isinstance(result, np.ndarray), "Return type must be np.ndarray"
        except Exception as e:
            pytest.fail(f"Valid inputs should not raise exception: {e}")

        # Test with empty detections
        empty_detections = [[]]
        result = draw_detections(sample_image, empty_detections, sample_class_names, sample_colors)
        assert isinstance(result, np.ndarray), "Empty detections should return valid image"

        # Test with None plate_results
        result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors, plate_results=None)
        assert isinstance(result, np.ndarray), "None plate_results should be handled"

    def test_draw_detections_output_format_contract(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Contract: draw_detections must return BGR numpy array compatible with cv2."""
        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

        # Output format requirements
        assert isinstance(result, np.ndarray), "Output must be numpy array"
        assert result.dtype == np.uint8, "Output must be uint8 dtype"
        assert len(result.shape) == 3, "Output must be 3D array (H, W, C)"
        assert result.shape[2] == 3, "Output must have 3 channels (BGR)"
        assert result.shape[:2] == sample_image.shape[:2], "Output dimensions must match input"

    def test_draw_detections_ocr_integration_contract(self, sample_image, sample_detections,
                                                    sample_class_names, sample_colors, sample_plate_results):
        """Contract: draw_detections must properly integrate OCR results for plate class."""
        from onnxtools.utils.drawing import draw_detections

        # Test with OCR results
        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=sample_plate_results
        )

        assert isinstance(result, np.ndarray), "OCR integration should return valid image"
        assert result.shape == sample_image.shape, "OCR integration should preserve image dimensions"

    def test_draw_detections_supervision_integration_contract(self, sample_image, sample_detections,
                                                            sample_class_names, sample_colors):
        """Contract: Supervision-based implementation must be available."""
        try:
            from onnxtools.utils.drawing import draw_detections
            assert callable(draw_detections), "draw_detections must be callable"

            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)
            assert isinstance(result, np.ndarray), "Supervision implementation must return numpy array"

        except ImportError:
            pytest.fail("draw_detections function must be implemented")

    def test_convert_to_supervision_detections_contract(self, sample_detections, sample_class_names):
        """Contract: Format conversion function must be available and work correctly."""
        # This test will fail initially and pass after implementation
        try:
            import supervision as sv

            from onnxtools.utils.drawing import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            assert isinstance(result, sv.Detections), "Must return supervision.Detections object"
            assert hasattr(result, 'xyxy'), "Must have xyxy attribute"
            assert hasattr(result, 'confidence'), "Must have confidence attribute"
            assert hasattr(result, 'class_id'), "Must have class_id attribute"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_supervision_always_used_contract(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Contract: Supervision library must always be used (no fallback)."""
        # After refactoring, only supervision implementation exists
        try:
            import inspect

            from onnxtools.utils.drawing import draw_detections

            # Verify use_supervision parameter is removed
            sig = inspect.signature(draw_detections)
            assert 'use_supervision' not in sig.parameters, "use_supervision parameter should not exist"

            # Test normal operation uses supervision
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)
            assert isinstance(result, np.ndarray), "Supervision-only implementation must work"

        except Exception as e:
            pytest.fail(f"Supervision-only implementation failed: {e}")

    def test_chinese_font_support_contract(self, sample_image, sample_detections,
                                         sample_class_names, sample_colors, sample_plate_results):
        """Contract: Chinese font support must be maintained."""
        from onnxtools.utils.drawing import draw_detections

        # Test with Chinese OCR results
        chinese_plate_results = [
            None,  # vehicle
            {
                "plate_text": "京A12345",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            }
        ]

        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=chinese_plate_results
        )

        assert isinstance(result, np.ndarray), "Chinese font support must be maintained"
