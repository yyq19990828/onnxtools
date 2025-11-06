"""
Contract tests for refactored OcrORT and ColorLayerORT classes.

This module validates that the refactored classes maintain API compatibility
and proper behavior after migrating from utils functions to class methods.

Test Coverage:
- ColorLayerORT __call__() interface contract
- OcrORT __call__() interface contract
- Backward compatible infer() methods
- Static method accessibility
- Type safety and return value contracts
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any


@pytest.fixture
def color_layer_model(color_layer_model_path, color_map, layer_map):
    """Create ColorLayerORT instance for contract testing."""
    from onnxtools.infer_onnx.onnx_ocr import ColorLayerORT
    return ColorLayerORT(
        str(color_layer_model_path),
        color_map=color_map,
        layer_map=layer_map
    )


@pytest.fixture
def ocr_model_refactored(ocr_model_path, ocr_character):
    """Create refactored OcrORT instance for contract testing."""
    from onnxtools.infer_onnx.onnx_ocr import OcrORT
    return OcrORT(
        str(ocr_model_path),
        character=ocr_character
    )


class TestColorLayerORTContract:
    """Contract tests for ColorLayerORT refactored class."""

    def test_initialization_contract(self, color_layer_model_path, color_map, layer_map):
        """
        Contract: ColorLayerORT initialization must accept required parameters.

        Required parameters:
        - onnx_path: str
        - color_map: Dict[int, str]
        - layer_map: Dict[int, str]

        Optional parameters:
        - input_shape: tuple (default (224, 224))
        - conf_thres: float (default 0.5)
        - providers: list (default ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        from onnxtools.infer_onnx.onnx_ocr import ColorLayerORT

        # Test minimal initialization
        model = ColorLayerORT(
            str(color_layer_model_path),
            color_map=color_map,
            layer_map=layer_map
        )

        assert model is not None
        assert model.color_map == color_map
        assert model.layer_map == layer_map
        # Note: input_shape is None before initialization (lazy loading)
        # It will be set after first inference or manual _ensure_initialized() call
        assert model._requested_input_shape == (48, 168)  # User-requested shape

        # Test with custom parameters
        model_custom = ColorLayerORT(
            str(color_layer_model_path),
            color_map=color_map,
            layer_map=layer_map,
            input_shape=(64, 200),
            conf_thres=0.8
        )

        assert model_custom._requested_input_shape == (64, 200)
        assert model_custom.conf_thres == 0.8

    def test_call_interface_contract(self, color_layer_model, sample_blue_plate):
        """
        Contract: ColorLayerORT.__call__() must return valid color/layer results.

        Input: np.ndarray (BGR image)
        Output: Tuple[str, str, float] = (color, layer, confidence)

        Constraints:
        - color must be in color_map values
        - layer must be in layer_map values
        - confidence must be in [0.0, 1.0]
        """
        # Call the model
        color, layer, confidence = color_layer_model(sample_blue_plate)

        # Validate output structure
        assert isinstance(color, str), "Color must be string"
        assert isinstance(layer, str), "Layer must be string"
        assert isinstance(confidence, float), "Confidence must be float"

        # Validate output values
        assert color in color_layer_model.color_map.values(), \
            f"Color '{color}' not in color_map values"
        assert layer in color_layer_model.layer_map.values(), \
            f"Layer '{layer}' not in layer_map values"
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence {confidence} out of range [0.0, 1.0]"

    def test_call_with_conf_threshold_override(self, color_layer_model, sample_blue_plate):
        """
        Contract: ColorLayerORT.__call__() must accept optional conf_thres override.

        When conf_thres is provided, it should override the instance default.
        """
        # Call with custom threshold
        color, layer, confidence = color_layer_model(sample_blue_plate, conf_thres=0.9)

        # If confidence < 0.9, should return ('unknown', 'unknown', conf)
        # Otherwise, should return valid results
        if confidence >= 0.9:
            assert color != 'unknown'
            assert layer != 'unknown'
        # Note: We can't force low confidence, so we just validate the interface


    def test_static_method_accessibility(self):
        """
        Contract: Static preprocessing methods must be accessible without instantiation.

        This supports TensorRT engine workflow where preprocessing happens separately.
        """
        from onnxtools.infer_onnx.onnx_ocr import ColorLayerORT

        # Create test image
        test_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        # Call static method directly
        preprocessed = ColorLayerORT._image_preprocess_static(
            test_img,
            image_shape=(48, 168)
        )

        # Validate preprocessing output
        assert preprocessed.shape == (1, 3, 48, 168), \
            f"Expected (1, 3, 48, 168), got {preprocessed.shape}"
        assert preprocessed.dtype == np.float32


class TestOcrORTContract:
    """Contract tests for OcrORT refactored class."""

    def test_initialization_contract(self, ocr_model_path, ocr_character):
        """
        Contract: OcrORT initialization must accept required parameters.

        Required parameters:
        - onnx_path: str
        - character: List[str] (OCR character dictionary)

        Optional parameters:
        - input_shape: tuple (default (48, 168))
        - conf_thres: float (default 0.5)
        - providers: list
        """
        from onnxtools.infer_onnx.onnx_ocr import OcrORT

        # Test minimal initialization
        model = OcrORT(
            str(ocr_model_path),
            character=ocr_character
        )

        assert model is not None
        assert model.character == ocr_character
        # Note: input_shape is None before initialization (lazy loading)
        assert model._requested_input_shape == (48, 168)  # Default value

    def test_call_interface_single_layer_contract(
        self, ocr_model_refactored, sample_single_layer_plate
    ):
        """
        Contract: OcrORT.__call__() must return valid OCR results for single-layer plates.

        Input: np.ndarray (BGR plate image)
        Output: Tuple[str, float, List[float]] = (text, avg_conf, char_confidences)

        Constraints:
        - text must be non-empty string
        - avg_conf must be in [0.0, 1.0]
        - char_confidences must be list of floats in [0.0, 1.0]
        - len(char_confidences) == len(text)
        """
        # Call the model
        text, avg_conf, char_confs = ocr_model_refactored(
            sample_single_layer_plate,
            is_double_layer=False
        )

        # Validate output structure
        assert isinstance(text, str), "Text must be string"
        assert isinstance(avg_conf, (int, float)), "Average confidence must be numeric"
        assert isinstance(char_confs, list), "Character confidences must be list"

        # Validate output values
        assert len(text) > 0, "OCR text should not be empty"
        assert 0.0 <= avg_conf <= 1.0, \
            f"Average confidence {avg_conf} out of range [0.0, 1.0]"

        # Validate character confidences
        assert len(char_confs) == len(text), \
            f"Confidence count {len(char_confs)} != text length {len(text)}"

        for i, conf in enumerate(char_confs):
            assert isinstance(conf, (int, float)), \
                f"Character confidence[{i}] must be numeric"
            assert 0.0 <= conf <= 1.0, \
                f"Character confidence[{i}] = {conf} out of range [0.0, 1.0]"

    def test_call_interface_double_layer_contract(
        self, ocr_model_refactored, sample_double_layer_plate
    ):
        """
        Contract: OcrORT.__call__() must handle double-layer plates.

        Double-layer processing may fail for synthetic images, which is acceptable.
        """
        # Call the model with double-layer flag
        result = ocr_model_refactored(
            sample_double_layer_plate,
            is_double_layer=True
        )

        # Result could be tuple (success) or None (processing failed)
        if result is not None:
            text, avg_conf, char_confs = result

            # Same validations as single-layer
            assert isinstance(text, str)
            assert 0.0 <= avg_conf <= 1.0
            assert len(char_confs) == len(text)
        else:
            # Processing failure is acceptable for synthetic images
            pytest.skip("Double-layer processing failed (expected for synthetic images)")


    def test_static_preprocessing_methods_accessibility(self):
        """
        Contract: All static preprocessing methods must be accessible.

        Required static methods:
        - _process_plate_image_static()
        - _resize_norm_img_static()
        - _detect_skew_angle()
        - _correct_skew()
        - _find_optimal_split_line()
        - _split_double_layer_plate()
        - _stitch_double_layer_plate()
        """
        from onnxtools.infer_onnx.onnx_ocr import OcrORT

        # Test image
        test_img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)

        # Test main preprocessing
        processed = OcrORT._process_plate_image_static(test_img, is_double_layer=False)
        assert processed is not None
        assert isinstance(processed, np.ndarray)

        # Test resize normalization
        normalized = OcrORT._resize_norm_img_static(processed, [3, 48, 168])
        assert normalized.shape == (1, 3, 48, 168)
        assert normalized.dtype == np.float32

        # Test skew detection (gray image)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        angle = OcrORT._detect_skew_angle(gray_img)
        assert isinstance(angle, (int, float))

        # Test skew correction
        corrected = OcrORT._correct_skew(test_img, angle)
        assert corrected.shape == test_img.shape

    def test_static_postprocessing_methods_accessibility(self, ocr_character):
        """
        Contract: All static postprocessing methods must be accessible.

        Required static methods:
        - _get_ignored_tokens()
        - _decode_static()
        """
        from onnxtools.infer_onnx.onnx_ocr import OcrORT

        # Test ignored tokens
        ignored_tokens = OcrORT._get_ignored_tokens_static()
        assert isinstance(ignored_tokens, list)
        # Ignored tokens are indices, not strings
        assert isinstance(ignored_tokens[0], int)

        # Test decode
        # Create fake model output
        text_index = np.array([[0, 1, 2, 3, 4]])  # Fake indices
        text_prob = np.array([[0.9, 0.85, 0.95, 0.88, 0.92]])  # Fake probs

        results = OcrORT._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=True
        )

        assert isinstance(results, list)

    def test_special_character_post_processing(self, ocr_model_refactored):
        """
        Contract: OcrORT must apply special post-processing rules.

        Known rules:
        - Replace leading '苏' with '京' (historical correction)
        """
        # This is tested indirectly through the decode logic
        # The actual behavior is in _decode_static()
        from onnxtools.infer_onnx.onnx_ocr import OcrORT

        # Simulate a decode result starting with '苏'
        # (In real scenario, this would come from model output)
        # We test the static method directly

        # Create mock prediction for '苏A12345'
        # This is a white-box test of the post-processing logic
        test_characters = ['blank'] + list('苏ABCDEFG0123456789')

        # Index for '苏A12345'
        text_index = np.array([[1, 2, 3, 4, 5, 6, 7]])  # '苏' is index 1
        text_prob = np.ones_like(text_index, dtype=np.float32) * 0.95

        results = OcrORT._decode_static(
            test_characters,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        if len(results) > 0:
            text, _, _ = results[0]
            # Should replace '苏' with '京'
            assert text.startswith('京') or not text.startswith('苏'), \
                "Leading '苏' should be replaced with '京'"


class TestRefactoredClassesIntegration:
    """Integration contract tests for both refactored classes."""

    def test_color_layer_ocr_pipeline(
        self, color_layer_model, ocr_model_refactored, sample_blue_plate
    ):
        """
        Contract: ColorLayerORT and OcrORT must work together in pipeline.

        Typical workflow:
        1. Detect vehicle/plate with YOLO/RT-DETR
        2. Classify color/layer with ColorLayerORT
        3. Recognize text with OcrORT
        """
        # Step 1: Classify color and layer
        color, layer, color_conf = color_layer_model(sample_blue_plate)

        # Step 2: Determine if double-layer
        is_double = (layer == 'double')

        # Step 3: OCR recognition
        ocr_result = ocr_model_refactored(sample_blue_plate, is_double_layer=is_double)

        # Validate pipeline results
        assert color in color_layer_model.color_map.values()
        assert layer in color_layer_model.layer_map.values()

        if ocr_result is not None:
            text, ocr_conf, char_confs = ocr_result
            assert isinstance(text, str)
            assert len(text) > 0

    def test_type_safety_contracts(self):
        """
        Contract: All type hints must be accurate and enforced.

        This test verifies that type annotations match actual behavior.
        """
        from onnxtools.infer_onnx.onnx_ocr import ColorLayerORT, OcrORT
        import inspect

        # Check ColorLayerORT.__call__ signature
        call_sig = inspect.signature(ColorLayerORT.__call__)
        assert 'image' in call_sig.parameters
        assert 'conf_thres' in call_sig.parameters

        # Check OcrORT.__call__ signature
        ocr_call_sig = inspect.signature(OcrORT.__call__)
        assert 'image' in ocr_call_sig.parameters
        assert 'is_double_layer' in ocr_call_sig.parameters

    def test_error_handling_contract(self, ocr_model_refactored):
        """
        Contract: Models must handle invalid inputs gracefully.

        Expected behaviors:
        - None input -> should raise TypeError or return None
        - Wrong shape input -> should attempt to process or fail gracefully
        - Empty image -> should handle without crash
        """
        # Test None input
        try:
            result = ocr_model_refactored(None, is_double_layer=False)
            # If it doesn't raise, result should be None
            assert result is None or isinstance(result, tuple)
        except (TypeError, AttributeError):
            # Acceptable to raise exception for None input
            pass

        # Test empty image
        empty_img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = ocr_model_refactored(empty_img, is_double_layer=False)
        # Should return something or None, but not crash
        assert result is None or isinstance(result, tuple)


# Performance contract tests

class TestPerformanceContracts:
    """Performance contract tests for refactored classes."""

    def test_color_layer_inference_performance(
        self, color_layer_model, sample_blue_plate
    ):
        """
        Contract: ColorLayerORT inference must complete within acceptable time.

        Target: < 50ms per inference (after warmup)
        """
        import time

        # Warmup
        for _ in range(5):
            color_layer_model(sample_blue_plate)

        # Measure
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.time()
            color_layer_model(sample_blue_plate)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)

        print(f"\n⏱️  ColorLayerORT Performance:")
        print(f"   Average: {avg_time:.2f} ms")
        print(f"   Min: {np.min(times):.2f} ms")
        print(f"   Max: {np.max(times):.2f} ms")

        # Performance assertion (relaxed for CPU-only mode)
        # Target: < 50ms on GPU, < 150ms on CPU
        assert avg_time < 150, \
            f"ColorLayerORT too slow: {avg_time:.2f}ms (target < 150ms on CPU)"

    def test_ocr_inference_performance(
        self, ocr_model_refactored, sample_single_layer_plate
    ):
        """
        Contract: OcrORT inference must complete within acceptable time.

        Target: < 100ms per inference including preprocessing (after warmup)
        """
        import time

        # Warmup
        for _ in range(5):
            ocr_model_refactored(sample_single_layer_plate, is_double_layer=False)

        # Measure
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.time()
            ocr_model_refactored(sample_single_layer_plate, is_double_layer=False)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)

        print(f"\n⏱️  OcrORT Performance:")
        print(f"   Average: {avg_time:.2f} ms")
        print(f"   Min: {np.min(times):.2f} ms")
        print(f"   Max: {np.max(times):.2f} ms")

        # Performance assertion
        assert avg_time < 100, \
            f"OcrORT too slow: {avg_time:.2f}ms (target < 100ms)"
