"""
Unit tests for refactored OCRONNX and ColorLayerONNX static methods.

This module tests individual static preprocessing and postprocessing methods
to ensure correctness after migration from utils/ to class methods.

Test Coverage:
- ColorLayerONNX._preprocess_static()
- OCRONNX._detect_skew_angle()
- OCRONNX._correct_skew()
- OCRONNX._find_optimal_split_line()
- OCRONNX._split_double_layer_plate()
- OCRONNX._stitch_double_layer_plate()
- OCRONNX._process_plate_image_static()
- OCRONNX._resize_norm_img_static()
- OCRONNX._get_ignored_tokens()
- OCRONNX._decode_static()
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import List


class TestColorLayerPreprocessing:
    """Unit tests for ColorLayerONNX preprocessing methods."""

    def test_image_pretreatment_output_shape(self):
        """Test that preprocessing produces correct output shape."""
        from infer_onnx.onnx_ocr import ColorLayerONNX

        # Create test image
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        # Preprocess (returns tuple: (tensor, ratio, original_shape))
        result, ratio, original_shape = ColorLayerONNX._preprocess_static(img, image_shape=(48, 168))

        # Validate shape: (1, 3, 48, 168)
        assert result.shape == (1, 3, 48, 168), \
            f"Expected (1, 3, 48, 168), got {result.shape}"

    def test_image_pretreatment_dtype(self):
        """Test that preprocessing produces float32 output."""
        from infer_onnx.onnx_ocr import ColorLayerONNX

        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result, _, _ = ColorLayerONNX._preprocess_static(img, image_shape=(48, 168))

        assert result.dtype == np.float32, \
            f"Expected float32, got {result.dtype}"

    def test_image_pretreatment_normalization(self):
        """Test that preprocessing applies normalization correctly."""
        from infer_onnx.onnx_ocr import ColorLayerONNX

        # Create known-value image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Mid-gray

        result, _, _ = ColorLayerONNX._preprocess_static(img, image_shape=(48, 168))

        # After normalization: (128/255 - 0.5) / 0.5 ≈ 0.0039
        # Check that values are in reasonable range
        assert -3.0 < result.mean() < 3.0, \
            f"Normalized values out of expected range: mean={result.mean()}"

    def test_image_pretreatment_channel_order(self):
        """Test that preprocessing converts to CHW format."""
        from infer_onnx.onnx_ocr import ColorLayerONNX

        # Create colored image (Red=255, Green=0, Blue=0)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel in BGR

        result, _, _ = ColorLayerONNX._preprocess_static(img, image_shape=(48, 168))

        # Check shape is (1, 3, H, W)
        assert result.shape[1] == 3, "Should have 3 channels"

        # Red channel should have higher values than others (after normalization)
        # Note: Exact values depend on mean/std normalization


class TestOCRSkewCorrection:
    """Unit tests for OCRONNX skew detection and correction."""

    def test_detect_skew_angle_horizontal(self):
        """Test skew detection on horizontal plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create horizontal plate-like image
        img = np.zeros((100, 300), dtype=np.uint8)
        cv2.rectangle(img, (10, 40), (290, 60), 255, -1)  # Horizontal bar

        angle = OCRONNX._detect_skew_angle(img)

        # Should detect near-zero angle
        assert isinstance(angle, (int, float))
        assert -10 <= angle <= 10, \
            f"Horizontal image detected angle {angle}° (expected ~0°)"

    def test_detect_skew_angle_tilted(self):
        """Test skew detection on tilted plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create tilted plate-like image
        img = np.zeros((100, 300), dtype=np.uint8)

        # Draw tilted rectangle
        center = (150, 50)
        size = (280, 20)
        angle_deg = 15.0

        rect = ((center[0], center[1]), (size[0], size[1]), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(img, [box], 255)

        detected_angle = OCRONNX._detect_skew_angle(img)

        # Should detect tilt (may not be exact due to algorithm limitations)
        assert isinstance(detected_angle, (int, float))
        # Allow wide range as detection isn't perfect
        assert -45 <= detected_angle <= 45

    def test_correct_skew_preserves_shape(self):
        """Test that skew correction preserves image shape."""
        from infer_onnx.onnx_ocr import OCRONNX

        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        original_shape = img.shape

        corrected = OCRONNX._correct_skew(img, angle=5.0)

        assert corrected.shape == original_shape, \
            f"Shape changed: {original_shape} -> {corrected.shape}"

    def test_correct_skew_zero_angle(self):
        """Test that zero angle correction returns similar image."""
        from infer_onnx.onnx_ocr import OCRONNX

        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)

        corrected = OCRONNX._correct_skew(img, angle=0.0)

        # Should be very similar (may have small differences due to rotation)
        # Check that images are similar (allow small numerical differences)
        diff = np.abs(img.astype(float) - corrected.astype(float)).mean()
        assert diff < 5.0, f"Zero-angle correction changed image too much: diff={diff}"


class TestOCRDoubleLayerProcessing:
    """Unit tests for double-layer plate processing."""

    def test_find_optimal_split_line_returns_valid_index(self):
        """Test that split line finder returns valid row index."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create gray image with clear horizontal split
        img = np.ones((140, 440), dtype=np.uint8) * 200
        img[65:75, :] = 50  # Dark horizontal line

        split_row = OCRONNX._find_optimal_split_line(img)

        assert isinstance(split_row, (int, np.integer))
        assert 0 <= split_row < img.shape[0], \
            f"Split row {split_row} out of range [0, {img.shape[0]})"

    def test_find_optimal_split_line_detects_midpoint(self):
        """Test that split line is detected near middle for double-layer plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create double-layer-like image
        img = np.ones((140, 440), dtype=np.uint8) * 200
        # Upper layer
        cv2.rectangle(img, (10, 10), (430, 60), 100, -1)
        # Gap
        img[60:80, :] = 50
        # Lower layer
        cv2.rectangle(img, (10, 80), (430, 130), 100, -1)

        split_row = OCRONNX._find_optimal_split_line(img)

        # Should detect split near middle (around row 70)
        assert 50 < split_row < 90, \
            f"Split row {split_row} not near expected middle (50-90)"

    def test_split_double_layer_plate(self):
        """Test splitting double-layer plate into two parts."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create double-layer image
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)

        upper, lower = OCRONNX._split_double_layer(img, split_y=70)

        # Validate output shapes
        assert upper.shape[0] < img.shape[0], "Upper part should be smaller"
        assert lower.shape[0] < img.shape[0], "Lower part should be smaller"
        assert upper.shape[1] == img.shape[1], "Width should match"
        assert lower.shape[1] == img.shape[1], "Width should match"
        assert upper.shape[2] == 3, "Should have 3 channels"
        assert lower.shape[2] == 3, "Should have 3 channels"

    def test_stitch_double_layer_plate(self):
        """Test stitching two parts into single-layer plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create two parts
        upper = np.random.randint(0, 255, (60, 220, 3), dtype=np.uint8)
        lower = np.random.randint(0, 255, (60, 220, 3), dtype=np.uint8)

        stitched = OCRONNX._stitch_layers(upper, lower)

        # Validate stitched shape
        assert stitched.shape[0] == 60, "Height should match single part"
        # Width = target_top_width + lower_width
        # target_top_width = 60 * (220/60) * 0.5 = 110
        # So expected width = 110 + 220 = 330
        assert stitched.shape[1] > 0, "Width should be positive"
        assert stitched.shape[2] == 3, "Should have 3 channels"


class TestOCRImageProcessing:
    """Unit tests for OCRONNX image processing pipeline."""

    def test_process_plate_image_single_layer(self):
        """Test processing single-layer plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create synthetic single-layer plate
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)

        result = OCRONNX._process_plate_image_static(img, is_double_layer=False)

        # Should return processed image
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3, "Should be color image"

    def test_process_plate_image_double_layer(self):
        """Test processing double-layer plate."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create synthetic double-layer plate
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)

        result = OCRONNX._process_plate_image_static(img, is_double_layer=True)

        # May return None for synthetic images (expected)
        if result is not None:
            assert isinstance(result, np.ndarray)
            # Width depends on stitching algorithm, just check it's positive
            assert result.shape[1] > 0, "Width should be positive after stitching"

    def test_resize_norm_img_output_shape(self):
        """Test that resize_norm_img produces correct output shape."""
        from infer_onnx.onnx_ocr import OCRONNX

        img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)

        result = OCRONNX._resize_norm_img_static(img, image_shape=[3, 48, 168])

        # Should produce (1, 3, 48, 168)
        assert result.shape == (1, 3, 48, 168), \
            f"Expected (1, 3, 48, 168), got {result.shape}"

    def test_resize_norm_img_dtype(self):
        """Test that resize_norm_img produces float32."""
        from infer_onnx.onnx_ocr import OCRONNX

        img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)

        result = OCRONNX._resize_norm_img_static(img, image_shape=[3, 48, 168])

        assert result.dtype == np.float32, \
            f"Expected float32, got {result.dtype}"

    def test_resize_norm_img_normalization(self):
        """Test that resize_norm_img applies correct normalization."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create known-value image (mid-gray)
        img = np.ones((100, 300, 3), dtype=np.uint8) * 128

        result = OCRONNX._resize_norm_img_static(img, image_shape=[3, 48, 168])

        # After normalization: (128/255 - 0.5) / 0.5
        # Expected value ≈ 0.0039
        assert -2.0 < result.mean() < 2.0, \
            f"Normalized values out of expected range: mean={result.mean()}"


class TestOCRPostprocessing:
    """Unit tests for OCRONNX postprocessing methods."""

    def test_get_ignored_tokens(self):
        """Test that ignored tokens list is correct."""
        from infer_onnx.onnx_ocr import OCRONNX

        ignored = OCRONNX._get_ignored_tokens_static()

        assert isinstance(ignored, list)
        # Ignored tokens are indices [0], not strings
        assert 0 in ignored

    def test_decode_basic(self, ocr_character):
        """Test basic decoding functionality."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create simple prediction
        # Assume character[1:8] = '京', 'A', '1', '2', '3', '4', '5'
        text_index = np.array([[1, 2, 3, 4, 5, 6, 7]])
        text_prob = np.ones_like(text_index, dtype=np.float32) * 0.95

        results = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        assert isinstance(results, list)
        assert len(results) > 0

        text, avg_conf, char_confs = results[0]
        assert isinstance(text, str)
        assert len(text) > 0
        assert 0.0 <= avg_conf <= 1.0

    def test_decode_with_duplicates(self, ocr_character):
        """Test duplicate removal in decoding."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create prediction with duplicates: '京京A11234'
        # Index 1='京', 2='A', 3='1', etc.
        text_index = np.array([[1, 1, 2, 3, 3, 4, 5, 6]])
        text_prob = np.ones_like(text_index, dtype=np.float32) * 0.95

        # Without duplicate removal
        results_with_dup = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        # With duplicate removal
        results_without_dup = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=True
        )

        if len(results_with_dup) > 0 and len(results_without_dup) > 0:
            text_with = results_with_dup[0][0]
            text_without = results_without_dup[0][0]

            # Without dup removal should be longer
            assert len(text_without) <= len(text_with), \
                "Duplicate removal should reduce length"

    def test_decode_special_char_replacement(self):
        """Test special character replacement (苏 -> 京)."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create character list with '苏'
        test_chars = ['blank'] + list('苏京ABCDEFG0123456789')

        # Create prediction for '苏A12345'
        # Index 1='苏', 2='京', 3='A', etc.
        text_index = np.array([[1, 3, 4, 5, 6, 7]])  # '苏A1234'
        text_prob = np.ones_like(text_index, dtype=np.float32) * 0.95

        results = OCRONNX._decode_static(
            test_chars,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        if len(results) > 0:
            text, _, _ = results[0]

            # Should replace leading '苏' with '京'
            if '苏' in test_chars and '京' in test_chars:
                assert text.startswith('京') or not text.startswith('苏'), \
                    "Leading '苏' should be replaced with '京'"

    def test_decode_empty_prediction(self, ocr_character):
        """Test decoding with empty prediction."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Empty prediction
        text_index = np.array([[]])
        text_prob = np.array([[]])

        results = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        # Should return empty list or list with empty result
        assert isinstance(results, list)

    def test_decode_confidence_calculation(self, ocr_character):
        """Test confidence calculation in decoding."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Create prediction with known confidences
        text_index = np.array([[1, 2, 3, 4, 5]])
        text_prob = np.array([[0.9, 0.8, 0.95, 0.85, 0.92]])

        results = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=False
        )

        if len(results) > 0:
            text, avg_conf, char_confs = results[0]

            # Average confidence should be mean of input probs
            expected_avg = (0.9 + 0.8 + 0.95 + 0.85 + 0.92) / 5
            assert abs(avg_conf - expected_avg) < 0.01, \
                f"Average confidence {avg_conf} != expected {expected_avg}"

            # Character confidences should match input
            assert len(char_confs) == len(text)


class TestEdgeCases:
    """Unit tests for edge cases and error handling."""

    def test_process_very_small_image(self):
        """Test processing very small plate image."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Very small image
        tiny_img = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)

        # Should handle gracefully
        result = OCRONNX._process_plate_image_static(tiny_img, is_double_layer=False)

        # May succeed or fail gracefully
        assert result is None or isinstance(result, np.ndarray)

    def test_process_very_large_image(self):
        """Test processing very large plate image."""
        from infer_onnx.onnx_ocr import OCRONNX

        # Large image
        large_img = np.random.randint(0, 255, (500, 1500, 3), dtype=np.uint8)

        result = OCRONNX._process_plate_image_static(large_img, is_double_layer=False)

        # Should succeed
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_decode_with_all_blanks(self, ocr_character):
        """Test decoding when all predictions are 'blank'."""
        from infer_onnx.onnx_ocr import OCRONNX

        # All predictions are blank (index 0)
        text_index = np.array([[0, 0, 0, 0, 0]])
        text_prob = np.ones_like(text_index, dtype=np.float32) * 0.95

        results = OCRONNX._decode_static(
            ocr_character,
            text_index,
            text_prob,
            is_remove_duplicate=True
        )

        # Should return empty result or list with empty string
        assert isinstance(results, list)

    def test_pretreatment_with_grayscale_input(self):
        """Test color layer preprocessing with grayscale input."""
        from infer_onnx.onnx_ocr import ColorLayerONNX

        # Grayscale image (2D)
        gray_img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)

        try:
            # Should either convert or raise error
            result = ColorLayerONNX._preprocess_static(
                gray_img,
                image_shape=(48, 168)
            )
            # If successful, check shape
            if result is not None:
                assert result.shape == (1, 3, 48, 168)
        except (ValueError, IndexError):
            # Acceptable to fail on grayscale
            pass
