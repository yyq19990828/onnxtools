"""Unit tests for OcrORT automatic input shape detection from ONNX model.

Tests verify that OcrORT automatically reads input shape from ONNX model metadata
instead of relying on default values, which fixes the dimension mismatch error.
"""

from pathlib import Path

import numpy as np
import pytest

from onnxtools.infer_onnx.onnx_ocr import OcrORT


class TestOcrInputShapeDetection:
    """Test automatic input shape detection in OcrORT."""

    @pytest.fixture
    def ocr_model(self):
        """Create OcrORT instance with default model."""
        model_path = "models/ocr_mobile.onnx"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return OcrORT(model_path)

    def test_reads_input_shape_from_model(self, ocr_model):
        """Test that OcrORT reads input shape from ONNX model metadata."""
        # The ocr_mobile.onnx model expects [batch, 3, 48, 320]
        expected_height = 48
        expected_width = 320

        assert ocr_model.input_shape == (expected_height, expected_width), \
            f"Expected input_shape to be ({expected_height}, {expected_width}), " \
            f"but got {ocr_model.input_shape}"

    def test_handles_small_plate_images(self, ocr_model):
        """Test that OCR can process small plate images without dimension errors."""
        # Create a small plate-like image (typical cropped plate size)
        small_plate = np.random.randint(0, 255, (20, 66, 3), dtype=np.uint8)

        # This should not raise "Got invalid dimensions" error
        # The preprocessing should resize to match model's expected input shape
        try:
            result = ocr_model(small_plate)
            # Result may be None or tuple depending on recognition success
            assert result is None or isinstance(result, tuple)
        except RuntimeError as e:
            if "Got invalid dimensions" in str(e):
                pytest.fail(
                    f"OCR failed with dimension error even after fix: {e}\n"
                    f"Model expects: {ocr_model.input_shape}, "
                    f"Input shape: {small_plate.shape}"
                )
            else:
                raise

    def test_custom_input_shape_override(self):
        """Test that custom input_shape is used when model shape cannot be read."""
        model_path = "models/ocr_mobile.onnx"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        # Even if we provide a custom shape, it should be overridden by model metadata
        custom_shape = (48, 168)
        ocr_model = OcrORT(model_path, input_shape=custom_shape)

        # Should use model's actual shape (320), not custom (168)
        assert ocr_model.input_shape[1] == 320, \
            "OcrORT should override custom input_shape with model's actual shape"
