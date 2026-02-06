"""Unit tests for onnxtools/utils/image_processing.py

Tests UltralyticsLetterBox preprocessing:
- Output shape and dtype
- Scaling behavior (scaleup, no-scaleup)
- Padding and centering
- Various input shapes
"""

import numpy as np

from onnxtools.utils.image_processing import UltralyticsLetterBox


class TestUltralyticsLetterBox:
    """Test UltralyticsLetterBox preprocessing."""

    def test_output_shape_square_image(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor, scale, orig_shape, ratio_pad = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, scale, orig_shape, ratio_pad = lb(img)
        assert tensor.dtype == np.float32

    def test_output_range(self):
        """Output should be normalized to [0, 1]."""
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = lb(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_original_shape_returned(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, _, orig_shape, _ = lb(img)
        assert orig_shape == (480, 640)

    def test_rectangular_input_gets_padded(self):
        """Non-square input should be padded to target shape."""
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_no_scaleup(self):
        """Small images should not be scaled up when scaleup=False."""
        lb = UltralyticsLetterBox(new_shape=(640, 640), scaleup=False)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        tensor, scale, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert scale <= 1.0

    def test_scale_fill(self):
        """scale_fill should stretch without maintaining aspect ratio."""
        lb = UltralyticsLetterBox(new_shape=(640, 640), scale_fill=True)
        img = np.random.randint(0, 255, (480, 320, 3), dtype=np.uint8)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_different_target_shapes(self):
        lb = UltralyticsLetterBox(new_shape=(320, 320))
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_wide_image(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (200, 800, 3), dtype=np.uint8)
        tensor, _, orig_shape, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert orig_shape == (200, 800)

    def test_tall_image(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = np.random.randint(0, 255, (800, 200, 3), dtype=np.uint8)
        tensor, _, orig_shape, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert orig_shape == (800, 200)

    def test_half_precision(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640), half=True)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = lb(img)
        assert tensor.dtype == np.float16

    def test_custom_padding_value(self):
        lb = UltralyticsLetterBox(new_shape=(640, 640), padding_value=0)
        img = np.zeros((320, 640, 3), dtype=np.uint8)  # half-height black image
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
