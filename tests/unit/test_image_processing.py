"""Unit tests for onnxtools/utils/image_processing.py

Tests UltralyticsLetterBox preprocessing:
- Output shape and dtype
- Scaling behavior (scaleup, no-scaleup)
- Padding and centering
- Various input shapes
"""

import numpy as np

from onnxtools.utils.image_processing import UltralyticsLetterBox

_RNG_SEED = 42


def _make_image(height: int, width: int, seed: int = _RNG_SEED) -> np.ndarray:
    """Create a deterministic test image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (height, width, 3), dtype=np.uint8)


class TestUltralyticsLetterBox:
    """Test UltralyticsLetterBox preprocessing."""

    def test_output_shape_square_image(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(640, 640)
        tensor, scale, orig_shape, ratio_pad = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(480, 640)
        tensor, scale, orig_shape, ratio_pad = lb(img)
        assert tensor.dtype == np.float32

    def test_output_range(self) -> None:
        """Output should be normalized to [0, 1]."""
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(480, 640)
        tensor, _, _, _ = lb(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_original_shape_returned(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(480, 640)
        _, _, orig_shape, _ = lb(img)
        assert orig_shape == (480, 640)

    def test_rectangular_input_gets_padded(self) -> None:
        """Non-square input should be padded to target shape."""
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(480, 640)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_no_scaleup(self) -> None:
        """Small images should not be scaled up when scaleup=False."""
        lb = UltralyticsLetterBox(new_shape=(640, 640), scaleup=False)
        img = _make_image(320, 320)
        tensor, scale, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert scale <= 1.0

    def test_scale_fill(self) -> None:
        """scale_fill should stretch without maintaining aspect ratio."""
        lb = UltralyticsLetterBox(new_shape=(640, 640), scale_fill=True)
        img = _make_image(480, 320)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_different_target_shapes(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(320, 320))
        img = _make_image(640, 640)
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_wide_image(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(200, 800)
        tensor, _, orig_shape, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert orig_shape == (200, 800)

    def test_tall_image(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640))
        img = _make_image(800, 200)
        tensor, _, orig_shape, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert orig_shape == (800, 200)

    def test_half_precision(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640), half=True)
        img = _make_image(480, 640)
        tensor, _, _, _ = lb(img)
        assert tensor.dtype == np.float16

    def test_custom_padding_value(self) -> None:
        lb = UltralyticsLetterBox(new_shape=(640, 640), padding_value=0)
        img = np.zeros((320, 640, 3), dtype=np.uint8)  # half-height black image
        tensor, _, _, _ = lb(img)
        assert tensor.shape == (1, 3, 640, 640)
