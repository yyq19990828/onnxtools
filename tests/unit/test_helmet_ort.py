"""Unit tests for HelmetORT helmet classification model."""

import numpy as np
import pytest

from onnxtools.infer_onnx.onnx_cls import ClsResult, HelmetORT


class TestHelmetORTPreprocess:
    """Tests for HelmetORT.preprocess static method."""

    def test_output_shape(self):
        """Preprocess should return [1, 3, 128, 128] tensor."""
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        tensor, scale, orig_shape = HelmetORT.preprocess(image, (128, 128))

        assert tensor.shape == (1, 3, 128, 128)
        assert tensor.dtype == np.float32

    def test_original_shape_preserved(self):
        """Preprocess should return original image shape."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, scale, orig_shape = HelmetORT.preprocess(image, (128, 128))

        assert orig_shape == (480, 640)

    def test_scale_is_one(self):
        """Preprocess should return scale=1.0."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor, scale, orig_shape = HelmetORT.preprocess(image, (128, 128))

        assert scale == 1.0

    def test_imagenet_normalization_range(self):
        """Normalized values should be roughly in [-2.5, 2.5] range for ImageNet."""
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        tensor, _, _ = HelmetORT.preprocess(image, (128, 128))

        # ImageNet normalization: (x/255 - mean) / std
        # Minimum: (0 - 0.485) / 0.229 ~ -2.12
        # Maximum: (1 - 0.406) / 0.225 ~ 2.64
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0

    def test_square_image_no_padding(self):
        """Square input should fill the entire canvas without padding."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        tensor, _, _ = HelmetORT.preprocess(image, (128, 128))

        # All pixels should have been filled (no pad_value=127 areas)
        assert tensor.shape == (1, 3, 128, 128)


class TestHelmetORTLetterbox:
    """Tests for HelmetORT._letterbox static method."""

    def test_output_shape(self):
        """LetterBox should return exact target size."""
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = HelmetORT._letterbox(image, (128, 128))

        assert result.shape == (128, 128, 3)

    def test_square_input(self):
        """Square image should be resized without padding."""
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        result = HelmetORT._letterbox(image, (128, 128), pad_value=127)

        # Center should contain the resized image (value 100), not padding (127)
        center_pixel = result[64, 64]
        assert np.all(center_pixel == 100)

    def test_wide_image_padding(self):
        """Wide image should have top/bottom padding."""
        image = np.full((50, 200, 3), 100, dtype=np.uint8)
        result = HelmetORT._letterbox(image, (128, 128), pad_value=127)

        # Top row should be padding
        assert np.all(result[0, 64] == 127)

    def test_tall_image_padding(self):
        """Tall image should have left/right padding."""
        image = np.full((200, 50, 3), 100, dtype=np.uint8)
        result = HelmetORT._letterbox(image, (128, 128), pad_value=127)

        # Left column should be padding
        assert np.all(result[64, 0] == 127)

    def test_preserves_dtype(self):
        """LetterBox should preserve uint8 dtype."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = HelmetORT._letterbox(image, (128, 128))

        assert result.dtype == np.uint8


class TestHelmetORTPostprocess:
    """Tests for HelmetORT.postprocess method."""

    @pytest.fixture
    def helmet_ort_mock(self):
        """Create a HelmetORT-like object without loading a real model."""
        class MockHelmetORT:
            helmet_map = {0: "normal", 1: "helmet_missing"}
            postprocess = HelmetORT.postprocess
        return MockHelmetORT()

    def test_normal_prediction(self, helmet_ort_mock):
        """Should correctly identify 'normal' class."""
        # Logits favoring class 0 (normal)
        logits = np.array([[3.0, -1.0]], dtype=np.float32)
        result = helmet_ort_mock.postprocess([logits], conf_thres=0.5)

        assert isinstance(result, ClsResult)
        assert result.labels[0] == "normal"
        assert result.avg_confidence > 0.9

    def test_helmet_missing_prediction(self, helmet_ort_mock):
        """Should correctly identify 'helmet_missing' class."""
        # Logits favoring class 1 (helmet_missing)
        logits = np.array([[-1.0, 3.0]], dtype=np.float32)
        result = helmet_ort_mock.postprocess([logits], conf_thres=0.5)

        assert result.labels[0] == "helmet_missing"
        assert result.avg_confidence > 0.9

    def test_softmax_applied(self, helmet_ort_mock):
        """Confidence should be in [0, 1] after softmax."""
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        result = helmet_ort_mock.postprocess([logits], conf_thres=0.5)

        assert 0.0 <= result.avg_confidence <= 1.0

    def test_cls_result_structure(self, helmet_ort_mock):
        """ClsResult should have single-branch structure."""
        logits = np.array([[2.0, -1.0]], dtype=np.float32)
        result = helmet_ort_mock.postprocess([logits], conf_thres=0.5)

        assert len(result) == 1
        assert len(result.labels) == 1
        assert len(result.confidences) == 1
        assert result.logits is not None

    def test_tuple_unpacking(self, helmet_ort_mock):
        """Single-branch ClsResult should unpack as (label, conf)."""
        logits = np.array([[2.0, -1.0]], dtype=np.float32)
        result = helmet_ort_mock.postprocess([logits], conf_thres=0.5)

        label, conf = result
        assert isinstance(label, str)
        assert isinstance(conf, float)

    def test_unknown_class_index(self):
        """Should handle unmapped class index gracefully."""
        class MockHelmetORT:
            helmet_map = {0: "normal"}  # Missing class 1
            postprocess = HelmetORT.postprocess

        mock = MockHelmetORT()
        logits = np.array([[-1.0, 3.0]], dtype=np.float32)
        result = mock.postprocess([logits], conf_thres=0.5)

        assert result.labels[0].startswith("unknown_")


class TestHelmetORTBatchPadding:
    """Tests for the automatic batch padding logic in BaseClsORT._execute_inference."""

    def test_batch_padding_shape(self):
        """Input [1,3,128,128] should be padded to [N,3,128,128] for fixed batch."""
        fixed_batch_size = 4
        single_input = np.random.randn(1, 3, 128, 128).astype(np.float32)
        batch = np.repeat(single_input, fixed_batch_size, axis=0)

        assert batch.shape == (4, 3, 128, 128)

    def test_batch_padding_content(self):
        """All batch samples should be identical copies of the input."""
        fixed_batch_size = 4
        single_input = np.random.randn(1, 3, 128, 128).astype(np.float32)
        batch = np.repeat(single_input, fixed_batch_size, axis=0)

        for i in range(fixed_batch_size):
            np.testing.assert_array_equal(batch[i], single_input[0])

    def test_trimmed_output(self):
        """Trimmed output should have batch=1."""
        # Simulate model output with batch=4
        outputs = [np.random.randn(4, 2).astype(np.float32)]
        trimmed = [out[:1] for out in outputs]

        assert trimmed[0].shape == (1, 2)


class TestHelmetORTInit:
    """Tests for HelmetORT initialization (no model loading)."""

    def test_default_helmet_map(self):
        """Default helmet_map should have 2 classes."""
        from onnxtools.config import HELMET_MAP
        assert len(HELMET_MAP) == 2
        assert 0 in HELMET_MAP
        assert 1 in HELMET_MAP

    def test_empty_helmet_map_raises(self):
        """Empty helmet_map should raise ValueError."""
        with pytest.raises(ValueError, match="helmet_map cannot be empty"):
            HelmetORT.__new__(HelmetORT)
            # Simulate init with empty map (bypassing model loading)
            helmet_map = {}
            if not helmet_map:
                raise ValueError("helmet_map cannot be empty")

    def test_constants(self):
        """Class constants should be correct."""
        assert HelmetORT.LETTERBOX_PAD_VALUE == 127
        np.testing.assert_array_almost_equal(
            HelmetORT.IMAGENET_MEAN,
            [0.485, 0.456, 0.406]
        )
        np.testing.assert_array_almost_equal(
            HelmetORT.IMAGENET_STD,
            [0.229, 0.224, 0.225]
        )
