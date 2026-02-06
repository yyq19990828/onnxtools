"""Unit tests for onnxtools/infer_onnx/onnx_cls.py ClsResult class.

Tests ClsResult data structure:
- Initialization and attributes
- Tuple unpacking (single, dual, multi-branch)
- __repr__, __len__, __getitem__
- Preprocessing static methods
"""

import numpy as np
import pytest

from onnxtools.infer_onnx.onnx_cls import ClsResult, ColorLayerORT


class TestClsResultInit:
    """Test ClsResult initialization."""

    def test_basic_init(self):
        result = ClsResult(
            labels=["blue"],
            confidences=[0.95],
            avg_confidence=0.95,
        )
        assert result.labels == ["blue"]
        assert result.confidences == [0.95]
        assert result.avg_confidence == 0.95
        assert result.logits is None

    def test_dual_branch_init(self):
        result = ClsResult(
            labels=["blue", "single"],
            confidences=[0.95, 0.88],
            avg_confidence=0.915,
        )
        assert len(result.labels) == 2
        assert result.labels[0] == "blue"
        assert result.labels[1] == "single"

    def test_with_logits(self):
        logits = [np.array([[0.1, 0.9]]), np.array([[0.8, 0.2]])]
        result = ClsResult(
            labels=["blue", "single"],
            confidences=[0.9, 0.8],
            avg_confidence=0.85,
            logits=logits,
        )
        assert result.logits is not None
        assert len(result.logits) == 2


class TestClsResultTupleUnpacking:
    """Test backward-compatible tuple unpacking."""

    def test_single_branch_unpacking(self):
        result = ClsResult(labels=["cat"], confidences=[0.95], avg_confidence=0.95)
        label, conf = result
        assert label == "cat"
        assert conf == 0.95

    def test_dual_branch_unpacking(self):
        result = ClsResult(
            labels=["blue", "single"],
            confidences=[0.95, 0.88],
            avg_confidence=0.915,
        )
        color, layer, conf = result
        assert color == "blue"
        assert layer == "single"
        assert conf == 0.915

    def test_multi_branch_unpacking(self):
        result = ClsResult(
            labels=["a", "b", "c"],
            confidences=[0.9, 0.8, 0.7],
            avg_confidence=0.8,
        )
        labels, confidences, conf = result
        assert labels == ["a", "b", "c"]
        assert confidences == [0.9, 0.8, 0.7]


class TestClsResultMethods:
    """Test ClsResult special methods."""

    def test_repr(self):
        result = ClsResult(labels=["blue", "single"], confidences=[0.95, 0.88], avg_confidence=0.915)
        repr_str = repr(result)
        assert "ClsResult" in repr_str
        assert "blue" in repr_str
        assert "single" in repr_str

    def test_len(self):
        result = ClsResult(labels=["blue", "single"], confidences=[0.95, 0.88], avg_confidence=0.915)
        assert len(result) == 2

    def test_getitem(self):
        result = ClsResult(labels=["blue", "single"], confidences=[0.95, 0.88], avg_confidence=0.915)
        label, conf = result[0]
        assert label == "blue"
        assert conf == 0.95

        label, conf = result[1]
        assert label == "single"
        assert conf == 0.88

    def test_getitem_out_of_range(self):
        result = ClsResult(labels=["blue"], confidences=[0.95], avg_confidence=0.95)
        with pytest.raises(IndexError):
            result[5]


class TestColorLayerORTPreprocess:
    """Test ColorLayerORT.preprocess static method (no model needed)."""

    def test_output_shape(self):
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        tensor, scale, orig_shape = ColorLayerORT.preprocess(img, (48, 168))
        assert tensor.shape == (1, 3, 48, 168)

    def test_output_dtype(self):
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        tensor, _, _ = ColorLayerORT.preprocess(img, (48, 168))
        assert tensor.dtype == np.float32

    def test_normalization_range(self):
        """Output should be normalized to [-1, 1]."""
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        tensor, _, _ = ColorLayerORT.preprocess(img, (48, 168))
        assert tensor.min() >= -1.0 - 1e-6
        assert tensor.max() <= 1.0 + 1e-6

    def test_scale_is_one(self):
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        _, scale, _ = ColorLayerORT.preprocess(img, (48, 168))
        assert scale == 1.0

    def test_original_shape(self):
        img = np.random.randint(0, 255, (140, 440, 3), dtype=np.uint8)
        _, _, orig_shape = ColorLayerORT.preprocess(img, (48, 168))
        assert orig_shape == (140, 440)

    def test_different_input_sizes(self):
        """Preprocessing should handle various input sizes."""
        for h, w in [(50, 100), (200, 600), (48, 168)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            tensor, _, orig_shape = ColorLayerORT.preprocess(img, (48, 168))
            assert tensor.shape == (1, 3, 48, 168)
            assert orig_shape == (h, w)


class TestColorLayerORTSoftmax:
    """Test ColorLayerORT._softmax static method."""

    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = ColorLayerORT._softmax(x)
        np.testing.assert_almost_equal(result.sum(), 1.0)

    def test_numerical_stability(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = ColorLayerORT._softmax(x)
        np.testing.assert_almost_equal(result.sum(), 1.0)
        assert not np.any(np.isnan(result))
