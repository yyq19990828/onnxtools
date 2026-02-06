"""Unit tests for onnxtools/utils/nms.py

Tests NMS utility functions:
- softmax, sigmoid, xywh2xyxy
- non_max_suppression (modern and legacy YOLO formats)
"""

import numpy as np
import pytest

from onnxtools.utils.nms import non_max_suppression, sigmoid, softmax, xywh2xyxy

_RNG_SEED = 42


class TestSoftmax:
    """Test softmax function."""

    def test_basic(self) -> None:
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        assert result.shape == x.shape
        np.testing.assert_almost_equal(result.sum(axis=1), [1.0])

    def test_uniform_input(self) -> None:
        x = np.array([[1.0, 1.0, 1.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(result, [[1 / 3, 1 / 3, 1 / 3]])

    def test_large_values_numerical_stability(self) -> None:
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(result.sum(axis=1), [1.0])

    def test_batch_dimension(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result.sum(axis=1), [1.0, 1.0])


class TestSigmoid:
    """Test sigmoid function."""

    def test_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        assert sigmoid(-100.0) == pytest.approx(0.0)

    def test_array(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        result = sigmoid(x)
        assert result[1] == pytest.approx(0.5)
        assert result[0] < 0.5
        assert result[2] > 0.5

    def test_symmetry(self) -> None:
        x = np.array([2.0])
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)


class TestXywh2xyxy:
    """Test xywh2xyxy coordinate conversion."""

    def test_single_box(self) -> None:
        # center_x=100, center_y=100, w=50, h=30
        xywh = np.array([[100.0, 100.0, 50.0, 30.0]])
        xyxy = xywh2xyxy(xywh)
        expected = np.array([[75.0, 85.0, 125.0, 115.0]])
        np.testing.assert_array_almost_equal(xyxy, expected)

    def test_multiple_boxes(self) -> None:
        xywh = np.array(
            [
                [50.0, 50.0, 20.0, 20.0],
                [100.0, 100.0, 40.0, 60.0],
            ]
        )
        xyxy = xywh2xyxy(xywh)
        assert xyxy.shape == (2, 4)
        # First box: center 50,50 size 20x20 -> (40,40,60,60)
        np.testing.assert_array_almost_equal(xyxy[0], [40.0, 40.0, 60.0, 60.0])

    def test_zero_size_box(self) -> None:
        xywh = np.array([[50.0, 50.0, 0.0, 0.0]])
        xyxy = xywh2xyxy(xywh)
        np.testing.assert_array_almost_equal(xyxy, [[50.0, 50.0, 50.0, 50.0]])

    def test_does_not_modify_input(self) -> None:
        xywh = np.array([[100.0, 100.0, 50.0, 30.0]])
        original = xywh.copy()
        xywh2xyxy(xywh)
        np.testing.assert_array_equal(xywh, original)


class TestNonMaxSuppression:
    """Test non_max_suppression function."""

    def _make_prediction(
        self,
        boxes_xywh: np.ndarray,
        class_scores: np.ndarray,
        batch: bool = True,
    ) -> np.ndarray:
        """Helper: create prediction array in modern YOLO format [batch, anchors, 4+num_classes]."""
        pred = np.concatenate([boxes_xywh, class_scores], axis=1)
        if batch:
            pred = pred[np.newaxis, ...]  # add batch dim
        return pred.astype(np.float32)

    def test_empty_batch(self) -> None:
        pred = np.zeros((1, 5, 6), dtype=np.float32)  # all zeros
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        assert len(results) == 1
        assert results[0].shape[0] == 0

    def test_single_detection_modern_yolo(self) -> None:
        """Modern YOLO: no objectness, [batch, anchors, 4 + num_classes]."""
        boxes = np.array([[320.0, 320.0, 100.0, 100.0]])  # xywh
        scores = np.array([[0.9, 0.1]])  # 2 classes, class 0 wins
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        assert results[0].shape[0] == 1
        assert results[0][0, 5] == 0  # class 0
        assert results[0][0, 4] == pytest.approx(0.9)

    def test_nms_suppresses_overlapping(self) -> None:
        """Two highly overlapping boxes, one should be suppressed."""
        boxes = np.array(
            [
                [100.0, 100.0, 50.0, 50.0],
                [105.0, 105.0, 50.0, 50.0],  # highly overlapping
            ]
        )
        scores = np.array([[0.9, 0.1], [0.8, 0.1]])
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.3)
        assert results[0].shape[0] == 1  # one suppressed

    def test_nms_keeps_non_overlapping(self) -> None:
        """Two non-overlapping boxes should both be kept."""
        boxes = np.array(
            [
                [100.0, 100.0, 50.0, 50.0],
                [500.0, 500.0, 50.0, 50.0],  # far apart
            ]
        )
        scores = np.array([[0.9, 0.1], [0.8, 0.1]])
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        assert results[0].shape[0] == 2

    def test_confidence_threshold(self) -> None:
        """Boxes below confidence threshold should be filtered."""
        boxes = np.array(
            [
                [100.0, 100.0, 50.0, 50.0],
                [300.0, 300.0, 50.0, 50.0],
            ]
        )
        scores = np.array([[0.9, 0.1], [0.3, 0.1]])  # second box below 0.5
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        assert results[0].shape[0] == 1

    def test_class_filter(self) -> None:
        """Only specified classes should be kept."""
        boxes = np.array(
            [
                [100.0, 100.0, 50.0, 50.0],
                [300.0, 300.0, 50.0, 50.0],
            ]
        )
        scores = np.array([[0.9, 0.1], [0.1, 0.9]])  # class 0 and class 1
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5, classes=[0])
        assert results[0].shape[0] == 1
        assert results[0][0, 5] == 0

    def test_max_det_limit(self) -> None:
        """Should respect max_det limit."""
        rng = np.random.RandomState(_RNG_SEED)
        n = 500
        boxes = rng.rand(n, 4).astype(np.float32) * 600
        boxes[:, 2:] = 50  # uniform size
        scores = rng.rand(n, 2).astype(np.float32)
        scores[:, 0] = np.linspace(0.6, 1.0, n)  # all above threshold
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.99, max_det=10)
        assert results[0].shape[0] <= 10

    def test_legacy_yolo_with_objectness(self) -> None:
        """Legacy YOLO: has objectness, [batch, anchors, 4 + 1 + num_classes]."""
        boxes = np.array([[320.0, 320.0, 100.0, 100.0]])
        obj = np.array([[0.95]])
        cls = np.array([[0.9, 0.1]])
        pred = np.concatenate([boxes, obj, cls], axis=1)[np.newaxis, ...].astype(np.float32)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5, has_objectness=True)
        assert results[0].shape[0] == 1

    def test_output_format(self) -> None:
        """Output should be [x1, y1, x2, y2, conf, class_id]."""
        boxes = np.array([[320.0, 320.0, 100.0, 100.0]])
        scores = np.array([[0.9, 0.1]])
        pred = self._make_prediction(boxes, scores)
        results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        det = results[0]
        assert det.shape[1] == 6  # x1,y1,x2,y2,conf,cls

    def test_invalid_conf_threshold(self) -> None:
        pred = np.zeros((1, 1, 6), dtype=np.float32)
        with pytest.raises(AssertionError):
            non_max_suppression(pred, conf_thres=1.5)

    def test_invalid_iou_threshold(self) -> None:
        pred = np.zeros((1, 1, 6), dtype=np.float32)
        with pytest.raises(AssertionError):
            non_max_suppression(pred, iou_thres=-0.1)
