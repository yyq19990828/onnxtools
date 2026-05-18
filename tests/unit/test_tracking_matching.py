"""Unit tests for matching primitives."""

from __future__ import annotations

import numpy as np

from onnxtools.tracking.matching import box_giou_batch, box_iou_batch, fuse_score, iou_distance, linear_assignment


class TestBoxIou:
    def test_self_iou_is_one(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 100, 100]], dtype=np.float32)
        iou = box_iou_batch(boxes, boxes)
        assert iou.shape == (2, 2)
        np.testing.assert_allclose(np.diag(iou), [1.0, 1.0], atol=1e-6)

    def test_disjoint_is_zero(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[100, 100, 110, 110]], dtype=np.float32)
        assert box_iou_batch(a, b)[0, 0] == 0.0

    def test_known_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        # intersect = 5*5=25; union = 100+100-25 = 175.
        np.testing.assert_allclose(box_iou_batch(a, b), [[25 / 175]], atol=1e-6)

    def test_empty_inputs(self):
        a = np.empty((0, 4), dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert box_iou_batch(a, b).shape == (0, 1)
        assert box_iou_batch(b, a).shape == (1, 0)

    def test_giou_bounded_by_iou(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, size=(5, 2))
        a = np.concatenate([a, a + rng.uniform(5, 30, size=(5, 2))], axis=1).astype(np.float32)
        b = rng.uniform(0, 100, size=(5, 2))
        b = np.concatenate([b, b + rng.uniform(5, 30, size=(5, 2))], axis=1).astype(np.float32)
        iou = box_iou_batch(a, b)
        giou = box_giou_batch(a, b)
        assert np.all(giou <= iou + 1e-6)
        assert np.all(giou >= -1.0 - 1e-6)


class TestLinearAssignment:
    def test_perfect_match(self):
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert sorted(map(tuple, matches.tolist())) == [(0, 0), (1, 1)]
        assert ua.size == 0 and ub.size == 0

    def test_threshold_blocks_high_cost(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert sorted(map(tuple, matches.tolist())) == [(0, 0), (1, 1)]

    def test_all_above_threshold_no_match(self):
        cost = np.array([[0.9, 0.95], [0.99, 0.92]])
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.shape == (0, 2)
        assert set(ua.tolist()) == {0, 1}
        assert set(ub.tolist()) == {0, 1}

    def test_rectangular(self):
        cost = np.array([[0.0, 0.9, 0.9]])  # one row, three cols
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.tolist() == [[0, 0]]
        assert ua.size == 0
        assert set(ub.tolist()) == {1, 2}

    def test_empty(self):
        cost = np.empty((0, 0), dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.shape == (0, 2)
        assert ua.size == 0 and ub.size == 0


class TestFuseScore:
    def test_high_score_reduces_cost(self):
        cost = np.array([[0.3, 0.3]], dtype=np.float32)
        scores = np.array([0.9, 0.5], dtype=np.float32)
        fused = fuse_score(cost, scores)
        # Higher score detection should have *lower* fused cost.
        assert fused[0, 0] < fused[0, 1]

    def test_empty(self):
        out = fuse_score(np.empty((0, 0), dtype=np.float32), np.empty((0,)))
        assert out.shape == (0, 0)


def test_iou_distance_equals_one_minus_iou():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 5, 15, 15]], dtype=np.float32)
    np.testing.assert_allclose(iou_distance(a, b), 1.0 - box_iou_batch(a, b), atol=1e-6)
