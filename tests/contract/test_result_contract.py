"""Contract tests for Result class API.

This module implements contract tests based on the API specification
in contracts/result_api.yaml, verifying that the Result class adheres
to its documented behavior and error handling contracts.

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import numpy as np
import pytest

from onnxtools import Result


class TestResultInitializationContract:
    """Test Result initialization contract based on result_api.yaml (T010)."""

    def test_valid_init_with_all_params(self):
        """Contract: valid_init_with_all_params from result_api.yaml."""
        # Input from contract
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        orig_shape = (640, 640)
        names = {0: "vehicle"}

        # Expected: status 200, properties accessible
        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_shape=orig_shape,
            names=names
        )

        # Verify properties are accessible
        assert result.boxes is not None
        assert result.scores is not None
        assert result.class_ids is not None
        assert result.orig_shape == orig_shape
        assert result.names == names

    def test_valid_init_with_none_boxes(self):
        """Contract: valid_init_with_none_boxes from result_api.yaml."""
        # Input from contract
        boxes = None
        scores = None
        class_ids = None
        orig_shape = (640, 640)

        # Expected: status 200, boxes_shape (0, 4), scores_shape (0,)
        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_shape=orig_shape
        )

        # Verify empty array shapes
        assert result.boxes.shape == (0, 4)
        assert result.scores.shape == (0,)
        assert result.class_ids.shape == (0,)

    def test_invalid_init_missing_orig_shape(self):
        """Contract: invalid_init_missing_orig_shape from result_api.yaml."""
        # Input from contract
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)

        # Expected: status 400, error_type TypeError
        with pytest.raises(TypeError, match="orig_shape is required"):
            Result(boxes=boxes, orig_shape=None)

    def test_invalid_init_wrong_boxes_shape(self):
        """Contract: Boxes with wrong shape should return status 400, ValueError."""
        boxes = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)  # (N, 5) instead of (N, 4)
        orig_shape = (640, 640)

        with pytest.raises(ValueError, match="boxes must have shape \\(N, 4\\)"):
            Result(boxes=boxes, orig_shape=orig_shape)

    def test_invalid_init_length_mismatch(self):
        """Contract: Length mismatch should return status 400, ValueError."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)  # Length 1, boxes length 2
        class_ids = np.array([0, 1], dtype=np.int32)
        orig_shape = (640, 640)

        with pytest.raises(ValueError, match="boxes, scores, and class_ids must have the same length"):
            Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=orig_shape)


class TestResultIndexingContract:
    """Test Result indexing contract based on result_api.yaml (T023)."""

    def test_integer_indexing_returns_result(self):
        """Contract: Integer indexing must return Result object with single detection."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Contract: result[0] must return Result with len() == 1
        first = result[0]
        assert isinstance(first, Result)
        assert len(first) == 1
        assert first.boxes.shape == (1, 4)
        assert first.scores.shape == (1,)
        assert first.class_ids.shape == (1,)

    def test_negative_indexing_returns_result(self):
        """Contract: Negative indexing must return Result object."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Contract: result[-1] must return Result with last detection
        last = result[-1]
        assert isinstance(last, Result)
        assert len(last) == 1
        np.testing.assert_array_equal(last.boxes[0], boxes[-1])

    def test_slicing_returns_result(self):
        """Contract: Slicing must return Result object with subset."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = Result(boxes=boxes, scores=scores, orig_shape=(640, 640))

        # Contract: result[1:3] must return Result with 2 detections
        subset = result[1:3]
        assert isinstance(subset, Result)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.boxes, boxes[1:3])
        np.testing.assert_array_equal(subset.scores, scores[1:3])

    def test_indexing_out_of_bounds_raises_error(self):
        """Contract: Out of bounds indexing must raise IndexError."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Contract: result[10] must raise IndexError
        with pytest.raises(IndexError):
            _ = result[10]

        # Contract: result[-10] must raise IndexError
        with pytest.raises(IndexError):
            _ = result[-10]

    def test_indexing_preserves_metadata(self):
        """Contract: Indexing must preserve orig_shape, names, path."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        names = {0: 'vehicle', 1: 'plate'}
        result = Result(boxes=boxes, orig_shape=(640, 480), names=names, path='/test.jpg')

        # Contract: Indexed result must share metadata
        first = result[0]
        assert first.orig_shape == (640, 480)
        assert first.names == names
        assert first.path == '/test.jpg'

    def test_empty_slice_returns_empty_result(self):
        """Contract: Empty slice must return valid empty Result."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Contract: result[5:10] on single-element result must return empty Result
        empty = result[5:10]
        assert isinstance(empty, Result)
        assert len(empty) == 0
        assert empty.boxes.shape == (0, 4)


class TestResultFilteringContract:
    """Test Result filtering contract based on result_api.yaml (T047)."""

    def test_filter_by_confidence_contract(self):
        """Contract: filter() with conf_threshold must return filtered Result."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.6], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Contract: filter with conf_threshold=0.7 returns Result with length 1
        filtered = result.filter(conf_threshold=0.7)
        assert isinstance(filtered, Result)
        assert len(filtered) == 1

    def test_filter_by_class_contract(self):
        """Contract: filter() with classes must return Result with specified classes."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Contract: filter with classes=[0] returns only class 0
        filtered = result.filter(classes=[0])
        assert isinstance(filtered, Result)
        assert all(filtered.class_ids == 0)

    def test_filter_preserves_immutability_contract(self):
        """Contract: filter() must return new Result, not modify original."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.6], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        original_len = len(result)
        # Contract: filter must not modify original
        filtered = result.filter(conf_threshold=0.8)

        assert len(result) == original_len  # Original unchanged
        assert len(filtered) != len(result)  # New Result is different
        assert filtered is not result  # Different objects

    def test_summary_contract(self):
        """Contract: summary() must return dict with required keys."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        stats = result.summary()

        # Contract: must return dict with specific keys
        required_keys = {'total_detections', 'class_counts', 'avg_confidence',
                        'min_confidence', 'max_confidence'}
        assert set(stats.keys()) == required_keys

        # Contract: types must be correct
        assert isinstance(stats['total_detections'], int)
        assert isinstance(stats['class_counts'], dict)
        assert isinstance(stats['avg_confidence'], float)
        assert isinstance(stats['min_confidence'], float)
        assert isinstance(stats['max_confidence'], float)


class TestResultVisualizationContract:
    """Test Result visualization contract based on result_api.yaml (T034)."""

    def test_to_supervision_contract(self):
        """Contract: to_supervision() must return supervision.Detections."""
        import supervision as sv

        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640), names={0: 'test'})

        sv_detections = result.to_supervision()
        assert isinstance(sv_detections, sv.Detections)

    def test_plot_contract_returns_ndarray(self):
        """Contract: plot() must return numpy.ndarray with dtype=uint8."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        orig_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids,
                       orig_img=orig_img, orig_shape=(640, 640), names={0: 'test'})

        annotated = result.plot()
        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8
        assert annotated.shape == orig_img.shape

    def test_plot_contract_validates_orig_img(self):
        """Contract: plot() must raise ValueError if orig_img is None."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        with pytest.raises(ValueError, match="Cannot plot detections: orig_img is None"):
            result.plot()

    def test_save_contract_creates_file(self):
        """Contract: save() must create a file at specified path."""
        import os
        import tempfile

        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        orig_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids,
                       orig_img=orig_img, orig_shape=(640, 640), names={0: 'test'})

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result.save(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
