"""Unit tests for Result class.

This module contains unit tests for the Result class, covering:
- Initialization and validation
- Property access
- Length operations
- Indexing and slicing
- Empty result handling
- Filtering
- Data conversion
- Visualization methods

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import numpy as np
import pytest

from onnxtools import Result


class TestResultInitialization:
    """Test Result class initialization and validation (T009)."""

    def test_valid_initialization_with_all_params(self):
        """Test successful initialization with all parameters."""
        boxes = np.array([[10, 20, 100, 150], [200, 300, 400, 500]], dtype=np.float32)
        scores = np.array([0.95, 0.87], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
        orig_shape = (640, 640)
        names = {0: "vehicle", 1: "plate"}
        path = "/path/to/image.jpg"

        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=orig_shape,
            names=names,
            path=path,
        )

        assert result is not None
        assert len(result) == 2
        np.testing.assert_array_equal(result.boxes, boxes)
        np.testing.assert_array_equal(result.scores, scores)
        np.testing.assert_array_equal(result.class_ids, class_ids)
        assert result.orig_shape == orig_shape
        assert result.names == names
        assert result.path == path

    def test_valid_initialization_with_none_boxes(self):
        """Test initialization with None boxes (empty result)."""
        result = Result(boxes=None, scores=None, class_ids=None, orig_shape=(640, 640))

        assert result is not None
        assert len(result) == 0
        assert result.boxes.shape == (0, 4)
        assert result.scores.shape == (0,)
        assert result.class_ids.shape == (0,)
        assert result.orig_img is None
        assert result.names == {}
        assert result.path is None

    def test_v1_orig_shape_required(self):
        """Test V1: orig_shape cannot be None."""
        with pytest.raises(TypeError, match="orig_shape is required and cannot be None"):
            Result(boxes=np.array([[10, 20, 30, 40]], dtype=np.float32), orig_shape=None)

    def test_v2_orig_shape_must_be_tuple(self):
        """Test V2: orig_shape must be a tuple of length 2."""
        with pytest.raises(ValueError, match="orig_shape must be a tuple of"):
            Result(
                boxes=np.array([[10, 20, 30, 40]], dtype=np.float32),
                orig_shape=[640, 640],  # List instead of tuple
            )

        with pytest.raises(ValueError, match="orig_shape must be a tuple of"):
            Result(
                boxes=np.array([[10, 20, 30, 40]], dtype=np.float32),
                orig_shape=(640,),  # Wrong length
            )

    def test_v3_boxes_shape_validation(self):
        """Test V3: boxes must have shape (N, 4)."""
        with pytest.raises(ValueError, match="boxes must have shape \\(N, 4\\)"):
            Result(
                boxes=np.array([[10, 20, 30]], dtype=np.float32),  # Wrong shape (N, 3)
                orig_shape=(640, 640),
            )

        with pytest.raises(ValueError, match="boxes must have shape \\(N, 4\\)"):
            Result(
                boxes=np.array([10, 20, 30, 40], dtype=np.float32),  # Wrong shape (4,)
                orig_shape=(640, 640),
            )

    def test_v4_scores_shape_validation(self):
        """Test V4: scores must have shape (N,)."""
        with pytest.raises(ValueError, match="scores must have shape \\(N,\\)"):
            Result(
                scores=np.array([[0.9], [0.8]], dtype=np.float32),  # Wrong shape (N, 1)
                orig_shape=(640, 640),
            )

    def test_v5_class_ids_shape_validation(self):
        """Test V5: class_ids must have shape (N,)."""
        with pytest.raises(ValueError, match="class_ids must have shape \\(N,\\)"):
            Result(
                class_ids=np.array([[0], [1]], dtype=np.int32),  # Wrong shape (N, 1)
                orig_shape=(640, 640),
            )

    def test_v6_length_consistency(self):
        """Test V6: boxes, scores, and class_ids must have same length."""
        with pytest.raises(ValueError, match="boxes, scores, and class_ids must have the same length"):
            Result(
                boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32),
                scores=np.array([0.9], dtype=np.float32),  # Length mismatch
                class_ids=np.array([0, 1], dtype=np.int32),
                orig_shape=(640, 640),
            )

        with pytest.raises(ValueError, match="boxes, scores, and class_ids must have the same length"):
            Result(
                boxes=np.array([[10, 20, 30, 40]], dtype=np.float32),
                scores=np.array([0.9, 0.8], dtype=np.float32),  # Length mismatch
                class_ids=np.array([0], dtype=np.int32),
                orig_shape=(640, 640),
            )


class TestResultProperties:
    """Test Result class property access (T016)."""

    def test_boxes_property_access(self):
        """Test boxes property returns correct array."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        np.testing.assert_array_equal(result.boxes, boxes)
        assert result.boxes.shape == (2, 4)

    def test_scores_property_access(self):
        """Test scores property returns correct array."""
        scores = np.array([0.95, 0.87], dtype=np.float32)
        result = Result(
            boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32), scores=scores, orig_shape=(640, 640)
        )

        np.testing.assert_array_equal(result.scores, scores)
        assert result.scores.shape == (2,)

    def test_class_ids_property_access(self):
        """Test class_ids property returns correct array."""
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(
            boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32),
            class_ids=class_ids,
            orig_shape=(640, 640),
        )

        np.testing.assert_array_equal(result.class_ids, class_ids)
        assert result.class_ids.shape == (2,)

    def test_orig_shape_property_access(self):
        """Test orig_shape property returns correct tuple."""
        orig_shape = (640, 480)
        result = Result(orig_shape=orig_shape)

        assert result.orig_shape == orig_shape
        assert isinstance(result.orig_shape, tuple)

    def test_names_property_access(self):
        """Test names property returns correct dict."""
        names = {0: "vehicle", 1: "plate"}
        result = Result(orig_shape=(640, 640), names=names)

        assert result.names == names
        assert result.names[0] == "vehicle"

    def test_path_property_access(self):
        """Test path property returns correct string."""
        path = "/path/to/image.jpg"
        result = Result(orig_shape=(640, 640), path=path)

        assert result.path == path

    def test_orig_img_property_access(self):
        """Test orig_img property returns correct array."""
        orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = Result(orig_img=orig_img, orig_shape=(640, 640))

        assert result.orig_img is not None
        np.testing.assert_array_equal(result.orig_img, orig_img)
        assert result.orig_img.shape == (640, 640, 3)


class TestResultLength:
    """Test Result class __len__ method (T018)."""

    def test_len_with_detections(self):
        """Test __len__ returns correct count with detections."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        assert len(result) == 3

    def test_len_empty_result(self):
        """Test __len__ returns 0 for empty result."""
        result = Result(boxes=None, orig_shape=(640, 640))

        assert len(result) == 0

    def test_len_single_detection(self):
        """Test __len__ returns 1 for single detection."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        assert len(result) == 1


class TestResultIndexing:
    """Test Result class __getitem__ method (T019, T020)."""

    def test_integer_index_positive(self):
        """Test integer indexing with positive index."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        first = result[0]
        assert len(first) == 1
        np.testing.assert_array_equal(first.boxes, np.array([[10, 20, 30, 40]], dtype=np.float32))
        np.testing.assert_array_equal(first.scores, np.array([0.9], dtype=np.float32))
        np.testing.assert_array_equal(first.class_ids, np.array([0], dtype=np.int32))

    def test_integer_index_negative(self):
        """Test integer indexing with negative index."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = Result(boxes=boxes, scores=scores, orig_shape=(640, 640))

        last = result[-1]
        assert len(last) == 1
        np.testing.assert_array_equal(last.boxes, np.array([[90, 100, 110, 120]], dtype=np.float32))
        np.testing.assert_array_equal(last.scores, np.array([0.7], dtype=np.float32))

    def test_integer_index_out_of_bounds(self):
        """Test integer indexing raises IndexError when out of bounds."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        with pytest.raises(IndexError):
            _ = result[10]

    def test_slice_range(self):
        """Test slicing with range [1:3]."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = Result(boxes=boxes, scores=scores, orig_shape=(640, 640))

        subset = result[1:3]
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.boxes, np.array([[50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32))
        np.testing.assert_array_equal(subset.scores, np.array([0.8, 0.7], dtype=np.float32))

    def test_slice_from_start(self):
        """Test slicing from start [:2]."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        subset = result[:2]
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.boxes[0], [10, 20, 30, 40])

    def test_slice_to_end(self):
        """Test slicing to end [1:]."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        subset = result[1:]
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.boxes[0], [50, 60, 70, 80])

    def test_indexing_preserves_shared_attributes(self):
        """Test that indexing preserves shared attributes."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
        names = {0: "vehicle", 1: "plate"}
        result = Result(boxes=boxes, orig_img=orig_img, orig_shape=(640, 640), names=names, path="/test.jpg")

        subset = result[0]
        assert subset.orig_shape == (640, 640)
        assert subset.names == names
        assert subset.path == "/test.jpg"
        assert subset.orig_img is orig_img  # Shared reference


class TestResultEmptyCases:
    """Test Result class with empty detection results (T021)."""

    def test_empty_initialization_with_none(self):
        """Test initialization with None for boxes/scores/class_ids."""
        result = Result(boxes=None, scores=None, class_ids=None, orig_shape=(640, 640))

        assert len(result) == 0
        assert result.boxes.shape == (0, 4)
        assert result.scores.shape == (0,)
        assert result.class_ids.shape == (0,)

    def test_empty_result_indexing_raises_error(self):
        """Test that indexing empty result raises IndexError."""
        result = Result(boxes=None, orig_shape=(640, 640))

        with pytest.raises(IndexError):
            _ = result[0]

    def test_empty_result_slicing_returns_empty(self):
        """Test that slicing empty result returns empty result."""
        result = Result(boxes=None, orig_shape=(640, 640))

        subset = result[0:10]
        assert len(subset) == 0

    def test_empty_result_properties_return_empty_arrays(self):
        """Test that properties return correct empty arrays."""
        result = Result(orig_shape=(640, 640))

        assert result.boxes.dtype == np.float32
        assert result.scores.dtype == np.float32
        assert result.class_ids.dtype == np.int32


class TestResultFiltering:
    """Test Result class filter() method (T041-T046)."""

    def test_filter_by_confidence_threshold(self):
        """Test filter() with confidence threshold only (T041)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.6, 0.3], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter with threshold 0.7
        filtered = result.filter(conf_threshold=0.7)

        assert len(filtered) == 1
        assert filtered.scores[0] == 0.9
        np.testing.assert_array_equal(filtered.boxes[0], boxes[0])

    def test_filter_by_confidence_boundary_conditions(self):
        """Test filter() with boundary confidence values (T041)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([1.0, 0.0], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Threshold 1.0 should include 1.0
        filtered = result.filter(conf_threshold=1.0)
        assert len(filtered) == 1
        assert filtered.scores[0] == 1.0

        # Threshold 0.0 should include all
        filtered = result.filter(conf_threshold=0.0)
        assert len(filtered) == 2

    def test_filter_by_single_class(self):
        """Test filter() with single class ID (T042)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter for class 0
        filtered = result.filter(classes=[0])
        assert len(filtered) == 2
        assert all(filtered.class_ids == 0)

        # Filter for class 1
        filtered = result.filter(classes=[1])
        assert len(filtered) == 1
        assert filtered.class_ids[0] == 1

    def test_filter_by_multiple_classes(self):
        """Test filter() with multiple class IDs (T042)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        class_ids = np.array([0, 1, 2], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter for classes [0, 2]
        filtered = result.filter(classes=[0, 2])
        assert len(filtered) == 2
        assert 1 not in filtered.class_ids

    def test_filter_by_non_existent_class(self):
        """Test filter() with non-existent class ID (T042)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter for non-existent class 5
        filtered = result.filter(classes=[5])
        assert len(filtered) == 0

    def test_filter_combined_confidence_and_class(self):
        """Test filter() with both confidence and class filtering (T043)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.6, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter for class 0 with conf > 0.7
        filtered = result.filter(conf_threshold=0.7, classes=[0])
        assert len(filtered) == 2
        assert all(filtered.class_ids == 0)
        assert all(filtered.scores >= 0.7)

    def test_filter_invalid_confidence_threshold(self):
        """Test filter() raises ValueError for invalid conf_threshold (T044)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Test out of range
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            result.filter(conf_threshold=1.5)

        with pytest.raises(ValueError, match="conf_threshold must be in"):
            result.filter(conf_threshold=-0.1)

        # Test wrong type
        with pytest.raises(ValueError, match="conf_threshold must be a number"):
            result.filter(conf_threshold="0.5")

    def test_filter_invalid_classes_type(self):
        """Test filter() raises ValueError for invalid classes type (T044)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Test wrong type
        with pytest.raises(ValueError, match="classes must be a list or tuple"):
            result.filter(classes=0)

        # Test non-integer values
        with pytest.raises(ValueError, match="classes must contain int or str values"):
            result.filter(classes=[0.5, 1.5])

    def test_filter_returns_empty_result(self):
        """Test filter() returns empty Result when no matches (T045)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.5, 0.6], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640))

        # Filter with impossible threshold
        filtered = result.filter(conf_threshold=0.99)
        assert isinstance(filtered, Result)
        assert len(filtered) == 0
        assert filtered.boxes.shape == (0, 4)

    def test_filter_preserves_metadata(self):
        """Test filter() preserves orig_shape, names, path (T045)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        names = {0: "vehicle", 1: "plate"}
        result = Result(
            boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 480), names=names, path="/test.jpg"
        )

        # Filter and verify metadata
        filtered = result.filter(conf_threshold=0.85)
        assert filtered.orig_shape == (640, 480)
        assert filtered.names == names
        assert filtered.path == "/test.jpg"

    def test_summary_with_detections(self):
        """Test summary() returns correct statistics (T046)."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.7, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)
        names = {0: "vehicle", 1: "plate"}
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640), names=names)

        stats = result.summary()

        assert stats["total_detections"] == 3
        assert stats["class_counts"] == {"vehicle": 2, "plate": 1}
        assert stats["avg_confidence"] == pytest.approx(0.8, abs=0.01)
        assert stats["min_confidence"] == pytest.approx(0.7, abs=0.01)
        assert stats["max_confidence"] == pytest.approx(0.9, abs=0.01)

    def test_summary_empty_result(self):
        """Test summary() returns zero statistics for empty result (T046)."""
        result = Result(boxes=None, orig_shape=(640, 640))

        stats = result.summary()

        assert stats["total_detections"] == 0
        assert stats["class_counts"] == {}
        assert stats["avg_confidence"] == 0.0
        assert stats["min_confidence"] == 0.0
        assert stats["max_confidence"] == 0.0


class TestResultConversion:
    """Test Result class conversion methods (T024)."""

    def test_numpy_method_returns_self(self):
        """Test numpy() method returns self (idempotent)."""
        result = Result(orig_shape=(640, 640))

        result_np = result.numpy()
        assert result_np is result


class TestResultVisualization:
    """Test Result class visualization methods (T030, T031, T032)."""

    def test_to_supervision_returns_detections_object(self):
        """Test that to_supervision() returns supervision.Detections object (T030)."""
        import supervision as sv

        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        names = {0: "vehicle", 1: "plate"}
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640), names=names)

        sv_detections = result.to_supervision()

        # Verify return type
        assert isinstance(sv_detections, sv.Detections)

        # Verify data consistency
        np.testing.assert_array_equal(sv_detections.xyxy, boxes)
        np.testing.assert_array_equal(sv_detections.confidence, scores)
        np.testing.assert_array_equal(sv_detections.class_id, class_ids)

        # Verify class names
        assert "class_name" in sv_detections.data
        assert sv_detections.data["class_name"] == ["vehicle", "plate"]

    def test_to_supervision_empty_result(self):
        """Test that to_supervision() returns empty Detections for empty result (T030)."""
        import supervision as sv

        result = Result(boxes=None, orig_shape=(640, 640))

        sv_detections = result.to_supervision()

        assert isinstance(sv_detections, sv.Detections)
        assert len(sv_detections) == 0

    def test_to_supervision_handles_missing_class_names(self):
        """Test that to_supervision() handles missing class names gracefully (T030)."""

        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([5], dtype=np.int32)  # Class ID not in names
        names = {0: "vehicle", 1: "plate"}
        result = Result(boxes=boxes, scores=scores, class_ids=class_ids, orig_shape=(640, 640), names=names)

        sv_detections = result.to_supervision()

        # Should use fallback name
        assert sv_detections.data["class_name"] == ["class_5"]

    def test_plot_with_default_preset(self):
        """Test plot() method with default annotator_preset (T031)."""
        boxes = np.array([[10, 20, 100, 150]], dtype=np.float32)
        scores = np.array([0.95], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        orig_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        names = {0: "vehicle"}
        result = Result(
            boxes=boxes, scores=scores, class_ids=class_ids, orig_img=orig_img, orig_shape=(640, 640), names=names
        )

        # plot() should return annotated image
        annotated = result.plot()

        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8
        assert annotated.shape == orig_img.shape
        # Annotated image should be different from original
        assert not np.array_equal(annotated, orig_img)

    def test_plot_with_custom_preset(self):
        """Test plot() method with custom annotator_preset (T031)."""
        boxes = np.array([[10, 20, 100, 150]], dtype=np.float32)
        scores = np.array([0.95], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        orig_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = Result(
            boxes=boxes, scores=scores, class_ids=class_ids, orig_img=orig_img, orig_shape=(640, 640), names={0: "test"}
        )

        # Test with different presets
        for preset in ["debug", "lightweight"]:
            annotated = result.plot(annotator_preset=preset)
            assert isinstance(annotated, np.ndarray)
            assert annotated.shape == orig_img.shape

    def test_plot_empty_result(self):
        """Test plot() with empty result returns original image (T031)."""
        orig_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = Result(boxes=None, orig_img=orig_img, orig_shape=(640, 640))

        annotated = result.plot()

        # Empty result should return copy of original image
        assert isinstance(annotated, np.ndarray)
        np.testing.assert_array_equal(annotated, orig_img)

    def test_plot_raises_error_when_orig_img_none(self):
        """Test plot() raises ValueError when orig_img is None (T032)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        with pytest.raises(ValueError, match="Cannot plot detections: orig_img is None"):
            result.plot()

    def test_show_raises_error_when_orig_img_none(self):
        """Test show() raises ValueError when orig_img is None (T032)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        with pytest.raises(ValueError, match="Cannot plot detections: orig_img is None"):
            result.show()

    def test_save_raises_error_when_orig_img_none(self):
        """Test save() raises ValueError when orig_img is None (T032)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        with pytest.raises(ValueError, match="Cannot plot detections: orig_img is None"):
            result.save("/tmp/test.jpg")
