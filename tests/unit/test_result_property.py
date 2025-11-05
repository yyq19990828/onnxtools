"""Unit tests for Result class read-only property protection.

This module tests that Result class properties are read-only and
attempting to assign values raises AttributeError as expected.

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import pytest
import numpy as np
from onnxtools import Result


class TestResultReadOnlyProperties:
    """Test that Result class properties are read-only (T017)."""

    def test_boxes_property_read_only(self):
        """Test that boxes property cannot be reassigned."""
        result = Result(boxes=np.array([[10, 20, 30, 40]], dtype=np.float32), orig_shape=(640, 640))

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.boxes = np.array([[50, 60, 70, 80]], dtype=np.float32)

    def test_scores_property_read_only(self):
        """Test that scores property cannot be reassigned."""
        result = Result(scores=np.array([0.9], dtype=np.float32), orig_shape=(640, 640))

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.scores = np.array([0.8], dtype=np.float32)

    def test_class_ids_property_read_only(self):
        """Test that class_ids property cannot be reassigned."""
        result = Result(class_ids=np.array([0], dtype=np.int32), orig_shape=(640, 640))

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.class_ids = np.array([1], dtype=np.int32)

    def test_orig_shape_property_read_only(self):
        """Test that orig_shape property cannot be reassigned."""
        result = Result(orig_shape=(640, 640))

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.orig_shape = (480, 480)

    def test_names_property_read_only(self):
        """Test that names property cannot be reassigned."""
        result = Result(orig_shape=(640, 640), names={0: 'vehicle'})

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.names = {1: 'plate'}

    def test_path_property_read_only(self):
        """Test that path property cannot be reassigned."""
        result = Result(orig_shape=(640, 640), path='/path/to/image.jpg')

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.path = '/another/path.jpg'

    def test_orig_img_property_read_only(self):
        """Test that orig_img property cannot be reassigned."""
        orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = Result(orig_img=orig_img, orig_shape=(640, 640))

        with pytest.raises(AttributeError, match="can't set attribute"):
            result.orig_img = np.ones((640, 640, 3), dtype=np.uint8)

    def test_internal_array_elements_can_be_modified(self):
        """Test that internal numpy array elements can be modified (shallow immutability)."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = Result(boxes=boxes, orig_shape=(640, 640))

        # Should be able to modify array elements
        result.boxes[0, 0] = 15.0
        assert result.boxes[0, 0] == 15.0
