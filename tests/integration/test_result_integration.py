"""Integration tests for Result class with BaseORT.

This module tests the integration of Result class with BaseORT subclasses,
verifying that detection models return Result objects correctly.

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import pytest
import numpy as np
from pathlib import Path
from onnxtools import Result


# Skip tests if models are not available
def check_model_exists(model_path: str) -> bool:
    """Check if model file exists."""
    return Path(model_path).exists()


#TODO maybe replace with pretrained models, not custom models
class TestResultBaseORTIntegration:
    """Test Result class integration with BaseORT subclasses (T022)."""

    @pytest.fixture
    def sample_test_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_yolo_returns_result_object(self, sample_test_image):
        """Test that YoloORT.__call__() returns Result object."""
        model_path = 'models/yolo11n.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        detector = create_detector('yolo', model_path, conf_thres=0.5, iou_thres=0.5)
        result = detector(sample_test_image)

        # Verify return type
        assert isinstance(result, Result), f"Expected Result object, got {type(result)}"

        # Verify Result attributes
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'class_ids')
        assert hasattr(result, 'orig_shape')

        # Verify orig_shape matches input
        assert result.orig_shape == sample_test_image.shape[:2]

        # Verify all arrays have consistent length
        assert len(result.boxes) == len(result.scores) == len(result.class_ids)

    def test_rtdetr_returns_result_object(self, sample_test_image):
        """Test that RtdetrORT.__call__() returns Result object."""
        model_path = 'models/rtdetr-2024080100.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        detector = create_detector('rtdetr', model_path, conf_thres=0.5)
        result = detector(sample_test_image)

        # Verify return type
        assert isinstance(result, Result), f"Expected Result object, got {type(result)}"

        # Verify Result attributes
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'class_ids')
        assert hasattr(result, 'orig_shape')

        # Verify orig_shape matches input
        assert result.orig_shape == sample_test_image.shape[:2]

        # Verify all arrays have consistent length
        assert len(result.boxes) == len(result.scores) == len(result.class_ids)

    def test_rfdetr_returns_result_object(self, sample_test_image):
        """Test that RfdetrORT.__call__() returns Result object."""
        model_path = 'models/rfdetr-20250919_medium.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        detector = create_detector('rfdetr', model_path, conf_thres=0.5)
        result = detector(sample_test_image)

        # Verify return type
        assert isinstance(result, Result), f"Expected Result object, got {type(result)}"

        # Verify Result attributes
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'class_ids')
        assert hasattr(result, 'orig_shape')

        # Verify orig_shape matches input
        assert result.orig_shape == sample_test_image.shape[:2]

        # Verify all arrays have consistent length
        assert len(result.boxes) == len(result.scores) == len(result.class_ids)

    def test_result_indexing_with_real_detections(self, sample_test_image):
        """Test Result indexing works with real detection outputs."""
        model_path = 'models/yolo11n.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        detector = create_detector('yolo', model_path, conf_thres=0.5, iou_thres=0.5)
        result = detector(sample_test_image)

        if len(result) > 0:
            # Test integer indexing
            first = result[0]
            assert isinstance(first, Result)
            assert len(first) == 1

            # Test slicing
            if len(result) > 1:
                subset = result[0:2]
                assert isinstance(subset, Result)
                assert len(subset) <= 2

    def test_result_properties_with_real_detections(self, sample_test_image):
        """Test Result properties work correctly with real detections."""
        model_path = 'models/yolo11n.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        detector = create_detector('yolo', model_path, conf_thres=0.5, iou_thres=0.5)
        result = detector(sample_test_image)

        # Test all properties are accessible
        _ = result.boxes
        _ = result.scores
        _ = result.class_ids
        _ = result.orig_shape
        _ = result.names
        _ = result.path

        # Test __len__
        length = len(result)
        assert length >= 0

        # Test __str__ and __repr__
        str_repr = str(result)
        repr_repr = repr(result)
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)

    def test_empty_result_from_baseort(self):
        """Test that BaseORT returns valid Result even with no detections."""
        model_path = 'models/yolo11n.onnx'

        if not check_model_exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from onnxtools import create_detector

        # Use very high confidence threshold to force no detections
        detector = create_detector('yolo', model_path, conf_thres=0.99, iou_thres=0.5)

        # Create a blank image (likely no detections)
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector(blank_image)

        # Should still be a valid Result object
        assert isinstance(result, Result)
        assert len(result) >= 0  # Could be 0 or more
        assert result.boxes.shape[1] == 4  # Should have (N, 4) shape
        assert result.orig_shape == blank_image.shape[:2]
