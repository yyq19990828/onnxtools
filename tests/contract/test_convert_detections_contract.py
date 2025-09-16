"""Contract tests for convert_to_supervision_detections API - 这些测试必须在实现前编写且必须失败."""

import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.contract
class TestConvertDetectionsContract:
    """Contract tests for detection format conversion to supervision.Detections."""

    def test_convert_function_exists(self):
        """Contract: convert_to_supervision_detections function must exist."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections
            assert callable(convert_to_supervision_detections), "Function must be callable"
        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_function_signature(self):
        """Contract: Function must accept correct parameters."""
        try:
            import inspect
            from utils.supervision_converter import convert_to_supervision_detections

            sig = inspect.signature(convert_to_supervision_detections)
            params = list(sig.parameters.keys())

            expected_params = ['detections_array', 'class_names']
            assert params == expected_params, f"Expected {expected_params}, got {params}"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_empty_detections(self, sample_class_names):
        """Contract: Must handle empty detection arrays gracefully."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections
            import supervision as sv

            # Test empty detection list
            empty_detections = [[]]
            result = convert_to_supervision_detections(empty_detections, sample_class_names)

            assert isinstance(result, sv.Detections), "Must return supervision.Detections"
            assert len(result.xyxy) == 0, "Empty input should produce empty detections"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_valid_detections_format(self, sample_detections, sample_class_names):
        """Contract: Must convert current detection format to supervision format correctly."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections
            import supervision as sv

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check return type
            assert isinstance(result, sv.Detections), "Must return supervision.Detections object"

            # Check required attributes exist
            assert hasattr(result, 'xyxy'), "Must have xyxy attribute"
            assert hasattr(result, 'confidence'), "Must have confidence attribute"
            assert hasattr(result, 'class_id'), "Must have class_id attribute"
            assert hasattr(result, 'data'), "Must have data attribute"

            # Check array shapes
            num_detections = len(sample_detections[0])
            assert result.xyxy.shape == (num_detections, 4), f"xyxy shape mismatch: {result.xyxy.shape}"
            assert result.confidence.shape == (num_detections,), f"confidence shape mismatch: {result.confidence.shape}"
            assert result.class_id.shape == (num_detections,), f"class_id shape mismatch: {result.class_id.shape}"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_xyxy_coordinates(self, sample_detections, sample_class_names):
        """Contract: Bounding box coordinates must be correctly extracted."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check first detection coordinates
            expected_box = sample_detections[0][0][:4]  # [x1, y1, x2, y2]
            actual_box = result.xyxy[0]

            np.testing.assert_array_equal(actual_box, expected_box,
                                        "Bounding box coordinates must match input")

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_confidence_scores(self, sample_detections, sample_class_names):
        """Contract: Confidence scores must be correctly extracted."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check confidence scores
            expected_conf = [det[4] for det in sample_detections[0]]
            actual_conf = result.confidence

            np.testing.assert_array_almost_equal(actual_conf, expected_conf, decimal=6,
                                                err_msg="Confidence scores must match input")

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_class_ids(self, sample_detections, sample_class_names):
        """Contract: Class IDs must be correctly extracted and converted."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check class IDs
            expected_class_ids = [int(det[5]) for det in sample_detections[0]]
            actual_class_ids = result.class_id

            np.testing.assert_array_equal(actual_class_ids, expected_class_ids,
                                        "Class IDs must match input")

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_class_names_metadata(self, sample_detections, sample_class_names):
        """Contract: Class names must be included in data metadata."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check class names in data
            assert 'class_name' in result.data, "class_name must be in data metadata"

            expected_names = [sample_class_names[int(det[5])] for det in sample_detections[0]]
            actual_names = result.data['class_name']

            assert actual_names == expected_names, "Class names must match expected mapping"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_data_types(self, sample_detections, sample_class_names):
        """Contract: Output arrays must have correct data types."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            result = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Check data types
            assert result.xyxy.dtype == np.float32 or result.xyxy.dtype == np.float64, \
                f"xyxy must be float type, got {result.xyxy.dtype}"
            assert result.confidence.dtype == np.float32 or result.confidence.dtype == np.float64, \
                f"confidence must be float type, got {result.confidence.dtype}"
            assert result.class_id.dtype == np.int32 or result.class_id.dtype == np.int64, \
                f"class_id must be int type, got {result.class_id.dtype}"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_multiple_batches(self, sample_class_names):
        """Contract: Must handle multiple detection batches correctly."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            # Create multi-batch detections (should take first batch)
            multi_batch_detections = [
                [[100.0, 100.0, 200.0, 150.0, 0.95, 0]],  # First batch
                [[150.0, 150.0, 250.0, 200.0, 0.88, 1]]   # Second batch (should be ignored)
            ]

            result = convert_to_supervision_detections(multi_batch_detections, sample_class_names)

            # Should only process first batch
            assert len(result.xyxy) == 1, "Should only process first batch"
            np.testing.assert_array_equal(result.xyxy[0], [100.0, 100.0, 200.0, 150.0])

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_invalid_class_id_handling(self, sample_class_names):
        """Contract: Must handle invalid class IDs gracefully."""
        try:
            from utils.supervision_converter import convert_to_supervision_detections

            # Create detection with invalid class ID
            invalid_detections = [
                [[100.0, 100.0, 200.0, 150.0, 0.95, 999]]  # Invalid class ID
            ]

            # Should not crash, but may use default handling
            result = convert_to_supervision_detections(invalid_detections, sample_class_names)
            assert isinstance(result, type(result)), "Should handle invalid class IDs gracefully"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")

    def test_convert_performance_requirement(self, sample_class_names):
        """Contract: Conversion must be fast enough for real-time processing."""
        try:
            import time
            from utils.supervision_converter import convert_to_supervision_detections

            # Create larger detection set
            large_detections = []
            detection_list = []
            for i in range(50):  # 50 detections
                x1, y1 = i * 10, i * 10
                x2, y2 = x1 + 50, y1 + 30
                detection_list.append([x1, y1, x2, y2, 0.9, i % 2])
            large_detections.append(detection_list)

            # Measure conversion time
            start_time = time.time()
            for _ in range(100):  # 100 iterations
                result = convert_to_supervision_detections(large_detections, sample_class_names)
            avg_time = (time.time() - start_time) / 100 * 1000  # ms

            assert avg_time < 5.0, f"Conversion too slow: {avg_time:.2f}ms (target: <5ms)"

        except ImportError:
            pytest.fail("convert_to_supervision_detections function must be implemented")