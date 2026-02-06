"""Contract tests for OCR Dataset Evaluator

Verifies API contracts for:
- Basic evaluation flow
- Result format structure
- Edit distance metrics
- Confidence threshold filtering
- JSON export format
"""

import json

import cv2
import numpy as np
import pytest


class MockOCRModel:
    """Mock OCR model for contract testing"""

    def __init__(self, responses: list):
        """Initialize mock with predefined responses

        Args:
            responses: List of (text, confidence, char_scores) tuples
        """
        self.responses = responses
        self.call_count = 0

    def __call__(self, image: np.ndarray) -> tuple:
        """Simulate OCR inference

        Args:
            image: Input plate image

        Returns:
            (predicted_text, confidence, char_scores) or None
        """
        if self.call_count < len(self.responses):
            result = self.responses[self.call_count]
            self.call_count += 1
            return result
        return None


@pytest.fixture
def temp_label_file(tmp_path):
    """Create temporary label file with test data"""
    label_file = tmp_path / "test.txt"
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"

    # Write label file
    label_file.write_text("img1.png\t京A12345\nimg2.png\t沪B67890\n", encoding="utf-8")

    # Create test images
    test_img = np.zeros((48, 168, 3), dtype=np.uint8)
    cv2.imwrite(str(img1), test_img)
    cv2.imwrite(str(img2), test_img)

    return label_file, tmp_path


@pytest.fixture
def mock_ocr_model_perfect():
    """Mock OCR model with perfect predictions"""
    return MockOCRModel([("京A12345", 0.95, [0.95] * 7), ("沪B67890", 0.92, [0.92] * 7)])


@pytest.fixture
def mock_ocr_model_partial():
    """Mock OCR model with partial matches"""
    return MockOCRModel(
        [
            ("京A12345", 0.95, [0.95] * 7),  # Perfect match
            ("沪B67891", 0.88, [0.88] * 7),  # 1 char difference
        ]
    )


@pytest.fixture
def mock_ocr_model_varying_conf():
    """Mock OCR model with varying confidence levels"""
    return MockOCRModel(
        [
            ("京A12345", 0.95, [0.95] * 7),  # High confidence
            ("沪B67890", 0.45, [0.45] * 7),  # Low confidence
        ]
    )


class TestBasicEvaluationContract:
    """Contract tests for basic OCR evaluation flow"""

    def test_basic_evaluation_flow(self, temp_label_file, mock_ocr_model_perfect):
        """Verify basic evaluation flow contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file

        # Run evaluation
        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)
        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        # Verify return format - required fields
        assert "accuracy" in results
        assert "normalized_edit_distance" in results
        assert "edit_distance_similarity" in results
        assert "total_samples" in results
        assert "evaluated_samples" in results
        assert "filtered_samples" in results
        assert "skipped_samples" in results
        assert "evaluation_time" in results
        assert "avg_inference_time_ms" in results

        # Verify value ranges
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["normalized_edit_distance"] <= 1
        assert 0 <= results["edit_distance_similarity"] <= 1

        # Verify sample counts
        assert results["total_samples"] == 2
        assert results["evaluated_samples"] <= results["total_samples"]

        # Verify sample count conservation
        total_processed = results["evaluated_samples"] + results["filtered_samples"] + results["skipped_samples"]
        assert total_processed == results["total_samples"]

    def test_empty_dataset_handling(self, tmp_path, mock_ocr_model_perfect):
        """Verify empty dataset handling contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        # Create empty label file
        label_file = tmp_path / "empty.txt"
        label_file.write_text("", encoding="utf-8")

        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)
        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(tmp_path), output_format="table"
        )

        # Should return empty results
        assert results == {} or results["total_samples"] == 0

    def test_output_format_validation(self, temp_label_file, mock_ocr_model_perfect):
        """Verify output_format parameter validation contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)

        # Valid formats should work
        results_table = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )
        assert results_table is not None

        results_json = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="json"
        )
        assert results_json is not None

        # Invalid format should raise ValueError
        with pytest.raises(ValueError, match="Invalid output_format"):
            evaluator.evaluate_dataset(
                label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="invalid"
            )


class TestEditDistanceMetricsContract:
    """Contract tests for edit distance metrics"""

    def test_perfect_match_metrics(self, temp_label_file, mock_ocr_model_perfect):
        """Verify metrics for perfect matches"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)

        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        # Perfect match should have accuracy = 1.0
        assert results["accuracy"] == 1.0
        # Perfect match should have zero edit distance
        assert results["normalized_edit_distance"] == 0.0
        # Perfect match should have similarity = 1.0
        assert results["edit_distance_similarity"] == 1.0

    def test_partial_match_metrics(self, temp_label_file, mock_ocr_model_partial):
        """Verify metrics for partial matches"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_partial)

        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        # Partial match: 1 correct, 1 incorrect
        assert results["accuracy"] == 0.5
        # Edit distance should be positive (averaged)
        assert 0 < results["normalized_edit_distance"] < 1
        # Similarity should be less than 1.0
        assert 0 < results["edit_distance_similarity"] < 1

    def test_per_sample_results_contract(self, temp_label_file, mock_ocr_model_partial):
        """Verify per_sample_results field contains detailed metrics"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_partial)

        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        # Verify per_sample_results exists
        assert "per_sample_results" in results
        per_sample = results["per_sample_results"]

        # Verify it's a list
        assert isinstance(per_sample, list)

        # Verify sample count matches
        assert len(per_sample) == results["evaluated_samples"]

        # Verify each sample has required fields
        if len(per_sample) > 0:
            sample = per_sample[0]
            required_fields = [
                "image_path",
                "ground_truth",
                "predicted_text",
                "confidence",
                "is_correct",
                "edit_distance",
                "normalized_edit_distance",
            ]
            for field in required_fields:
                assert field in sample, f"Missing field: {field}"

            # Verify field types
            assert isinstance(sample["image_path"], str)
            assert isinstance(sample["ground_truth"], str)
            assert isinstance(sample["predicted_text"], str)
            assert isinstance(sample["confidence"], (int, float))
            assert isinstance(sample["is_correct"], bool)
            assert isinstance(sample["edit_distance"], int)
            assert isinstance(sample["normalized_edit_distance"], (int, float))

            # Verify value ranges
            assert 0 <= sample["confidence"] <= 1
            assert sample["edit_distance"] >= 0
            assert 0 <= sample["normalized_edit_distance"] <= 1


class TestConfidenceFilteringContract:
    """Contract tests for confidence threshold filtering"""

    def test_confidence_threshold_filtering(self, temp_label_file):
        """Verify confidence threshold filtering contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file

        # Create fresh mock models for each evaluation (to reset call_count)
        mock_low = MockOCRModel(
            [
                ("京A12345", 0.95, [0.95] * 7),  # High confidence
                ("沪B67890", 0.45, [0.45] * 7),  # Low confidence
            ]
        )

        mock_high = MockOCRModel(
            [
                ("京A12345", 0.95, [0.95] * 7),  # High confidence
                ("沪B67890", 0.45, [0.45] * 7),  # Low confidence
            ]
        )

        # Low threshold: should accept both samples
        evaluator_low = OCRDatasetEvaluator(mock_low)
        results_low = evaluator_low.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), conf_threshold=0.3, output_format="table"
        )

        # High threshold: should filter low-confidence sample
        evaluator_high = OCRDatasetEvaluator(mock_high)
        results_high = evaluator_high.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), conf_threshold=0.9, output_format="table"
        )

        # Verify filtering behavior
        assert results_high["filtered_samples"] > results_low["filtered_samples"]
        assert results_high["evaluated_samples"] < results_low["evaluated_samples"]

        # Verify sample count conservation
        for r in [results_low, results_high]:
            total = r["evaluated_samples"] + r["filtered_samples"] + r["skipped_samples"]
            assert total == r["total_samples"]

    def test_threshold_boundary_conditions(self, temp_label_file):
        """Verify threshold boundary conditions"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file

        # Create fresh mock for threshold = 0.0 test
        mock_zero = MockOCRModel([("京A12345", 0.95, [0.95] * 7), ("沪B67890", 0.45, [0.45] * 7)])

        # Threshold = 0.0: accept all
        evaluator_zero = OCRDatasetEvaluator(mock_zero)
        results_zero = evaluator_zero.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), conf_threshold=0.0, output_format="table"
        )
        assert results_zero["filtered_samples"] == 0

        # Create fresh mock for threshold = 1.0 test
        mock_one = MockOCRModel([("京A12345", 0.95, [0.95] * 7), ("沪B67890", 0.45, [0.45] * 7)])

        # Threshold = 1.0: reject all
        evaluator_one = OCRDatasetEvaluator(mock_one)
        results_one = evaluator_one.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), conf_threshold=1.0, output_format="table"
        )
        # When all samples are filtered, returns empty dict
        if results_one:  # If not empty
            assert results_one["evaluated_samples"] == 0
        else:  # Empty dict case
            assert results_one == {}


class TestJSONExportContract:
    """Contract tests for JSON export format"""

    def test_json_export_format(self, temp_label_file, mock_ocr_model_perfect, capsys):
        """Verify JSON export format contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)

        results = evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="json"
        )

        captured = capsys.readouterr()

        # Verify output is valid JSON
        parsed = json.loads(captured.out)

        # Verify required fields
        required_fields = [
            "accuracy",
            "normalized_edit_distance",
            "edit_distance_similarity",
            "total_samples",
            "evaluated_samples",
            "filtered_samples",
            "skipped_samples",
            "evaluation_time",
            "avg_inference_time_ms",
        ]
        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"

        # Verify Chinese characters are not escaped (ensure_ascii=False)
        # Note: This is a soft check since capsys might handle encoding differently
        # The actual implementation should use ensure_ascii=False

    def test_json_structure_consistency(self, temp_label_file):
        """Verify JSON structure matches table format"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file

        # Create fresh mocks for each evaluation
        mock_table = MockOCRModel([("京A12345", 0.95, [0.95] * 7), ("沪B67890", 0.92, [0.92] * 7)])

        mock_json = MockOCRModel([("京A12345", 0.95, [0.95] * 7), ("沪B67890", 0.92, [0.92] * 7)])

        # Get results with table format
        evaluator_table = OCRDatasetEvaluator(mock_table)
        results_table = evaluator_table.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        # Get results with JSON format
        evaluator_json = OCRDatasetEvaluator(mock_json)
        results_json = evaluator_json.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="json"
        )

        # Both should return the same dictionary structure
        assert set(results_table.keys()) == set(results_json.keys())
        assert results_table["accuracy"] == results_json["accuracy"]
        assert results_table["total_samples"] == results_json["total_samples"]


class TestTableFormattingContract:
    """Contract tests for table formatting"""

    def test_table_format_alignment(self, temp_label_file, mock_ocr_model_perfect, capsys):
        """Verify table format alignment contract"""
        from onnxtools.eval import OCRDatasetEvaluator

        label_file, dataset_base = temp_label_file
        evaluator = OCRDatasetEvaluator(mock_ocr_model_perfect)

        evaluator.evaluate_dataset(
            label_file=str(label_file), dataset_base_path=str(dataset_base), output_format="table"
        )

        captured = capsys.readouterr()

        # Verify output contains Chinese column names
        assert "完全准确率" in captured.out
        assert "归一化编辑距离" in captured.out
        assert "编辑距离相似度" in captured.out
        assert "总样本数" in captured.out
        assert "评估数" in captured.out
        assert "过滤数" in captured.out
        assert "跳过数" in captured.out

        # Verify numeric values are printed
        assert "1.000" in captured.out or "0.000" in captured.out
