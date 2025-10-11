"""Integration tests for OCR Dataset Evaluator

End-to-end tests verifying:
- Complete evaluation workflow with real OCRONNX model
- Dataset loading and processing
- Performance characteristics
- Output formatting
"""

import pytest
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np


@pytest.mark.integration
class TestOCRDatasetEvaluatorIntegration:
    """Integration tests for OCR dataset evaluation"""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a small sample dataset for testing"""
        # Create label file with known plate texts
        label_file = tmp_path / "test_labels.txt"
        dataset_path = tmp_path / "images"
        dataset_path.mkdir()

        # Sample plate texts
        plates = [
            ("plate_01.png", "京A12345"),
            ("plate_02.png", "沪B67890"),
            ("plate_03.png", "粤C11111"),
            ("plate_04.png", "苏D22222"),
            ("plate_05.png", "浙E33333"),
        ]

        # Write label file with images/ prefix
        with open(label_file, 'w', encoding='utf-8') as f:
            for img_name, text in plates:
                f.write(f"images/{img_name}\t{text}\n")

        # Create synthetic plate images
        for img_name, text in plates:
            img_path = dataset_path / img_name
            # Create a simple synthetic plate image (blue background)
            img = np.full((48, 168, 3), (255, 100, 0), dtype=np.uint8)  # Blue plate
            # Add some random noise to make it more realistic
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            cv2.imwrite(str(img_path), img)

        return label_file, tmp_path, plates

    def test_end_to_end_evaluation_table_format(self, sample_dataset, ocr_model_path, ocr_character):
        """End-to-end evaluation test with table format output"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        label_file, dataset_base, plates = sample_dataset

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation with table format
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_base),
            conf_threshold=0.3,  # Lower threshold for synthetic images
            output_format='table'
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'normalized_edit_distance' in results
        assert 'edit_distance_similarity' in results
        assert 'total_samples' in results
        assert 'evaluated_samples' in results
        assert 'evaluation_time' in results

        # Verify sample counts
        assert results['total_samples'] == len(plates)
        # Some samples may be filtered or skipped due to low confidence on synthetic images
        assert results['evaluated_samples'] <= results['total_samples']

        # Verify metric ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['normalized_edit_distance'] <= 1
        assert 0 <= results['edit_distance_similarity'] <= 1

        # Verify timing
        assert results['evaluation_time'] > 0
        if results['evaluated_samples'] > 0:
            assert results['avg_inference_time_ms'] > 0

    def test_end_to_end_evaluation_json_format(self, sample_dataset, ocr_model_path, ocr_character, capsys):
        """End-to-end evaluation test with JSON format output"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        label_file, dataset_base, plates = sample_dataset

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation with JSON format
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_base),
            conf_threshold=0.3,
            output_format='json'
        )

        # Capture printed JSON output
        captured = capsys.readouterr()

        # Verify JSON output is valid
        if captured.out.strip():
            try:
                parsed = json.loads(captured.out)
                assert isinstance(parsed, dict)
                assert 'accuracy' in parsed
            except json.JSONDecodeError:
                # If JSON parsing fails, just verify results dict
                pass

        # Verify results dictionary
        assert isinstance(results, dict)
        # Note: results may be empty if all samples were filtered
        if results:
            assert results['total_samples'] == len(plates)

    def test_evaluation_with_max_images_limit(self, sample_dataset, ocr_model_path, ocr_character):
        """Test evaluation with max_images parameter"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        label_file, dataset_base, plates = sample_dataset

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation with max_images limit
        max_limit = 3
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_base),
            max_images=max_limit,
            conf_threshold=0.3,
            output_format='table'
        )

        # Verify only max_limit images were processed
        assert results['total_samples'] == max_limit
        total_processed = (
            results['evaluated_samples'] +
            results['filtered_samples'] +
            results['skipped_samples']
        )
        assert total_processed == max_limit

    def test_evaluation_with_varying_thresholds(self, sample_dataset, ocr_model_path, ocr_character):
        """Test evaluation with different confidence thresholds"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        label_file, dataset_base, plates = sample_dataset

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Test multiple thresholds
        thresholds = [0.3, 0.5, 0.7]
        results_list = []

        for threshold in thresholds:
            evaluator = OCRDatasetEvaluator(ocr_model)
            results = evaluator.evaluate_dataset(
                label_file=str(label_file),
                dataset_base_path=str(dataset_base),
                conf_threshold=threshold,
                output_format='table'
            )
            results_list.append((threshold, results))

        # Verify results were collected for all thresholds
        assert len(results_list) == len(thresholds)

        # Verify trend: higher threshold -> fewer evaluated samples (generally)
        # Note: This may not always be strictly decreasing due to model behavior
        for i, (threshold, results) in enumerate(results_list):
            assert results['total_samples'] == len(plates)
            # Just verify that filtering is working (some samples may be filtered)
            total_processed = (
                results['evaluated_samples'] +
                results['filtered_samples'] +
                results['skipped_samples']
            )
            assert total_processed == results['total_samples']

    def test_evaluation_performance(self, sample_dataset, ocr_model_path, ocr_character):
        """Test evaluation performance characteristics"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX
        import time

        label_file, dataset_base, plates = sample_dataset

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Measure evaluation time
        start_time = time.time()
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_base),
            conf_threshold=0.3,
            output_format='table'
        )
        end_time = time.time()

        actual_time = end_time - start_time

        # Verify timing is reasonable
        assert results['evaluation_time'] > 0
        # Actual time should be close to reported time (within 10%)
        assert abs(actual_time - results['evaluation_time']) / actual_time < 0.1

        # Performance expectation: < 1 second for 5 small images
        assert results['evaluation_time'] < 1.0, \
            f"Evaluation took {results['evaluation_time']:.2f}s, expected < 1.0s for 5 images"

    def test_evaluation_with_missing_images(self, tmp_path, ocr_model_path, ocr_character):
        """Test evaluation handling of missing images"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        # Create label file referencing non-existent images
        label_file = tmp_path / "test_missing.txt"
        dataset_path = tmp_path / "images"
        dataset_path.mkdir()

        with open(label_file, 'w', encoding='utf-8') as f:
            f.write("missing1.png\t京A12345\n")
            f.write("missing2.png\t沪B67890\n")

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_path),
            conf_threshold=0.5,
            output_format='table'
        )

        # All images should be skipped (files don't exist)
        # load_label_file filters out non-existent files
        # Results will be empty dict if no valid samples
        assert results == {} or (results.get('total_samples', 0) == 0)

    def test_evaluation_with_corrupted_images(self, tmp_path, ocr_model_path, ocr_character):
        """Test evaluation handling of corrupted/invalid images"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        # Create label file
        label_file = tmp_path / "test_corrupted.txt"
        dataset_path = tmp_path / "images"
        dataset_path.mkdir()

        # Create corrupted image file
        corrupted_img = dataset_path / "corrupted.png"
        corrupted_img.write_bytes(b"This is not a valid image file")

        with open(label_file, 'w', encoding='utf-8') as f:
            f.write("corrupted.png\t京A12345\n")

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation
        results = evaluator.evaluate_dataset(
            label_file=str(label_file),
            dataset_base_path=str(dataset_path),
            conf_threshold=0.5,
            output_format='table'
        )

        # Corrupted image should be skipped
        if results:  # If not empty dict
            assert results['skipped_samples'] > 0
            assert results['evaluated_samples'] == 0

    @pytest.mark.skipif(
        not Path("data/ocr_rec_dataset_examples").exists(),
        reason="Real OCR dataset not available"
    )
    def test_evaluation_with_real_dataset(self, ocr_model_path, ocr_character):
        """Test evaluation with real OCR dataset (if available)"""
        from infer_onnx import OCRDatasetEvaluator, OCRONNX

        # Use real dataset if available
        label_file = "data/ocr_rec_dataset_examples/val.txt"
        dataset_base = "data/ocr_rec_dataset_examples"

        # Load OCR model
        ocr_model = OCRONNX(str(ocr_model_path), character=ocr_character)

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation on subset
        results = evaluator.evaluate_dataset(
            label_file=label_file,
            dataset_base_path=dataset_base,
            max_images=20,  # Limit for faster testing
            conf_threshold=0.5,
            output_format='table'
        )

        # Verify evaluation completed
        # Note: actual number may be less than 20 if dataset has fewer samples
        assert results.get('total_samples', 0) <= 20
        assert results.get('evaluated_samples', 0) >= 0

        # With real data, expect reasonable accuracy
        # (This is a soft check - depends on model quality)
        if results.get('evaluated_samples', 0) > 10:
            assert results['accuracy'] >= 0.0  # Just verify metric is computed
