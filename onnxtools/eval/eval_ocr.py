"""OCR Dataset Evaluator Module

Provides OCR model performance evaluation functions, including:
- Complete accuracy calculation
- Normalized edit distance and similarity
- Confidence filtering
- Table and JSON output formats
- Per-sample detailed analysis

Example:
    >>> from onnxtools import OCRDatasetEvaluator, OcrORT
    >>> ocr_model = OcrORT('models/ocr.onnx', character=char_dict)
    >>> evaluator = OCRDatasetEvaluator(ocr_model)
    >>> results = evaluator.evaluate_dataset(
    ...     label_file='dataset/val.txt',
    ...     dataset_base_path='dataset'
    ... )
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

__all__ = ['OCRDatasetEvaluator', 'SampleEvaluation']


@dataclass
class SampleEvaluation:
    """Single sample evaluation result with detailed metrics

    Attributes:
        image_path: Path to the evaluated image
        ground_truth: Ground truth text
        predicted_text: Predicted text from OCR
        confidence: OCR prediction confidence score
        is_correct: Whether prediction exactly matches ground truth
        edit_distance: Raw Levenshtein distance
        normalized_edit_distance: Edit distance normalized by max length

    Examples:
        >>> sample = SampleEvaluation(
        ...     image_path='img1.png',
        ...     ground_truth='京A12345',
        ...     predicted_text='京A12345',
        ...     confidence=0.95,
        ...     is_correct=True,
        ...     edit_distance=0,
        ...     normalized_edit_distance=0.0
        ... )
        >>> sample.is_correct
        True
    """
    image_path: str
    ground_truth: str
    predicted_text: str
    confidence: float
    is_correct: bool
    edit_distance: int
    normalized_edit_distance: float


class OCRDatasetEvaluator:
    """OCR Dataset Evaluator

    Evaluates OCR model performance on labeled datasets with metrics including:
    - Complete accuracy (exact match ratio)
    - Normalized edit distance
    - Edit distance similarity
    - Confidence-based filtering
    - Performance statistics

    Attributes:
        ocr_model: OcrORT model instance

    Examples:
        >>> from infer_onnx import OCRONNX, OCRDatasetEvaluator
        >>> ocr_model = OcrORT('models/ocr.onnx', character=char_dict)
        >>> evaluator = OCRDatasetEvaluator(ocr_model)
        >>> results = evaluator.evaluate_dataset(
        ...     label_file='dataset/val.txt',
        ...     dataset_base_path='dataset',
        ...     conf_threshold=0.5
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.3f}")
    """

    @staticmethod
    def load_label_file(label_file: str, dataset_base_path: str) -> List[Tuple[str, str]]:
        """Load label file with tab-separated format

        Supports two formats:
        1. Single image: image_path<TAB>ground_truth
        2. Multiple images (JSON): ["img1.jpg", "img2.jpg"]<TAB>ground_truth

        Args:
            label_file: Path to label file (e.g., train.txt, val.txt)
            dataset_base_path: Dataset root directory for resolving relative paths

        Returns:
            List of (image_path, ground_truth_text) tuples
            For multiple images per label, expands to multiple entries

        Raises:
            FileNotFoundError: If label file does not exist
            IOError: If label file cannot be read

        Examples:
            >>> dataset = OCRDatasetEvaluator.load_label_file('data/train.txt', 'data/')
            >>> len(dataset) > 0
            True
            >>> dataset[0][1]  # ground truth text
            '京A12345'
        """
        import json

        label_path = Path(label_file)
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        dataset = []
        base_path = Path(dataset_base_path)

        with open(label_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    logging.warning(f"Skipping line {line_num}: Invalid format (expected tab-separated)")
                    continue

                image_path_str, gt_text = parts

                # Check if image_path is a JSON array (multiple images)
                image_paths = []
                if image_path_str.startswith('[') and image_path_str.endswith(']'):
                    try:
                        # Parse JSON array of image paths
                        image_paths = json.loads(image_path_str)
                        if not isinstance(image_paths, list):
                            logging.warning(f"Line {line_num}: Invalid JSON array format")
                            continue
                    except json.JSONDecodeError as e:
                        logging.warning(f"Line {line_num}: Failed to parse JSON array: {e}")
                        continue
                else:
                    # Single image path
                    image_paths = [image_path_str]

                # Expand multiple images to individual entries
                for img_path in image_paths:
                    full_path = base_path / img_path

                    if not full_path.exists():
                        logging.warning(f"Skipping image: File not found {full_path}")
                        continue

                    dataset.append((str(full_path), gt_text))

        logging.info(f"Loaded {len(dataset)} valid samples from {label_file}")
        return dataset

    def __init__(self, ocr_model):
        """Initialize evaluator

        Args:
            ocr_model: OcrORT instance for OCR inference
        """
        self.ocr_model = ocr_model

    def evaluate_dataset(
        self,
        label_file: str,
        dataset_base_path: str,
        conf_threshold: float = 0.5,
        max_images: Optional[int] = None,
        output_format: str = 'table',
        min_width: int = 40
    ) -> Dict[str, Any]:
        """Evaluate OCR model on dataset

        Args:
            label_file: Path to label file (tab-separated format)
            dataset_base_path: Dataset root directory
            conf_threshold: Confidence threshold for filtering (default: 0.5)
            max_images: Maximum number of images to evaluate (default: None, evaluate all)
            output_format: Output format, 'table' or 'json' (default: 'table')
            min_width: Minimum image width for evaluation (default: 40)

        Returns:
            Dictionary containing evaluation results:
                - accuracy: float in [0, 1]
                - normalized_edit_distance: float in [0, 1]
                - edit_distance_similarity: float in [0, 1]
                - total_samples: int
                - evaluated_samples: int
                - filtered_samples: int
                - skipped_samples: int
                - evaluation_time: float (seconds)
                - avg_inference_time_ms: float (milliseconds)

        Raises:
            ValueError: If output_format is not 'table' or 'json'
            FileNotFoundError: If label_file does not exist

        Examples:
            >>> evaluator = OCRDatasetEvaluator(ocr_model)
            >>> results = evaluator.evaluate_dataset(
            ...     label_file='data/val.txt',
            ...     dataset_base_path='data/',
            ...     conf_threshold=0.7,
            ...     output_format='json'
            ... )
            >>> 0 <= results['accuracy'] <= 1
            True
        """
        # Validate output format
        if output_format not in ['table', 'json']:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'table' or 'json'")

        # Import utility functions
        from onnxtools.utils.ocr_metrics import (
            calculate_accuracy,
            calculate_edit_distance_metrics,
            format_ocr_results_json,
            print_ocr_metrics,
        )

        # Load dataset
        dataset = self.load_label_file(label_file, dataset_base_path)
        if max_images:
            dataset = dataset[:max_images]

        logging.info(f"Starting OCR dataset evaluation, total {len(dataset)} images")

        # Initialize statistics
        evaluations = []  # For backward compatibility
        detailed_evaluations = []  # New: SampleEvaluation objects
        filtered_count = 0
        skipped_count = 0
        start_time = time.time()

        # Main evaluation loop
        for i, (image_path, gt_text) in enumerate(dataset):
            # Progress logging every 50 images
            if i % 50 == 0:
                percentage = (i / len(dataset)) * 100
                logging.info(f"Processing progress: {i}/{len(dataset)} ({percentage:.1f}%)")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Cannot read image: {image_path}")
                skipped_count += 1
                continue

            # Width filtering
            img_height, img_width = image.shape[:2]
            if img_width < min_width:
                logging.debug(f"Filtered by width: {image_path} (width={img_width} < {min_width})")
                filtered_count += 1
                continue

            # OCR inference
            try:
                result = self.ocr_model(image)
                if result is None:
                    skipped_count += 1
                    continue

                pred_text, confidence, _ = result

                # Confidence filtering
                if confidence < conf_threshold:
                    logging.debug(f"Filtered low confidence sample: {image_path} (conf={confidence:.3f} < {conf_threshold})")
                    filtered_count += 1
                    continue

                # Calculate edit distance metrics for this sample
                dist, norm_ed, ed_sim = calculate_edit_distance_metrics(pred_text, gt_text)

                # Create detailed evaluation record
                sample_eval = SampleEvaluation(
                    image_path=image_path,
                    ground_truth=gt_text,
                    predicted_text=pred_text,
                    confidence=confidence,
                    is_correct=(pred_text == gt_text),
                    edit_distance=dist,
                    normalized_edit_distance=norm_ed
                )
                detailed_evaluations.append(sample_eval)

                # Keep old format for backward compatibility
                evaluations.append((pred_text, gt_text))

            except Exception as e:
                logging.error(f"OCR inference failed: {image_path} - {e}")
                skipped_count += 1
                continue

        evaluation_time = time.time() - start_time

        # Aggregate metrics
        if not evaluations:
            logging.warning("No valid evaluation samples")
            return {}

        accuracy = calculate_accuracy(evaluations)

        # Calculate average edit distance metrics
        total_norm_ed = 0.0
        total_ed_sim = 0.0
        for pred, gt in evaluations:
            _, norm_ed, ed_sim = calculate_edit_distance_metrics(pred, gt)
            total_norm_ed += norm_ed
            total_ed_sim += ed_sim

        avg_norm_ed = total_norm_ed / len(evaluations)
        avg_ed_sim = total_ed_sim / len(evaluations)

        # Build results dictionary
        results = {
            'accuracy': accuracy,
            'normalized_edit_distance': avg_norm_ed,
            'edit_distance_similarity': avg_ed_sim,
            'total_samples': len(dataset),
            'evaluated_samples': len(evaluations),
            'filtered_samples': filtered_count,
            'skipped_samples': skipped_count,
            'evaluation_time': evaluation_time,
            'avg_inference_time_ms': (evaluation_time / len(evaluations) * 1000) if evaluations else 0,
            'per_sample_results': [
                {
                    'image_path': e.image_path,
                    'ground_truth': e.ground_truth,
                    'predicted_text': e.predicted_text,
                    'confidence': e.confidence,
                    'is_correct': e.is_correct,
                    'edit_distance': e.edit_distance,
                    'normalized_edit_distance': e.normalized_edit_distance
                }
                for e in detailed_evaluations
            ]
        }

        # Output results
        if output_format == 'json':
            print(format_ocr_results_json(results))
        else:  # table
            print_ocr_metrics(results)

        return results
