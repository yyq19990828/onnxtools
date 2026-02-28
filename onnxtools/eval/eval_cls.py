"""Classification Dataset Evaluator Module

Provides classification model performance evaluation, including:
- Per-class precision, recall, F1 score
- Confusion matrix computation
- Multi-branch evaluation (e.g., color + layer)
- CSV and ImageFolder dataset loading
- Table and JSON output formats
- Per-sample detailed analysis

Example:
    >>> from onnxtools import ClsDatasetEvaluator, HelmetORT
    >>> from onnxtools.eval.eval_cls import BranchConfig
    >>> model = HelmetORT('models/helmet.onnx')
    >>> evaluator = ClsDatasetEvaluator(model)
    >>> results = evaluator.evaluate_dataset(
    ...     csv_path='data/helmet_val.csv',
    ...     image_dir='data/helmet_images/',
    ...     branches=[BranchConfig(0, 'helmet_missing', {0: 'normal', 1: 'helmet_missing'}, 'helmet')],
    ... )
"""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from onnxtools.infer_onnx.onnx_cls import BaseClsORT

__all__ = ['ClsDatasetEvaluator', 'ClsSampleEvaluation', 'BranchConfig']


@dataclass
class BranchConfig:
    """Configuration for a classification branch in CSV dataset.

    Maps a CSV column to a model output branch with label mapping.

    Attributes:
        branch_index: Index into ClsResult.labels for this branch
        column_name: CSV column name containing ground truth
        label_map: Mapping from CSV values to model label strings.
                   Keys can be int or str; values must match model output labels.
        branch_name: Display name for this branch (default: column_name)

    Examples:
        >>> config = BranchConfig(
        ...     branch_index=0,
        ...     column_name='helmet_missing',
        ...     label_map={0: 'normal', 1: 'helmet_missing'},
        ...     branch_name='helmet'
        ... )
    """
    branch_index: int
    column_name: str
    label_map: Dict[Any, str]
    branch_name: str = ""

    def __post_init__(self):
        if not self.branch_name:
            self.branch_name = self.column_name


@dataclass
class ClsSampleEvaluation:
    """Single sample evaluation result for classification.

    Attributes:
        image_path: Path to the evaluated image
        branch_name: Name of the classification branch
        ground_truth: Ground truth label string
        predicted: Predicted label string
        confidence: Prediction confidence score
        is_correct: Whether prediction matches ground truth
    """
    image_path: str
    branch_name: str
    ground_truth: str
    predicted: str
    confidence: float
    is_correct: bool


class ClsDatasetEvaluator:
    """Classification Dataset Evaluator

    Evaluates classification model performance on labeled datasets with metrics
    including accuracy, precision, recall, F1, and confusion matrix.
    Supports single-branch and multi-branch classification models.

    Attributes:
        model: BaseClsORT model instance

    Examples:
        >>> from onnxtools import HelmetORT
        >>> model = HelmetORT('models/helmet.onnx')
        >>> evaluator = ClsDatasetEvaluator(model)
        >>> results = evaluator.evaluate_dataset(
        ...     csv_path='data/val.csv',
        ...     image_dir='data/images/',
        ...     branches=[BranchConfig(0, 'helmet_missing', {0: 'normal', 1: 'helmet_missing'})],
        ... )
    """

    def __init__(self, model: BaseClsORT):
        """Initialize evaluator.

        Args:
            model: BaseClsORT instance for classification inference
        """
        self.model = model

    @staticmethod
    def load_csv_dataset(
        csv_path: str,
        image_dir: str,
        branches: List[BranchConfig],
        image_column: str = "img_name",
    ) -> List[Dict[str, Any]]:
        """Load dataset from CSV file with branch configurations.

        Each row produces one dataset sample with ground truth labels
        for each configured branch.

        Args:
            csv_path: Path to CSV file
            image_dir: Directory containing images
            branches: List of BranchConfig defining branch-to-column mappings
            image_column: CSV column name for image filename (default: "img_name")

        Returns:
            List of dicts, each containing:
                - image_path: str - full path to image
                - branches: Dict[str, str] - {branch_name: ground_truth_label}

        Raises:
            FileNotFoundError: If CSV file does not exist
            KeyError: If required columns are missing

        Examples:
            >>> dataset = ClsDatasetEvaluator.load_csv_dataset(
            ...     'data/val.csv', 'data/images/',
            ...     [BranchConfig(0, 'label', {0: 'cat', 1: 'dog'})],
            ... )
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        image_base = Path(image_dir)
        dataset = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate columns
            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty or has no header: {csv_path}")

            missing_cols = []
            if image_column not in reader.fieldnames:
                missing_cols.append(image_column)
            for branch in branches:
                if branch.column_name not in reader.fieldnames:
                    missing_cols.append(branch.column_name)
            if missing_cols:
                raise KeyError(
                    f"Missing columns in CSV: {missing_cols}. "
                    f"Available: {reader.fieldnames}"
                )

            for line_num, row in enumerate(reader, 2):
                img_name = row[image_column].strip()
                if not img_name:
                    logging.warning(f"Line {line_num}: empty image name, skipping")
                    continue

                image_path = image_base / img_name
                if not image_path.exists():
                    logging.warning(f"Image not found: {image_path}")
                    continue

                # Map CSV values to labels for each branch
                branch_labels = {}
                skip = False
                for branch in branches:
                    raw_value = row[branch.column_name].strip()

                    # Try int conversion for numeric CSV values
                    mapped_label = None
                    try:
                        int_value = int(raw_value)
                        mapped_label = branch.label_map.get(int_value)
                    except (ValueError, TypeError):
                        pass

                    # Try string lookup
                    if mapped_label is None:
                        mapped_label = branch.label_map.get(raw_value)

                    if mapped_label is None:
                        logging.warning(
                            f"Line {line_num}: unmapped value '{raw_value}' "
                            f"for branch '{branch.branch_name}', skipping"
                        )
                        skip = True
                        break

                    branch_labels[branch.branch_name] = mapped_label

                if skip:
                    continue

                dataset.append({
                    'image_path': str(image_path),
                    'branches': branch_labels,
                })

        logging.info(f"Loaded {len(dataset)} valid samples from {csv_path}")
        return dataset

    @staticmethod
    def load_imagefolder_dataset(
        dataset_dir: str,
        branch_name: str = "class",
    ) -> List[Dict[str, Any]]:
        """Load dataset from ImageFolder structure.

        Expected structure: dataset_dir/class_name/image.jpg
        Only supports single-branch classification.

        Args:
            dataset_dir: Root directory containing class subdirectories
            branch_name: Name for the single branch (default: "class")

        Returns:
            List of dicts with image_path and branches

        Raises:
            FileNotFoundError: If dataset_dir does not exist

        Examples:
            >>> dataset = ClsDatasetEvaluator.load_imagefolder_dataset('data/val/')
        """
        base = Path(dataset_dir)
        if not base.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        dataset = []

        for class_dir in sorted(base.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in image_extensions:
                    continue

                dataset.append({
                    'image_path': str(img_path),
                    'branches': {branch_name: class_name},
                })

        logging.info(
            f"Loaded {len(dataset)} samples from ImageFolder: {dataset_dir}"
        )
        return dataset

    def evaluate_dataset(
        self,
        dataset: Optional[List[Dict]] = None,
        # CSV shortcut parameters
        csv_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        branches: Optional[List[BranchConfig]] = None,
        image_column: str = "img_name",
        # ImageFolder shortcut parameters
        dataset_dir: Optional[str] = None,
        # Common parameters
        conf_threshold: float = 0.5,
        max_images: Optional[int] = None,
        output_format: str = 'table',
    ) -> Dict[str, Any]:
        """Evaluate classification model on dataset.

        Supports three ways to provide data:
        1. Pre-loaded dataset (dataset parameter)
        2. CSV file (csv_path + image_dir + branches)
        3. ImageFolder (dataset_dir)

        Args:
            dataset: Pre-loaded dataset list of dicts
            csv_path: Path to CSV label file
            image_dir: Directory containing images (for CSV)
            branches: List of BranchConfig (for CSV)
            image_column: CSV column for image filename
            dataset_dir: ImageFolder root directory
            conf_threshold: Confidence threshold for filtering
            max_images: Maximum images to evaluate (None = all)
            output_format: 'table' or 'json'

        Returns:
            Dictionary containing evaluation results with structure:
                - overall_accuracy: float
                - total_samples: int
                - evaluated_samples: int
                - skipped_samples: int
                - evaluation_time: float
                - avg_inference_time_ms: float
                - branches: Dict[str, Dict] per-branch metrics
                - per_sample_results: List[Dict]

        Raises:
            ValueError: If no dataset source is provided or output_format invalid
        """
        if output_format not in ('table', 'json'):
            raise ValueError(
                f"Invalid output_format: {output_format}. Must be 'table' or 'json'"
            )

        # Load dataset
        if dataset is None:
            if csv_path is not None:
                if image_dir is None or branches is None:
                    raise ValueError(
                        "csv_path requires image_dir and branches parameters"
                    )
                dataset = self.load_csv_dataset(
                    csv_path, image_dir, branches, image_column
                )
            elif dataset_dir is not None:
                dataset = self.load_imagefolder_dataset(dataset_dir)
            else:
                raise ValueError(
                    "Must provide one of: dataset, csv_path+image_dir+branches, "
                    "or dataset_dir"
                )

        if max_images is not None:
            dataset = dataset[:max_images]

        logging.info(
            f"Starting classification evaluation, total {len(dataset)} images"
        )

        # Collect per-branch predictions
        # branch_name -> (y_true, y_pred, confidences)
        branch_collections: Dict[str, Dict[str, list]] = {}
        sample_evaluations: List[ClsSampleEvaluation] = []
        skipped_count = 0
        start_time = time.time()

        for i, sample in enumerate(dataset):
            if i % 50 == 0 and i > 0:
                pct = (i / len(dataset)) * 100
                logging.info(f"Progress: {i}/{len(dataset)} ({pct:.1f}%)")

            image_path = sample['image_path']
            gt_branches = sample['branches']

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Cannot read image: {image_path}")
                skipped_count += 1
                continue

            # Run inference
            try:
                result = self.model(image)
            except Exception as e:
                logging.error(f"Inference failed: {image_path} - {e}")
                skipped_count += 1
                continue

            # Process each branch
            for branch_name, gt_label in gt_branches.items():
                # Find the branch index from BranchConfig if available
                branch_idx = None
                if branches:
                    for bc in branches:
                        if bc.branch_name == branch_name:
                            branch_idx = bc.branch_index
                            break

                # For ImageFolder, use index 0
                if branch_idx is None:
                    branch_idx = 0

                # Get prediction
                if branch_idx < len(result.labels):
                    pred_label = result.labels[branch_idx]
                    pred_conf = result.confidences[branch_idx]
                else:
                    logging.warning(
                        f"Branch index {branch_idx} out of range for "
                        f"{image_path} (model has {len(result.labels)} branches)"
                    )
                    skipped_count += 1
                    continue

                is_correct = (pred_label == gt_label)

                # Initialize branch collection
                if branch_name not in branch_collections:
                    branch_collections[branch_name] = {
                        'y_true': [], 'y_pred': [], 'confidences': []
                    }

                branch_collections[branch_name]['y_true'].append(gt_label)
                branch_collections[branch_name]['y_pred'].append(pred_label)
                branch_collections[branch_name]['confidences'].append(pred_conf)

                sample_evaluations.append(ClsSampleEvaluation(
                    image_path=image_path,
                    branch_name=branch_name,
                    ground_truth=gt_label,
                    predicted=pred_label,
                    confidence=pred_conf,
                    is_correct=is_correct,
                ))

        evaluation_time = time.time() - start_time
        evaluated_count = len(dataset) - skipped_count

        if evaluated_count == 0:
            logging.warning("No valid evaluation samples")
            return {
                'overall_accuracy': 0.0,
                'total_samples': len(dataset),
                'evaluated_samples': 0,
                'skipped_samples': skipped_count,
                'evaluation_time': evaluation_time,
                'avg_inference_time_ms': 0.0,
                'branches': {},
                'per_sample_results': [],
            }

        # Compute per-branch metrics
        from onnxtools.utils.cls_metrics import (
            compute_classification_metrics,
            format_cls_results_json,
            print_cls_metrics,
        )

        branch_results = {}
        for branch_name, collections in branch_collections.items():
            metrics = compute_classification_metrics(
                y_true=collections['y_true'],
                y_pred=collections['y_pred'],
                confidences=collections['confidences'],
            )
            branch_results[branch_name] = metrics

        # Overall accuracy (average across branches)
        branch_accuracies = [
            br['accuracy'] for br in branch_results.values()
        ]
        overall_accuracy = (
            sum(branch_accuracies) / len(branch_accuracies)
            if branch_accuracies else 0.0
        )

        # Build results
        results = {
            'overall_accuracy': overall_accuracy,
            'total_samples': len(dataset),
            'evaluated_samples': evaluated_count,
            'skipped_samples': skipped_count,
            'evaluation_time': evaluation_time,
            'avg_inference_time_ms': (
                (evaluation_time / evaluated_count * 1000)
                if evaluated_count > 0 else 0.0
            ),
            'branches': branch_results,
            'per_sample_results': [
                {
                    'image_path': e.image_path,
                    'branch_name': e.branch_name,
                    'ground_truth': e.ground_truth,
                    'predicted': e.predicted,
                    'confidence': e.confidence,
                    'is_correct': e.is_correct,
                }
                for e in sample_evaluations
            ],
        }

        # Output
        if output_format == 'json':
            print(format_cls_results_json(results))
        else:
            for branch_name, metrics in branch_results.items():
                print_cls_metrics(metrics, branch_name=branch_name)

        return results
