"""Classification Metrics Calculation Functions

Provides core classification evaluation metrics including:
- Per-class precision, recall, F1 score
- Confusion matrix computation
- Overall accuracy calculation
- Table formatting for Chinese characters
- JSON export functionality
"""

import json
from typing import Any, Dict, List, Optional

__all__ = [
    'compute_classification_metrics',
    'print_cls_metrics',
    'format_cls_results_json',
]


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics without sklearn dependency.

    Calculates accuracy, per-class precision/recall/F1, average confidence,
    and confusion matrix from ground truth and predicted labels.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Confidence scores for each prediction
        class_names: Ordered list of class names. If None, inferred from y_true + y_pred.

    Returns:
        Dictionary containing:
            - accuracy: float in [0, 1]
            - per_class: Dict[str, Dict] with precision, recall, f1, support, avg_confidence
            - confusion_matrix: List[List[int]] rows=true, cols=pred
            - class_names: List[str] ordered class names
            - total_samples: int

    Examples:
        >>> results = compute_classification_metrics(
        ...     ['cat', 'dog', 'cat'], ['cat', 'cat', 'cat'],
        ...     [0.9, 0.7, 0.8]
        ... )
        >>> results['accuracy']
        0.6666666666666666
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(confidences):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"confidences={len(confidences)}"
        )

    if len(y_true) == 0:
        return {
            'accuracy': 0.0,
            'per_class': {},
            'confusion_matrix': [],
            'class_names': class_names or [],
            'total_samples': 0,
        }

    # Determine class names
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    # Build confusion matrix
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx.get(true_label)
        pred_idx = class_to_idx.get(pred_label)
        if true_idx is not None and pred_idx is not None:
            confusion[true_idx][pred_idx] += 1

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Per-class metrics
    per_class = {}
    for i, cls_name in enumerate(class_names):
        tp = confusion[i][i]
        # support = number of true samples for this class
        support = sum(confusion[i])
        # predicted as this class
        predicted_as = sum(confusion[j][i] for j in range(n_classes))

        precision = tp / predicted_as if predicted_as > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Average confidence for predictions of this class
        cls_confidences = [
            conf for pred, conf in zip(y_pred, confidences)
            if pred == cls_name
        ]
        avg_conf = (sum(cls_confidences) / len(cls_confidences)
                    if cls_confidences else 0.0)

        per_class[cls_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'avg_confidence': avg_conf,
        }

    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'confusion_matrix': confusion,
        'class_names': class_names,
        'total_samples': len(y_true),
    }


def print_cls_metrics(results: Dict[str, Any], branch_name: str = "") -> None:
    """Print classification evaluation metrics with Chinese character alignment.

    Prints per-class metrics table and overall statistics.

    Args:
        results: Classification metrics dictionary from compute_classification_metrics
        branch_name: Optional branch display name (e.g., 'helmet', 'color')

    Examples:
        >>> results = compute_classification_metrics(
        ...     ['a', 'b', 'a'], ['a', 'a', 'a'], [0.9, 0.7, 0.8]
        ... )
        >>> print_cls_metrics(results, branch_name='test')
    """
    def display_width(s: str) -> int:
        width = 0
        for char in s:
            if '\u4e00' <= char <= '\u9fff' or '\uff00' <= char <= '\uffef':
                width += 2
            else:
                width += 1
        return width

    def pad_string(s: str, target_width: int, align: str = 'left') -> str:
        current_width = display_width(s)
        padding_needed = target_width - current_width
        if padding_needed <= 0:
            return s
        if align == 'left':
            return s + ' ' * padding_needed
        elif align == 'right':
            return ' ' * padding_needed + s
        else:
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            return ' ' * left_pad + s + ' ' * right_pad

    title = f"分类评估 - {branch_name}" if branch_name else "分类评估"
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    accuracy = results.get('accuracy', 0.0)
    total = results.get('total_samples', 0)
    print(f"  总体准确率: {accuracy:.4f}  ({total} 样本)")
    print(f"{'-' * 70}")

    per_class = results.get('per_class', {})
    if per_class:
        # Header
        col_widths = [18, 12, 12, 12, 10, 14]
        headers = ['类别', 'Precision', 'Recall', 'F1', 'Support', 'Avg Conf']
        header_line = '  '.join(
            pad_string(h, w, 'left') for h, w in zip(headers, col_widths)
        )
        print(f"  {header_line}")
        print(f"  {'-' * (sum(col_widths) + 2 * (len(col_widths) - 1))}")

        # Per-class rows
        for cls_name, metrics in per_class.items():
            row = [
                pad_string(cls_name, col_widths[0]),
                pad_string(f"{metrics['precision']:.4f}", col_widths[1]),
                pad_string(f"{metrics['recall']:.4f}", col_widths[2]),
                pad_string(f"{metrics['f1']:.4f}", col_widths[3]),
                pad_string(str(metrics['support']), col_widths[4]),
                pad_string(f"{metrics['avg_confidence']:.4f}", col_widths[5]),
            ]
            print(f"  {'  '.join(row)}")

    print(f"{'=' * 70}\n")


def format_cls_results_json(results: Dict[str, Any]) -> str:
    """Format classification evaluation results as JSON string.

    Args:
        results: Evaluation results dictionary

    Returns:
        JSON formatted string with 2-space indentation and UTF-8 encoding

    Examples:
        >>> results = {'accuracy': 0.925, 'total_samples': 100}
        >>> json_str = format_cls_results_json(results)
        >>> import json
        >>> parsed = json.loads(json_str)
        >>> parsed['accuracy']
        0.925
    """
    return json.dumps(results, indent=2, ensure_ascii=False)
