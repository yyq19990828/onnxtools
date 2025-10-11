"""OCR Metrics Calculation Functions

Provides core OCR evaluation metrics including:
- Edit distance metrics (Levenshtein distance)
- Accuracy calculation
- Table formatting for Chinese characters
- JSON export functionality
"""

from typing import List, Tuple, Dict, Any
import Levenshtein
import json

__all__ = [
    'calculate_edit_distance_metrics',
    'calculate_accuracy',
    'print_ocr_metrics',
    'format_ocr_results_json'
]


def calculate_edit_distance_metrics(pred: str, gt: str) -> Tuple[int, float, float]:
    """Calculate edit distance related metrics

    Args:
        pred: Predicted text
        gt: Ground truth text

    Returns:
        (edit_distance, normalized_edit_distance, edit_distance_similarity)
        - edit_distance: Levenshtein distance (integer)
        - normalized_edit_distance: distance / max(len(pred), len(gt)) in [0, 1]
        - edit_distance_similarity: 1 - normalized_edit_distance in [0, 1]

    Examples:
        >>> calculate_edit_distance_metrics("京A12345", "京A12345")
        (0, 0.0, 1.0)
        >>> calculate_edit_distance_metrics("京A12345", "京A12346")
        (1, 0.14285714285714285, 0.8571428571428571)
    """
    distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    normalized_ed = distance / max_len if max_len > 0 else 0.0
    similarity = 1.0 - normalized_ed

    return distance, normalized_ed, similarity


def calculate_accuracy(predictions: List[Tuple[str, str]]) -> float:
    """Calculate complete accuracy (exact match ratio)

    Args:
        predictions: List of (predicted_text, ground_truth) tuples

    Returns:
        accuracy: Ratio of exact matches in [0, 1]

    Examples:
        >>> calculate_accuracy([("A", "A"), ("B", "B"), ("C", "C")])
        1.0
        >>> calculate_accuracy([("A", "A"), ("B", "X"), ("C", "C")])
        0.6666666666666666
    """
    if not predictions:
        return 0.0

    correct_count = sum(1 for pred, gt in predictions if pred == gt)
    return correct_count / len(predictions)


def print_ocr_metrics(results: Dict[str, Any]) -> None:
    """Print OCR evaluation metrics with Chinese character alignment

    Prints a two-row table format:
    - Row 1: Core metrics (accuracy, normalized edit distance, similarity)
    - Row 2: Sample statistics (total, evaluated, filtered, skipped)

    Args:
        results: Evaluation results dictionary containing:
            - accuracy: float
            - normalized_edit_distance: float
            - edit_distance_similarity: float
            - total_samples: int
            - evaluated_samples: int
            - filtered_samples: int
            - skipped_samples: int

    Examples:
        >>> results = {
        ...     'accuracy': 0.925,
        ...     'normalized_edit_distance': 0.045,
        ...     'edit_distance_similarity': 0.955,
        ...     'total_samples': 1000,
        ...     'evaluated_samples': 980,
        ...     'filtered_samples': 15,
        ...     'skipped_samples': 5
        ... }
        >>> print_ocr_metrics(results)
        指标                  完全准确率        归一化编辑距离      编辑距离相似度
        OCR评估              0.925           0.045           0.955

        统计信息              总样本数          评估数            过滤数            跳过数
        样本统计              1000            980             15              5
    """
    # Helper function to calculate display width (Chinese chars = 2, ASCII = 1)
    def display_width(s: str) -> int:
        """Calculate display width considering Chinese characters"""
        width = 0
        for char in s:
            # Chinese characters and full-width chars occupy 2 columns
            if '\u4e00' <= char <= '\u9fff' or '\uff00' <= char <= '\uffef':
                width += 2
            else:
                width += 1
        return width

    def pad_string(s: str, target_width: int, align: str = 'center') -> str:
        """Pad string to target display width considering Chinese characters"""
        current_width = display_width(s)
        padding_needed = target_width - current_width

        if padding_needed <= 0:
            return s

        if align == 'center':
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            return ' ' * left_pad + s + ' ' * right_pad
        elif align == 'left':
            return s + ' ' * padding_needed
        else:  # right
            return ' ' * padding_needed + s

    # First row: Core metrics header
    header_line1 = (
        pad_string('指标', 20) +
        pad_string('完全准确率', 20) +
        pad_string('归一化编辑距离', 20) +
        pad_string('编辑距离相似度', 20)
    )
    print(header_line1)

    # First row: Core metrics values
    acc = results.get('accuracy', 0)
    norm_ed = results.get('normalized_edit_distance', 0)
    ed_sim = results.get('edit_distance_similarity', 0)

    metrics_line = (
        pad_string('OCR评估', 20) +
        pad_string(f'{acc:.3f}', 20) +
        pad_string(f'{norm_ed:.3f}', 20) +
        pad_string(f'{ed_sim:.3f}', 20)
    )
    print(metrics_line)
    print()  # Empty line separator

    # Second row: Sample statistics header
    header_line2 = (
        pad_string('统计信息', 20) +
        pad_string('总样本数', 15) +
        pad_string('评估数', 15) +
        pad_string('过滤数', 15) +
        pad_string('跳过数', 15)
    )
    print(header_line2)

    # Second row: Sample statistics values
    total = results.get('total_samples', 0)
    evaluated = results.get('evaluated_samples', 0)
    filtered = results.get('filtered_samples', 0)
    skipped = results.get('skipped_samples', 0)

    stats_line = (
        pad_string('样本统计', 20) +
        pad_string(str(total), 15) +
        pad_string(str(evaluated), 15) +
        pad_string(str(filtered), 15) +
        pad_string(str(skipped), 15)
    )
    print(stats_line)


def format_ocr_results_json(results: Dict[str, Any]) -> str:
    """Format OCR evaluation results as JSON string

    Args:
        results: Evaluation results dictionary

    Returns:
        JSON formatted string with 2-space indentation and UTF-8 encoding

    Examples:
        >>> results = {'accuracy': 0.925, 'total_samples': 100}
        >>> json_str = format_ocr_results_json(results)
        >>> import json
        >>> parsed = json.loads(json_str)
        >>> parsed['accuracy']
        0.925
    """
    return json.dumps(results, indent=2, ensure_ascii=False)
