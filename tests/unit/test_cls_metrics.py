"""Unit tests for classification metrics calculation functions.

Tests compute_classification_metrics, print_cls_metrics, and format_cls_results_json.
"""

import json

import pytest

from onnxtools.utils.cls_metrics import compute_classification_metrics, format_cls_results_json, print_cls_metrics


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    def test_perfect_predictions(self):
        """All correct predictions should give accuracy=1.0."""
        results = compute_classification_metrics(
            y_true=['cat', 'dog', 'cat', 'dog'],
            y_pred=['cat', 'dog', 'cat', 'dog'],
            confidences=[0.9, 0.8, 0.95, 0.85],
        )
        assert results['accuracy'] == 1.0
        assert results['total_samples'] == 4

    def test_all_wrong_predictions(self):
        """All incorrect predictions should give accuracy=0.0."""
        results = compute_classification_metrics(
            y_true=['cat', 'dog'],
            y_pred=['dog', 'cat'],
            confidences=[0.9, 0.8],
        )
        assert results['accuracy'] == 0.0

    def test_partial_predictions(self):
        """Partial matches should give correct accuracy."""
        results = compute_classification_metrics(
            y_true=['a', 'b', 'a', 'b'],
            y_pred=['a', 'a', 'a', 'b'],
            confidences=[0.9, 0.7, 0.8, 0.85],
        )
        assert results['accuracy'] == 0.75

    def test_empty_inputs(self):
        """Empty inputs should return zero metrics."""
        results = compute_classification_metrics(
            y_true=[], y_pred=[], confidences=[],
        )
        assert results['accuracy'] == 0.0
        assert results['total_samples'] == 0
        assert results['per_class'] == {}
        assert results['confusion_matrix'] == []

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_classification_metrics(
                y_true=['a', 'b'],
                y_pred=['a'],
                confidences=[0.9, 0.8],
            )

    def test_per_class_precision(self):
        """Precision should be TP / (TP + FP)."""
        results = compute_classification_metrics(
            y_true=['cat', 'dog', 'dog'],
            y_pred=['cat', 'cat', 'dog'],
            confidences=[0.9, 0.7, 0.8],
        )
        # cat: TP=1, FP=1 (dog predicted as cat), precision=0.5
        assert results['per_class']['cat']['precision'] == 0.5
        # dog: TP=1, FP=0, precision=1.0
        assert results['per_class']['dog']['precision'] == 1.0

    def test_per_class_recall(self):
        """Recall should be TP / (TP + FN)."""
        results = compute_classification_metrics(
            y_true=['cat', 'cat', 'dog'],
            y_pred=['cat', 'dog', 'dog'],
            confidences=[0.9, 0.7, 0.8],
        )
        # cat: TP=1, FN=1, recall=0.5
        assert results['per_class']['cat']['recall'] == 0.5
        # dog: TP=1, FN=0, recall=1.0
        assert results['per_class']['dog']['recall'] == 1.0

    def test_per_class_f1(self):
        """F1 should be 2*P*R/(P+R)."""
        results = compute_classification_metrics(
            y_true=['cat', 'dog', 'cat'],
            y_pred=['cat', 'cat', 'cat'],
            confidences=[0.9, 0.7, 0.8],
        )
        cat_metrics = results['per_class']['cat']
        p = cat_metrics['precision']
        r = cat_metrics['recall']
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        assert abs(cat_metrics['f1'] - expected_f1) < 1e-10

    def test_per_class_support(self):
        """Support should be count of true samples per class."""
        results = compute_classification_metrics(
            y_true=['a', 'a', 'a', 'b', 'b'],
            y_pred=['a', 'a', 'b', 'b', 'a'],
            confidences=[0.9] * 5,
        )
        assert results['per_class']['a']['support'] == 3
        assert results['per_class']['b']['support'] == 2

    def test_per_class_avg_confidence(self):
        """Avg confidence should be mean of confidences for predictions of that class."""
        results = compute_classification_metrics(
            y_true=['a', 'b'],
            y_pred=['a', 'a'],
            confidences=[0.9, 0.7],
        )
        # Both predicted as 'a', avg_conf = (0.9 + 0.7) / 2 = 0.8
        assert abs(results['per_class']['a']['avg_confidence'] - 0.8) < 1e-10
        # Nothing predicted as 'b'
        assert results['per_class']['b']['avg_confidence'] == 0.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be NxN where N is number of classes."""
        results = compute_classification_metrics(
            y_true=['a', 'b', 'c'],
            y_pred=['a', 'c', 'b'],
            confidences=[0.9, 0.8, 0.7],
        )
        cm = results['confusion_matrix']
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_confusion_matrix_values(self):
        """Confusion matrix diagonal should be correct counts."""
        results = compute_classification_metrics(
            y_true=['a', 'a', 'b', 'b'],
            y_pred=['a', 'b', 'a', 'b'],
            confidences=[0.9, 0.8, 0.7, 0.6],
            class_names=['a', 'b'],
        )
        cm = results['confusion_matrix']
        # true=a, pred=a: 1
        assert cm[0][0] == 1
        # true=a, pred=b: 1
        assert cm[0][1] == 1
        # true=b, pred=a: 1
        assert cm[1][0] == 1
        # true=b, pred=b: 1
        assert cm[1][1] == 1

    def test_custom_class_names(self):
        """Custom class_names should be used for ordering."""
        results = compute_classification_metrics(
            y_true=['b', 'a'],
            y_pred=['a', 'a'],
            confidences=[0.9, 0.8],
            class_names=['a', 'b'],
        )
        assert results['class_names'] == ['a', 'b']

    def test_zero_division_protection(self):
        """Classes with no predictions should have precision=0, not error."""
        results = compute_classification_metrics(
            y_true=['a', 'b'],
            y_pred=['a', 'a'],
            confidences=[0.9, 0.8],
        )
        # 'b' has zero predictions, precision should be 0
        assert results['per_class']['b']['precision'] == 0.0
        # 'b' recall = 0/1 = 0
        assert results['per_class']['b']['recall'] == 0.0
        # f1 = 0 (both precision and recall are 0)
        assert results['per_class']['b']['f1'] == 0.0

    def test_single_class(self):
        """Single class should work correctly."""
        results = compute_classification_metrics(
            y_true=['only'],
            y_pred=['only'],
            confidences=[0.99],
        )
        assert results['accuracy'] == 1.0
        assert results['per_class']['only']['precision'] == 1.0


class TestPrintClsMetrics:
    """Tests for print_cls_metrics."""

    def test_basic_output(self, capsys):
        """Should print without errors."""
        results = compute_classification_metrics(
            y_true=['a', 'b', 'a'],
            y_pred=['a', 'a', 'a'],
            confidences=[0.9, 0.7, 0.8],
        )
        print_cls_metrics(results, branch_name='test')
        captured = capsys.readouterr()
        assert 'test' in captured.out
        assert 'Precision' in captured.out

    def test_empty_results(self, capsys):
        """Should handle empty results without error."""
        results = {
            'accuracy': 0.0,
            'per_class': {},
            'total_samples': 0,
        }
        print_cls_metrics(results)
        captured = capsys.readouterr()
        assert '0.0000' in captured.out

    def test_no_branch_name(self, capsys):
        """Should work without branch_name."""
        results = compute_classification_metrics(
            y_true=['a', 'b'],
            y_pred=['a', 'b'],
            confidences=[0.9, 0.8],
        )
        print_cls_metrics(results)
        captured = capsys.readouterr()
        assert '分类评估' in captured.out


class TestFormatClsResultsJson:
    """Tests for format_cls_results_json."""

    def test_valid_json(self):
        """Output should be valid JSON."""
        results = {'accuracy': 0.95, 'total_samples': 100}
        json_str = format_cls_results_json(results)
        parsed = json.loads(json_str)
        assert parsed['accuracy'] == 0.95

    def test_chinese_characters(self):
        """Should preserve Chinese characters."""
        results = {'class': '正常'}
        json_str = format_cls_results_json(results)
        assert '正常' in json_str

    def test_nested_structure(self):
        """Should handle nested dicts."""
        results = compute_classification_metrics(
            y_true=['a', 'b'],
            y_pred=['a', 'b'],
            confidences=[0.9, 0.8],
        )
        json_str = format_cls_results_json(results)
        parsed = json.loads(json_str)
        assert 'per_class' in parsed
        assert 'a' in parsed['per_class']
