"""Contract tests for Classification Dataset Evaluator

Verifies API contracts for:
- Basic evaluation flow and return format
- CSV and ImageFolder dataset loading
- Single-branch and multi-branch evaluation
- per_sample_results structure
- BranchConfig and ClsSampleEvaluation dataclasses
"""

import json

import cv2
import numpy as np
import pytest

from onnxtools.eval.eval_cls import BranchConfig, ClsDatasetEvaluator, ClsSampleEvaluation
from onnxtools.infer_onnx.onnx_cls import ClsResult


class MockClsModel:
    """Mock classification model for contract testing.

    Simulates BaseClsORT interface returning ClsResult objects.
    """

    def __init__(self, responses: list):
        """Initialize mock with predefined responses.

        Args:
            responses: List of ClsResult objects to return sequentially
        """
        self.responses = responses
        self.call_count = 0

    def __call__(self, image: np.ndarray, **kwargs) -> ClsResult:
        if self.call_count < len(self.responses):
            result = self.responses[self.call_count]
            self.call_count += 1
            return result
        # Fallback: return unknown
        return ClsResult(labels=['unknown'], confidences=[0.1], avg_confidence=0.1)


class MockDualBranchModel:
    """Mock dual-branch classification model."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    def __call__(self, image: np.ndarray, **kwargs) -> ClsResult:
        if self.call_count < len(self.responses):
            result = self.responses[self.call_count]
            self.call_count += 1
            return result
        return ClsResult(
            labels=['unknown', 'unknown'],
            confidences=[0.1, 0.1],
            avg_confidence=0.1,
        )


@pytest.fixture
def temp_csv_dataset(tmp_path):
    """Create temporary CSV dataset with single-branch labels."""
    csv_file = tmp_path / "val.csv"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create test images
    for name in ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / name), img)

    # Write CSV
    csv_file.write_text(
        "img_name,helmet_missing\n"
        "img1.jpg,0\n"
        "img2.jpg,1\n"
        "img3.jpg,0\n"
        "img4.jpg,1\n",
        encoding="utf-8",
    )

    return csv_file, img_dir


@pytest.fixture
def temp_dual_csv_dataset(tmp_path):
    """Create temporary CSV dataset with dual-branch labels."""
    csv_file = tmp_path / "val.csv"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for name in ["p1.jpg", "p2.jpg"]:
        img = np.random.randint(0, 255, (48, 168, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / name), img)

    csv_file.write_text(
        "img_name,color,layer\n"
        "p1.jpg,1,0\n"
        "p2.jpg,4,1\n",
        encoding="utf-8",
    )

    return csv_file, img_dir


@pytest.fixture
def temp_imagefolder_dataset(tmp_path):
    """Create temporary ImageFolder dataset."""
    base = tmp_path / "dataset"
    for cls_name in ["normal", "helmet_missing"]:
        cls_dir = base / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(2):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"img{i}.jpg"), img)

    return base


@pytest.fixture
def single_branch_config():
    """BranchConfig for helmet single-branch."""
    return [
        BranchConfig(
            branch_index=0,
            column_name='helmet_missing',
            label_map={0: 'normal', 1: 'helmet_missing'},
            branch_name='helmet',
        )
    ]


@pytest.fixture
def dual_branch_config():
    """BranchConfig for color+layer dual-branch."""
    return [
        BranchConfig(0, 'color', {1: 'blue', 4: 'yellow'}, 'color'),
        BranchConfig(1, 'layer', {0: 'single', 1: 'double'}, 'layer'),
    ]


@pytest.fixture
def mock_perfect_single_model():
    """Mock model with perfect single-branch predictions."""
    return MockClsModel([
        ClsResult(labels=['normal'], confidences=[0.95], avg_confidence=0.95),
        ClsResult(labels=['helmet_missing'], confidences=[0.92], avg_confidence=0.92),
        ClsResult(labels=['normal'], confidences=[0.88], avg_confidence=0.88),
        ClsResult(labels=['helmet_missing'], confidences=[0.90], avg_confidence=0.90),
    ])


@pytest.fixture
def mock_partial_single_model():
    """Mock model with partial single-branch predictions."""
    return MockClsModel([
        ClsResult(labels=['normal'], confidences=[0.95], avg_confidence=0.95),
        ClsResult(labels=['normal'], confidences=[0.70], avg_confidence=0.70),  # Wrong
        ClsResult(labels=['normal'], confidences=[0.88], avg_confidence=0.88),
        ClsResult(labels=['helmet_missing'], confidences=[0.90], avg_confidence=0.90),
    ])


class TestResultFormatContract:
    """Contract: evaluate_dataset return format must be stable."""

    def test_required_top_level_keys(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Result must contain all required top-level keys."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        required_keys = {
            'overall_accuracy', 'total_samples', 'evaluated_samples',
            'skipped_samples', 'evaluation_time', 'avg_inference_time_ms',
            'branches', 'per_sample_results',
        }
        assert required_keys.issubset(results.keys())

    def test_numeric_ranges(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Numeric fields must be in valid ranges."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        assert 0.0 <= results['overall_accuracy'] <= 1.0
        assert results['total_samples'] >= 0
        assert results['evaluated_samples'] >= 0
        assert results['skipped_samples'] >= 0
        assert results['evaluation_time'] >= 0.0
        assert results['avg_inference_time_ms'] >= 0.0

    def test_sample_conservation(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """total_samples = evaluated_samples + skipped_samples."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        assert (results['evaluated_samples'] + results['skipped_samples']
                == results['total_samples'])


class TestBranchMetricsContract:
    """Contract: branch metrics structure must be stable."""

    def test_branch_required_keys(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Each branch must have required metric keys."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        assert 'helmet' in results['branches']
        branch = results['branches']['helmet']
        required = {'accuracy', 'per_class', 'confusion_matrix', 'class_names', 'total_samples'}
        assert required.issubset(branch.keys())

    def test_per_class_required_keys(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """per_class metrics must have required keys."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        for cls_name, cls_metrics in results['branches']['helmet']['per_class'].items():
            required = {'precision', 'recall', 'f1', 'support', 'avg_confidence'}
            assert required.issubset(cls_metrics.keys()), (
                f"Missing keys for class '{cls_name}'"
            )

    def test_perfect_accuracy(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Perfect predictions should give accuracy=1.0."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        assert results['overall_accuracy'] == 1.0
        assert results['branches']['helmet']['accuracy'] == 1.0

    def test_partial_accuracy(
        self, temp_csv_dataset, single_branch_config, mock_partial_single_model
    ):
        """Partial predictions should give correct accuracy."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_partial_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        # 3 out of 4 correct
        assert results['overall_accuracy'] == 0.75


class TestPerSampleResultsContract:
    """Contract: per_sample_results structure must be stable."""

    def test_per_sample_required_keys(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Each per-sample result must have required keys."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        assert len(results['per_sample_results']) > 0
        for sample in results['per_sample_results']:
            required = {
                'image_path', 'branch_name', 'ground_truth',
                'predicted', 'confidence', 'is_correct',
            }
            assert required.issubset(sample.keys())

    def test_per_sample_types(
        self, temp_csv_dataset, single_branch_config, mock_perfect_single_model
    ):
        """Per-sample fields must have correct types."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='table',
        )

        for sample in results['per_sample_results']:
            assert isinstance(sample['image_path'], str)
            assert isinstance(sample['branch_name'], str)
            assert isinstance(sample['ground_truth'], str)
            assert isinstance(sample['predicted'], str)
            assert isinstance(sample['confidence'], float)
            assert isinstance(sample['is_correct'], bool)


class TestDualBranchContract:
    """Contract: dual-branch evaluation must work correctly."""

    def test_dual_branch_evaluation(self, temp_dual_csv_dataset, dual_branch_config):
        """Dual-branch model should produce results for both branches."""
        csv_file, img_dir = temp_dual_csv_dataset
        model = MockDualBranchModel([
            ClsResult(labels=['blue', 'single'], confidences=[0.9, 0.85], avg_confidence=0.875),
            ClsResult(labels=['yellow', 'double'], confidences=[0.88, 0.92], avg_confidence=0.9),
        ])
        evaluator = ClsDatasetEvaluator(model)
        results = evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=dual_branch_config,
            output_format='table',
        )

        assert 'color' in results['branches']
        assert 'layer' in results['branches']
        assert results['overall_accuracy'] == 1.0


class TestCSVLoadingContract:
    """Contract: CSV loading must follow specified format."""

    def test_missing_csv_raises(self):
        """Missing CSV file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ClsDatasetEvaluator.load_csv_dataset(
                '/nonexistent.csv', '/tmp',
                [BranchConfig(0, 'label', {0: 'a'})],
            )

    def test_missing_column_raises(self, tmp_path):
        """Missing column should raise KeyError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("img_name,other_col\nimg1.jpg,0\n")

        with pytest.raises(KeyError, match="Missing columns"):
            ClsDatasetEvaluator.load_csv_dataset(
                str(csv_file), str(tmp_path),
                [BranchConfig(0, 'nonexistent', {0: 'a'})],
            )


class TestImageFolderLoadingContract:
    """Contract: ImageFolder loading must follow expected structure."""

    def test_missing_dir_raises(self):
        """Missing directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ClsDatasetEvaluator.load_imagefolder_dataset('/nonexistent_dir/')

    def test_imagefolder_loading(self, temp_imagefolder_dataset):
        """ImageFolder should load class labels from directory names."""
        dataset = ClsDatasetEvaluator.load_imagefolder_dataset(
            str(temp_imagefolder_dataset)
        )
        assert len(dataset) == 4  # 2 classes x 2 images
        labels = {s['branches']['class'] for s in dataset}
        assert labels == {'normal', 'helmet_missing'}

    def test_imagefolder_evaluation(self, temp_imagefolder_dataset):
        """ImageFolder evaluation should work end-to-end."""
        model = MockClsModel([
            ClsResult(labels=['helmet_missing'], confidences=[0.9], avg_confidence=0.9),
            ClsResult(labels=['helmet_missing'], confidences=[0.85], avg_confidence=0.85),
            ClsResult(labels=['normal'], confidences=[0.92], avg_confidence=0.92),
            ClsResult(labels=['normal'], confidences=[0.88], avg_confidence=0.88),
        ])
        evaluator = ClsDatasetEvaluator(model)
        results = evaluator.evaluate_dataset(
            dataset_dir=str(temp_imagefolder_dataset),
            output_format='table',
        )

        assert results['evaluated_samples'] == 4
        assert 'class' in results['branches']


class TestJsonOutputContract:
    """Contract: JSON output format must be valid and complete."""

    def test_json_output_valid(
        self, temp_csv_dataset, single_branch_config,
        mock_perfect_single_model, capsys
    ):
        """JSON output must be valid JSON."""
        csv_file, img_dir = temp_csv_dataset
        evaluator = ClsDatasetEvaluator(mock_perfect_single_model)
        evaluator.evaluate_dataset(
            csv_path=str(csv_file),
            image_dir=str(img_dir),
            branches=single_branch_config,
            output_format='json',
        )

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert 'overall_accuracy' in parsed
        assert 'branches' in parsed


class TestDataclassContract:
    """Contract: dataclass structures must be stable."""

    def test_branch_config_fields(self):
        """BranchConfig must have required fields."""
        bc = BranchConfig(
            branch_index=0,
            column_name='col',
            label_map={0: 'a'},
            branch_name='test',
        )
        assert bc.branch_index == 0
        assert bc.column_name == 'col'
        assert bc.label_map == {0: 'a'}
        assert bc.branch_name == 'test'

    def test_branch_config_default_name(self):
        """BranchConfig.branch_name should default to column_name."""
        bc = BranchConfig(branch_index=0, column_name='my_col', label_map={})
        assert bc.branch_name == 'my_col'

    def test_cls_sample_evaluation_fields(self):
        """ClsSampleEvaluation must have required fields."""
        sample = ClsSampleEvaluation(
            image_path='/img.jpg',
            branch_name='helmet',
            ground_truth='normal',
            predicted='normal',
            confidence=0.95,
            is_correct=True,
        )
        assert sample.image_path == '/img.jpg'
        assert sample.is_correct is True
