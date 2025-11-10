"""Unit tests for Result.crop() and Result.save_crop() methods.

Tests cover:
- Basic cropping functionality
- Confidence filtering
- Class filtering
- Combined filtering
- Bounding box expansion (gain, pad, square)
- Empty results handling
- Error handling (missing orig_img, invalid parameters)
- Directory structure options
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

from onnxtools.infer_onnx.result import Result, _sanitize_filename


class TestSanitizeFilename:
    """Test _sanitize_filename() helper function."""

    def test_sanitize_basic(self):
        """Test basic filename sanitization."""
        assert _sanitize_filename("vehicle") == "vehicle"
        assert _sanitize_filename("class_0") == "class_0"
        assert _sanitize_filename("vehicle-plate") == "vehicle-plate"
        assert _sanitize_filename("test 123") == "test123"  # Spaces removed

    def test_sanitize_illegal_chars(self):
        """Test sanitization of illegal characters."""
        assert _sanitize_filename("vehicle/plate") == "vehicle_plate"
        assert _sanitize_filename("class:0") == "class_0"
        assert _sanitize_filename("test\\file") == "test_file"
        assert _sanitize_filename("a*b?c|d") == "a_b_c_d"


class TestResultFilter:
    """Test Result.filter() method with string class names."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample Result object with test data."""
        orig_img = np.zeros((100, 200, 3), dtype=np.uint8)

        boxes = np.array([
            [10, 10, 50, 50],    # vehicle (class 0)
            [60, 60, 90, 90],    # plate (class 1)
            [120, 20, 180, 80],  # vehicle (class 0)
        ], dtype=np.float32)

        scores = np.array([0.95, 0.85, 0.75], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)

        names = {0: 'vehicle', 1: 'plate'}

        return Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(100, 200),
            names=names
        )

    def test_filter_by_class_name(self, sample_result):
        """Test filtering by class name."""
        filtered = sample_result.filter(classes=['vehicle'])

        assert len(filtered) == 2
        assert all(filtered.class_ids == 0)

    def test_filter_by_multiple_class_names(self, sample_result):
        """Test filtering by multiple class names."""
        filtered = sample_result.filter(classes=['vehicle', 'plate'])

        assert len(filtered) == 3

    def test_filter_by_mixed_id_and_name(self, sample_result):
        """Test filtering by mixed IDs and names."""
        filtered = sample_result.filter(classes=[0, 'plate'])

        assert len(filtered) == 3

    def test_filter_invalid_class_name(self, sample_result):
        """Test error with invalid class name."""
        with pytest.raises(ValueError, match="Class name 'invalid' not found"):
            sample_result.filter(classes=['invalid'])

    def test_filter_class_name_with_conf(self, sample_result):
        """Test filtering by class name and confidence."""
        filtered = sample_result.filter(conf_threshold=0.8, classes=['vehicle'])

        assert len(filtered) == 1  # Only first vehicle with 0.95
        assert filtered.class_ids[0] == 0
        assert filtered.scores[0] >= 0.8


class TestResultCrop:
    """Test Result.crop() method."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample Result object with test data."""
        # Create a simple test image (100x200 BGR)
        orig_img = np.zeros((100, 200, 3), dtype=np.uint8)
        orig_img[:, :, 2] = 255  # Red channel

        boxes = np.array([
            [10, 10, 50, 50],    # vehicle (class 0)
            [60, 60, 90, 90],    # plate (class 1)
            [120, 20, 180, 80],  # vehicle (class 0)
        ], dtype=np.float32)

        scores = np.array([0.95, 0.85, 0.75], dtype=np.float32)
        class_ids = np.array([0, 1, 0], dtype=np.int32)

        names = {0: 'vehicle', 1: 'plate'}

        return Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(100, 200),
            names=names
        )

    def test_crop_basic(self, sample_result):
        """Test basic cropping without filters."""
        crops = sample_result.crop()

        assert len(crops) == 3
        assert all('image' in c for c in crops)
        assert all('box' in c for c in crops)
        assert all('crop_box' in c for c in crops)
        assert all('class_id' in c for c in crops)
        assert all('class_name' in c for c in crops)
        assert all('confidence' in c for c in crops)
        assert all('index' in c for c in crops)

        # Check first crop
        crop0 = crops[0]
        assert crop0['class_id'] == 0
        assert crop0['class_name'] == 'vehicle'
        assert pytest.approx(crop0['confidence'], abs=1e-6) == 0.95
        assert crop0['index'] == 0
        assert crop0['image'].shape[2] == 3  # BGR

    def test_crop_conf_filter(self, sample_result):
        """Test cropping with confidence threshold."""
        crops = sample_result.crop(conf_threshold=0.8)

        assert len(crops) == 2  # Only 0.95 and 0.85
        assert crops[0]['confidence'] >= 0.8
        assert crops[1]['confidence'] >= 0.8

    def test_crop_class_filter(self, sample_result):
        """Test cropping with class ID filter."""
        crops = sample_result.crop(classes=[0])  # Only vehicles

        assert len(crops) == 2
        assert all(c['class_id'] == 0 for c in crops)
        assert all(c['class_name'] == 'vehicle' for c in crops)

    def test_crop_class_name_filter(self, sample_result):
        """Test cropping with class name filter."""
        crops = sample_result.crop(classes=['plate'])  # Only plates

        assert len(crops) == 1
        assert crops[0]['class_id'] == 1
        assert crops[0]['class_name'] == 'plate'

    def test_crop_mixed_class_filter(self, sample_result):
        """Test cropping with mixed ID and name filter."""
        crops = sample_result.crop(classes=[0, 'plate'])  # vehicles + plates

        assert len(crops) == 3
        class_ids = {c['class_id'] for c in crops}
        assert class_ids == {0, 1}

    def test_crop_invalid_class_name(self, sample_result):
        """Test error with invalid class name."""
        with pytest.raises(ValueError, match="Class name 'invalid' not found"):
            sample_result.crop(classes=['invalid'])

    def test_crop_combined_filter(self, sample_result):
        """Test cropping with both confidence and class filters."""
        crops = sample_result.crop(conf_threshold=0.9, classes=[0])

        assert len(crops) == 1  # Only first vehicle with 0.95 conf
        assert crops[0]['confidence'] >= 0.9
        assert crops[0]['class_id'] == 0

    def test_crop_with_gain(self, sample_result):
        """Test cropping with gain expansion."""
        crops_default = sample_result.crop(gain=1.0, pad=0)
        crops_expanded = sample_result.crop(gain=1.2, pad=0)

        # Expanded crop should have larger dimensions
        crop0_default = crops_default[0]
        crop0_expanded = crops_expanded[0]

        w_default = crop0_default['crop_box'][2] - crop0_default['crop_box'][0]
        w_expanded = crop0_expanded['crop_box'][2] - crop0_expanded['crop_box'][0]

        assert w_expanded > w_default

    def test_crop_with_pad(self, sample_result):
        """Test cropping with padding."""
        crops_no_pad = sample_result.crop(gain=1.0, pad=0)
        crops_with_pad = sample_result.crop(gain=1.0, pad=10)

        # Padded crop should have larger dimensions
        crop0_no_pad = crops_no_pad[0]
        crop0_with_pad = crops_with_pad[0]

        w_no_pad = crop0_no_pad['crop_box'][2] - crop0_no_pad['crop_box'][0]
        w_with_pad = crop0_with_pad['crop_box'][2] - crop0_with_pad['crop_box'][0]

        assert w_with_pad > w_no_pad

    def test_crop_square(self, sample_result):
        """Test square cropping."""
        crops = sample_result.crop(square=True, gain=1.0, pad=0)

        for crop in crops:
            crop_box = crop['crop_box']
            w = crop_box[2] - crop_box[0]
            h = crop_box[3] - crop_box[1]
            # Should be approximately square (allowing for rounding)
            assert abs(w - h) <= 1

    def test_crop_empty_result(self):
        """Test cropping on empty result."""
        result = Result(
            boxes=None,
            scores=None,
            class_ids=None,
            orig_img=np.zeros((100, 100, 3), dtype=np.uint8),
            orig_shape=(100, 100)
        )

        crops = result.crop()
        assert crops == []

    def test_crop_no_orig_img(self):
        """Test error when orig_img is None."""
        result = Result(
            boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
            orig_img=None,  # Missing image
            orig_shape=(100, 100)
        )

        with pytest.raises(ValueError, match="Cannot crop detections: orig_img is None"):
            result.crop()

    def test_crop_invalid_gain(self, sample_result):
        """Test error with invalid gain parameter."""
        with pytest.raises(ValueError, match="gain must be positive"):
            sample_result.crop(gain=0)

        with pytest.raises(ValueError, match="gain must be positive"):
            sample_result.crop(gain=-1.5)

    def test_crop_invalid_pad(self, sample_result):
        """Test error with invalid pad parameter."""
        with pytest.raises(ValueError, match="pad must be non-negative"):
            sample_result.crop(pad=-10)

    def test_crop_boundary_clipping(self):
        """Test that crops are clipped to image boundaries."""
        # Create image with box near edge
        orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[90, 90, 110, 110]], dtype=np.float32)  # Extends beyond image
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(100, 100)
        )

        crops = result.crop()
        assert len(crops) == 1

        crop_box = crops[0]['crop_box']
        # Should be clipped to [0, 100]
        assert crop_box[0] >= 0
        assert crop_box[1] >= 0
        assert crop_box[2] <= 100
        assert crop_box[3] <= 100


class TestResultSaveCrop:
    """Test Result.save_crop() method."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample Result object with test data."""
        orig_img = np.zeros((100, 200, 3), dtype=np.uint8)
        orig_img[:, :, 2] = 255  # Red channel

        boxes = np.array([
            [10, 10, 50, 50],    # vehicle (class 0)
            [60, 60, 90, 90],    # plate (class 1)
        ], dtype=np.float32)

        scores = np.array([0.95, 0.85], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)

        names = {0: 'vehicle', 1: 'plate'}

        return Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(100, 200),
            names=names
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_save_crop_basic(self, sample_result, temp_dir):
        """Test basic save_crop functionality."""
        saved_paths = sample_result.save_crop(temp_dir)

        assert len(saved_paths) == 2
        assert all(isinstance(p, Path) for p in saved_paths)
        assert all(p.exists() for p in saved_paths)

        # Check directory structure: crops/vehicle/im_0.jpg, crops/plate/im_1.jpg
        assert (temp_dir / 'vehicle').exists()
        assert (temp_dir / 'plate').exists()
        assert (temp_dir / 'vehicle' / 'im_0.jpg').exists()
        assert (temp_dir / 'plate' / 'im_1.jpg').exists()

    def test_save_crop_no_class_dirs(self, sample_result, temp_dir):
        """Test save_crop without class subdirectories."""
        saved_paths = sample_result.save_crop(temp_dir, save_class_dirs=False)

        assert len(saved_paths) == 2
        # Files should be directly in temp_dir
        assert (temp_dir / 'im_0.jpg').exists()
        assert (temp_dir / 'im_1.jpg').exists()

    def test_save_crop_custom_filename(self, sample_result, temp_dir):
        """Test save_crop with custom filename."""
        saved_paths = sample_result.save_crop(temp_dir, file_name='crop.png')

        assert len(saved_paths) == 2
        assert (temp_dir / 'vehicle' / 'crop_0.png').exists()
        assert (temp_dir / 'plate' / 'crop_1.png').exists()

    def test_save_crop_with_filter(self, sample_result, temp_dir):
        """Test save_crop with confidence filter."""
        saved_paths = sample_result.save_crop(temp_dir, conf_threshold=0.9)

        assert len(saved_paths) == 1  # Only vehicle with 0.95
        assert (temp_dir / 'vehicle' / 'im_0.jpg').exists()

    def test_save_crop_empty_result(self, temp_dir):
        """Test save_crop on empty result."""
        result = Result(
            boxes=None,
            scores=None,
            class_ids=None,
            orig_img=np.zeros((100, 100, 3), dtype=np.uint8),
            orig_shape=(100, 100)
        )

        saved_paths = result.save_crop(temp_dir)
        assert saved_paths == []

    def test_save_crop_creates_directory(self, sample_result):
        """Test that save_crop creates non-existent directory."""
        with tempfile.TemporaryDirectory() as base_dir:
            save_dir = Path(base_dir) / 'non_existent' / 'crops'
            saved_paths = sample_result.save_crop(save_dir)

            assert len(saved_paths) == 2
            assert save_dir.exists()
            assert all(p.exists() for p in saved_paths)

    def test_save_crop_image_quality(self, sample_result, temp_dir):
        """Test that saved images are readable and correct."""
        saved_paths = sample_result.save_crop(temp_dir)

        for path in saved_paths:
            # Read saved image
            img = cv2.imread(str(path))
            assert img is not None
            assert img.shape[2] == 3  # BGR
            assert img.dtype == np.uint8

    def test_save_crop_no_orig_img(self, temp_dir):
        """Test error when orig_img is None."""
        result = Result(
            boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
            orig_img=None,
            orig_shape=(100, 100)
        )

        with pytest.raises(ValueError, match="Cannot crop detections: orig_img is None"):
            result.save_crop(temp_dir)

    def test_save_crop_illegal_class_names(self, temp_dir):
        """Test sanitization of illegal characters in class names."""
        orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        names = {0: 'vehicle/plate:test'}  # Illegal characters

        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(100, 100),
            names=names
        )

        saved_paths = result.save_crop(temp_dir)
        assert len(saved_paths) == 1
        # Should sanitize to 'vehicle_plate_test'
        assert (temp_dir / 'vehicle_plate_test').exists()
