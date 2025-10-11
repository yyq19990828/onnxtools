"""Unit tests for label file loading with JSON array support

Tests the load_label_file function's ability to handle:
1. Standard single-image format: image_path<TAB>ground_truth
2. JSON array format: ["img1.jpg", "img2.jpg"]<TAB>ground_truth
3. Mixed formats in the same file
4. Error handling for invalid JSON and missing files
"""

import pytest
import tempfile
from pathlib import Path
from infer_onnx.eval_ocr import load_label_file


class TestLoadLabelFile:
    """Test suite for label file loading functionality"""

    def test_single_image_format(self):
        """Test standard single-image per line format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image file
            img = tmpdir / "test_img.jpg"
            img.touch()

            # Create label file
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('test_img.jpg\t川A12345\n')

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 1
            assert dataset[0][1] == '川A12345'
            assert 'test_img.jpg' in dataset[0][0]

    def test_json_array_format(self):
        """Test JSON array format with multiple images per label"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image files
            img1 = tmpdir / "img1.jpg"
            img2 = tmpdir / "img2.jpg"
            img3 = tmpdir / "img3.jpg"
            img1.touch()
            img2.touch()
            img3.touch()

            # Create label file with JSON array
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('["img1.jpg", "img2.jpg", "img3.jpg"]\t京B67890\n')

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 3
            assert all(label == '京B67890' for _, label in dataset)
            assert 'img1.jpg' in dataset[0][0]
            assert 'img2.jpg' in dataset[1][0]
            assert 'img3.jpg' in dataset[2][0]

    def test_mixed_format(self):
        """Test mixed single-image and JSON array formats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image files
            img1 = tmpdir / "img1.jpg"
            img2 = tmpdir / "img2.jpg"
            img3 = tmpdir / "single.jpg"
            img1.touch()
            img2.touch()
            img3.touch()

            # Create label file with mixed formats
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                # JSON array format
                f.write('["img1.jpg", "img2.jpg"]\t川A12345\n')
                # Single image format
                f.write('single.jpg\t京B67890\n')

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 3
            assert dataset[0][1] == '川A12345'
            assert dataset[1][1] == '川A12345'
            assert dataset[2][1] == '京B67890'

    def test_empty_lines_and_whitespace(self):
        """Test handling of empty lines and whitespace"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image file
            img = tmpdir / "test.jpg"
            img.touch()

            # Create label file with empty lines and whitespace
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n')  # Empty line
                f.write('test.jpg\t川A12345\n')
                f.write('   \n')  # Whitespace only
                f.write('\n')  # Empty line

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 1
            assert dataset[0][1] == '川A12345'

    def test_missing_image_files(self):
        """Test handling of missing image files (should skip with warning)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create only one of two referenced images
            existing_img = tmpdir / "exists.jpg"
            existing_img.touch()
            # missing_img.jpg does not exist

            # Create label file referencing both images
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('["exists.jpg", "missing.jpg"]\t川A12345\n')

            # Load and verify (should only include existing image)
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 1
            assert 'exists.jpg' in dataset[0][0]

    def test_invalid_json_format(self):
        """Test handling of invalid JSON format (should skip with warning)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create label file with invalid JSON
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                # Invalid JSON (missing quotes)
                f.write('[img1.jpg, img2.jpg]\t川A12345\n')

            # Load and verify (should skip invalid line)
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 0

    def test_invalid_tab_separated_format(self):
        """Test handling of invalid tab-separated format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create label file with invalid format
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                # Missing tab separator
                f.write('img1.jpg川A12345\n')
                # Too many tabs
                f.write('img2.jpg\t川B67890\textra_field\n')

            # Load and verify (should skip invalid lines)
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 0

    def test_chinese_characters_in_labels(self):
        """Test handling of Chinese characters in ground truth labels"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image file
            img = tmpdir / "test.jpg"
            img.touch()

            # Create label file with various Chinese province abbreviations
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('test.jpg\t川A12345警\n')

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 1
            assert dataset[0][1] == '川A12345警'

    def test_label_file_not_found(self):
        """Test handling of non-existent label file"""
        with pytest.raises(FileNotFoundError):
            load_label_file('nonexistent_file.txt', '/tmp')

    def test_relative_paths_in_json_array(self):
        """Test relative paths with subdirectories in JSON array"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create subdirectory structure
            subdir = tmpdir / "plate" / "single" / "川"
            subdir.mkdir(parents=True)

            # Create test image files in subdirectory
            img1 = subdir / "川A12345_1.jpg"
            img2 = subdir / "川A12345_2.jpg"
            img1.touch()
            img2.touch()

            # Create label file with relative paths
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('["plate/single/川/川A12345_1.jpg", "plate/single/川/川A12345_2.jpg"]\t川A12345\n')

            # Load and verify
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 2
            assert all(label == '川A12345' for _, label in dataset)
            assert '川A12345_1.jpg' in dataset[0][0]
            assert '川A12345_2.jpg' in dataset[1][0]

    def test_empty_json_array(self):
        """Test handling of empty JSON array"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create label file with empty JSON array
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('[]\t川A12345\n')

            # Load and verify (should result in no entries)
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 0

    def test_json_array_with_non_list_value(self):
        """Test handling of JSON that is not a list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create label file with JSON object instead of array
            label_file = tmpdir / "labels.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('{"image": "img.jpg"}\t川A12345\n')

            # Load and verify (should skip invalid format)
            dataset = load_label_file(str(label_file), str(tmpdir))

            assert len(dataset) == 0
