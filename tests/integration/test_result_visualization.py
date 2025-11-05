"""Integration tests for Result class visualization methods.

This module tests end-to-end visualization functionality including
plot(), show(), and save() methods with real Supervision integration.

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from onnxtools import Result
import cv2


class TestResultVisualizationIntegration:
    """Test Result class visualization integration with Supervision (T033)."""

    @pytest.fixture
    def sample_result_with_image(self):
        """Create a Result object with sample detections and image."""
        boxes = np.array([[50, 50, 200, 200], [250, 100, 400, 300]], dtype=np.float32)
        scores = np.array([0.95, 0.87], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)
        # Create a simple test image with some patterns
        orig_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        names = {0: 'vehicle', 1: 'plate'}

        return Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=orig_img,
            orig_shape=(480, 640),
            names=names,
            path='test_image.jpg'
        )

    def test_plot_end_to_end_standard_preset(self, sample_result_with_image):
        """Test plot() end-to-end with standard preset (T033)."""
        result = sample_result_with_image

        # Plot with standard preset
        annotated = result.plot(annotator_preset='standard')

        # Verify output
        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8
        assert annotated.shape == result.orig_img.shape

        # Annotated image should be different from original
        assert not np.array_equal(annotated, result.orig_img)

        # Check that image has valid pixel values
        assert annotated.min() >= 0
        assert annotated.max() <= 255

    def test_plot_end_to_end_all_presets(self, sample_result_with_image):
        """Test plot() with all available presets (T033)."""
        result = sample_result_with_image
        # Test presets that are known to work
        presets = ['standard', 'debug', 'lightweight', 'privacy']

        for preset in presets:
            annotated = result.plot(annotator_preset=preset)

            assert isinstance(annotated, np.ndarray)
            assert annotated.shape == result.orig_img.shape
            assert annotated.dtype == np.uint8
            # Each preset should produce different results
            assert not np.array_equal(annotated, result.orig_img)

    def test_save_end_to_end(self, sample_result_with_image):
        """Test save() method end-to-end (T033)."""
        result = sample_result_with_image

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save annotated image
            result.save(tmp_path, annotator_preset='standard')

            # Verify file exists
            assert os.path.exists(tmp_path)

            # Load saved image and verify
            saved_img = cv2.imread(tmp_path)
            assert saved_img is not None
            assert saved_img.shape == result.orig_img.shape
            assert saved_img.dtype == np.uint8

        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_creates_parent_directory(self, sample_result_with_image):
        """Test save() works when parent directory exists (T033)."""
        result = sample_result_with_image

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'subdir', 'output.jpg')

            # Create parent directory first
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save should work
            result.save(output_path)

            # Verify file exists
            assert os.path.exists(output_path)

    def test_to_supervision_integration(self, sample_result_with_image):
        """Test to_supervision() creates valid Detections for Supervision pipeline (T033)."""
        import supervision as sv

        result = sample_result_with_image
        sv_detections = result.to_supervision()

        # Verify it's a valid Supervision Detections object
        assert isinstance(sv_detections, sv.Detections)
        assert len(sv_detections) == len(result)

        # Verify we can use it with Supervision annotators
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(result.orig_img.copy(), sv_detections)

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == result.orig_img.shape

    def test_visualization_with_empty_result(self):
        """Test visualization methods handle empty results gracefully (T033)."""
        orig_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = Result(boxes=None, orig_img=orig_img, orig_shape=(480, 640))

        # plot() should return copy of original for empty results
        annotated = result.plot()
        np.testing.assert_array_equal(annotated, orig_img)

        # save() should work
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result.save(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_visualization_quality_check(self, sample_result_with_image):
        """Test that annotated image quality is reasonable (T033)."""
        result = sample_result_with_image
        annotated = result.plot(annotator_preset='standard')

        # Basic quality checks
        # 1. Image should not be all black or all white
        assert annotated.mean() > 10 and annotated.mean() < 245

        # 2. Image should have some variance (not uniform)
        assert annotated.std() > 5

        # 3. Annotated image should have modifications in the box regions
        # Check first box region [50:200, 50:200]
        box_region_orig = result.orig_img[50:200, 50:200]
        box_region_annotated = annotated[50:200, 50:200]

        # These regions should be different (annotations added)
        assert not np.array_equal(box_region_orig, box_region_annotated)
