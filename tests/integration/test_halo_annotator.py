"""Integration test for HaloAnnotator."""

import numpy as np
import pytest
import supervision as sv

from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType


@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    xyxy = np.array([[100, 100, 250, 200], [300, 150, 450, 280]], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87]),
        class_id=np.array([0, 1])
    )


class TestHaloAnnotatorIntegration:
    """Integration tests for HaloAnnotator."""

    def test_halo_basic_rendering(self, test_image, test_detections):
        """Test basic halo effect rendering."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.HALO,
            {'opacity': 0.3, 'kernel_size': 40}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_halo_different_kernel_sizes(self, test_image, test_detections):
        """Test different kernel sizes for halo effect."""
        for kernel_size in [20, 40, 60, 80]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.HALO,
                {'opacity': 0.3, 'kernel_size': kernel_size}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_halo_different_opacity(self, test_image, test_detections):
        """Test different opacity values."""
        for opacity in [0.1, 0.3, 0.5, 0.7]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.HALO,
                {'opacity': opacity, 'kernel_size': 40}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_halo_with_other_annotators(self, test_image, test_detections):
        """Test halo combined with box annotator."""
        from utils.annotator_factory import AnnotatorPipeline

        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.HALO, {'opacity': 0.3, 'kernel_size': 40})
                    .add(AnnotatorType.BOX, {'thickness': 2}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
