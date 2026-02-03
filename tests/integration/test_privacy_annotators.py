"""Integration tests for privacy protection annotators (Blur, Pixelate)."""

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


class TestPrivacyAnnotators:
    """Integration tests for BlurAnnotator and PixelateAnnotator."""

    def test_blur_annotator_basic(self, test_image, test_detections):
        """Test basic blur effect."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.BLUR,
            {'kernel_size': 15}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_blur_different_kernel_sizes(self, test_image, test_detections):
        """Test different blur kernel sizes."""
        for kernel_size in [5, 15, 25, 35]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.BLUR,
                {'kernel_size': kernel_size}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_pixelate_annotator_basic(self, test_image, test_detections):
        """Test basic pixelate effect."""
        annotator = AnnotatorFactory.create(
            AnnotatorType.PIXELATE,
            {'pixel_size': 20}
        )
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_pixelate_different_pixel_sizes(self, test_image, test_detections):
        """Test different pixel sizes."""
        for pixel_size in [10, 20, 30, 40]:
            annotator = AnnotatorFactory.create(
                AnnotatorType.PIXELATE,
                {'pixel_size': pixel_size}
            )
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_blur_single_detection(self, test_image):
        """Test blur with single detection (e.g., license plate)."""
        plate_detection = sv.Detections(
            xyxy=np.array([[200, 300, 400, 350]], dtype=np.float32),
            confidence=np.array([0.95]),
            class_id=np.array([1])  # class_id=1 for plate
        )
        annotator = AnnotatorFactory.create(
            AnnotatorType.BLUR,
            {'kernel_size': 15}
        )
        result = annotator.annotate(test_image, plate_detection)
        assert result.shape == test_image.shape

    def test_pixelate_for_privacy(self, test_image):
        """Test pixelate for privacy protection."""
        sensitive_detection = sv.Detections(
            xyxy=np.array([[150, 250, 350, 400]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([1])
        )
        annotator = AnnotatorFactory.create(
            AnnotatorType.PIXELATE,
            {'pixel_size': 25}
        )
        result = annotator.annotate(test_image, sensitive_detection)
        assert result.shape == test_image.shape
