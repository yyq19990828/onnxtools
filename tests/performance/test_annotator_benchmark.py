"""
Performance benchmarks for supervision annotators.

This module provides pytest-benchmark tests for all 13 annotator types,
measuring rendering performance on 640x640 images with 20 detection objects.
"""

import numpy as np
import pytest
import supervision as sv

from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType

# Skip all tests in this module if pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")


# Test fixtures
@pytest.fixture
def test_image():
    """
    Generate a 640x640 test image.

    Returns:
        np.ndarray: Random RGB image (640, 640, 3)
    """
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """
    Generate 20 random detection objects.

    Returns:
        sv.Detections: Detections with bbox, confidence, class_id
    """
    # Generate random bounding boxes
    xyxy = np.random.rand(20, 4) * 640

    # Ensure valid boxes (x1 < x2, y1 < y2)
    xyxy[:, 2] = np.maximum(xyxy[:, 0] + 10, xyxy[:, 2])
    xyxy[:, 3] = np.maximum(xyxy[:, 1] + 10, xyxy[:, 3])

    return sv.Detections(xyxy=xyxy, confidence=np.random.rand(20), class_id=np.random.randint(0, 2, 20))


# Helper function for benchmark tests
def create_and_annotate(annotator_type, config, test_image, test_detections):
    """
    Create annotator and apply annotation for benchmarking.

    Args:
        annotator_type: AnnotatorType enum
        config: Configuration dict
        test_image: Test image
        test_detections: Test detections

    Returns:
        Annotated image
    """
    annotator = AnnotatorFactory.create(annotator_type, config)
    return annotator.annotate(test_image.copy(), test_detections)


# Border annotators benchmarks
@pytest.mark.benchmark(group="border_annotators")
def test_box_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark BoxAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.BOX, {"thickness": 2}, test_image, test_detections)
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="border_annotators")
def test_round_box_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark RoundBoxAnnotator rendering time."""
    result = benchmark(
        create_and_annotate, AnnotatorType.ROUND_BOX, {"thickness": 2, "roundness": 0.3}, test_image, test_detections
    )
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="border_annotators")
def test_box_corner_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark BoxCornerAnnotator rendering time."""
    result = benchmark(
        create_and_annotate,
        AnnotatorType.BOX_CORNER,
        {"thickness": 2, "corner_length": 20},
        test_image,
        test_detections,
    )
    assert result.shape == test_image.shape


# Geometric markers benchmarks
@pytest.mark.benchmark(group="geometric_markers")
def test_circle_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark CircleAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.CIRCLE, {"thickness": 2}, test_image, test_detections)
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="geometric_markers")
def test_triangle_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark TriangleAnnotator rendering time."""
    result = benchmark(
        create_and_annotate, AnnotatorType.TRIANGLE, {"base": 20, "height": 20}, test_image, test_detections
    )
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="geometric_markers")
def test_ellipse_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark EllipseAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.ELLIPSE, {"thickness": 2}, test_image, test_detections)
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="geometric_markers")
def test_dot_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark DotAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.DOT, {"radius": 5}, test_image, test_detections)
    assert result.shape == test_image.shape


# Fill annotators benchmarks
@pytest.mark.benchmark(group="fill_annotators")
def test_color_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark ColorAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.COLOR, {"opacity": 0.3}, test_image, test_detections)
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="fill_annotators")
def test_background_overlay_performance(benchmark, test_image, test_detections):
    """Benchmark BackgroundOverlayAnnotator rendering time."""
    result = benchmark(
        create_and_annotate, AnnotatorType.BACKGROUND_OVERLAY, {"opacity": 0.5}, test_image, test_detections
    )
    assert result.shape == test_image.shape


# Effect annotators benchmarks
@pytest.mark.benchmark(group="effect_annotators")
def test_halo_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark HaloAnnotator rendering time."""
    result = benchmark(
        create_and_annotate, AnnotatorType.HALO, {"opacity": 0.3, "kernel_size": 40}, test_image, test_detections
    )
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="effect_annotators")
def test_percentage_bar_performance(benchmark, test_image, test_detections):
    """Benchmark PercentageBarAnnotator rendering time."""
    result = benchmark(
        create_and_annotate, AnnotatorType.PERCENTAGE_BAR, {"height": 16, "width": 80}, test_image, test_detections
    )
    assert result.shape == test_image.shape


# Privacy protection benchmarks
@pytest.mark.benchmark(group="privacy_annotators")
def test_blur_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark BlurAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.BLUR, {"kernel_size": 15}, test_image, test_detections)
    assert result.shape == test_image.shape


@pytest.mark.benchmark(group="privacy_annotators")
@pytest.mark.skip(reason="PixelateAnnotator has issue with small ROIs in supervision library")
def test_pixelate_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark PixelateAnnotator rendering time."""
    result = benchmark(create_and_annotate, AnnotatorType.PIXELATE, {"pixel_size": 20}, test_image, test_detections)
    assert result.shape == test_image.shape
