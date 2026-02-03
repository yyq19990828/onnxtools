"""Performance benchmarks for Result.plot() method.

This module provides pytest-benchmark tests for Result.plot() performance,
measuring end-to-end rendering time with different presets and object counts.

Note: If pytest-benchmark is not installed, tests will measure time directly
and verify performance is within acceptable range (<50ms for 20 objects).

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
"""

import time

import numpy as np
import pytest

from onnxtools import Result

# Check if pytest-benchmark is available
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.fixture
def test_image():
    """Generate a 640x640 test image.

    Returns:
        np.ndarray: Random RGB image (640, 640, 3)
    """
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def result_with_20_detections(test_image):
    """Create Result object with 20 detections for benchmarking.

    Returns:
        Result: Result object with 20 detections
    """
    # Generate random bounding boxes
    boxes = np.random.rand(20, 4) * 640
    boxes = boxes.astype(np.float32)

    # Ensure valid boxes (x1 < x2, y1 < y2)
    boxes[:, 2] = np.maximum(boxes[:, 0] + 10, boxes[:, 2])
    boxes[:, 3] = np.maximum(boxes[:, 1] + 10, boxes[:, 3])

    scores = np.random.rand(20).astype(np.float32)
    class_ids = np.random.randint(0, 2, 20, dtype=np.int32)
    names = {0: 'vehicle', 1: 'plate'}

    return Result(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        orig_img=test_image,
        orig_shape=(640, 640),
        names=names
    )


@pytest.fixture
def result_with_5_detections(test_image):
    """Create Result object with 5 detections for benchmarking.

    Returns:
        Result: Result object with 5 detections
    """
    boxes = np.array([
        [50, 50, 200, 200],
        [250, 100, 400, 300],
        [100, 350, 250, 500],
        [300, 400, 450, 600],
        [500, 50, 620, 150]
    ], dtype=np.float32)

    scores = np.array([0.95, 0.87, 0.92, 0.78, 0.89], dtype=np.float32)
    class_ids = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    names = {0: 'vehicle', 1: 'plate'}

    return Result(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        orig_img=test_image,
        orig_shape=(640, 640),
        names=names
    )


# Simplified performance tests without pytest-benchmark
# These tests measure time directly and verify performance constraints

def test_plot_performance_standard_preset_20objects(result_with_20_detections):
    """Test plot() performance with standard preset and 20 objects (T035).

    Performance target: < 50ms for 20 objects on 640x640 image
    """
    # Warmup
    _ = result_with_20_detections.plot(annotator_preset='standard')

    # Measure time for 10 iterations
    times = []
    for _ in range(10):
        start = time.perf_counter()
        annotated = result_with_20_detections.plot(annotator_preset='standard')
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8

    avg_time = np.mean(times)
    print(f"\nAverage plot() time (standard, 20 objects): {avg_time:.2f}ms")

    # Performance assertion: Should be < 50ms on average
    assert avg_time < 50, f"plot() too slow: {avg_time:.2f}ms (target: <50ms)"


def test_plot_performance_debug_preset_20objects(result_with_20_detections):
    """Test plot() performance with debug preset and 20 objects (T035)."""
    # Warmup
    _ = result_with_20_detections.plot(annotator_preset='debug')

    # Measure time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        annotated = result_with_20_detections.plot(annotator_preset='debug')
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8

    avg_time = np.mean(times)
    print(f"\nAverage plot() time (debug, 20 objects): {avg_time:.2f}ms")
    assert avg_time < 100, f"plot() too slow: {avg_time:.2f}ms (target: <100ms)"


def test_plot_performance_lightweight_preset(result_with_20_detections):
    """Test plot() performance with lightweight preset (T035)."""
    # Warmup
    _ = result_with_20_detections.plot(annotator_preset='lightweight')

    # Measure time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        annotated = result_with_20_detections.plot(annotator_preset='lightweight')
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"\nAverage plot() time (lightweight, 20 objects): {avg_time:.2f}ms")
    # Lightweight should be fastest
    assert avg_time < 40, f"plot() too slow: {avg_time:.2f}ms (target: <40ms)"


def test_plot_performance_5objects(result_with_5_detections):
    """Test plot() performance with 5 objects (T035)."""
    # Warmup
    _ = result_with_5_detections.plot(annotator_preset='standard')

    # Measure time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        annotated = result_with_5_detections.plot(annotator_preset='standard')
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"\nAverage plot() time (standard, 5 objects): {avg_time:.2f}ms")
    # Fewer objects should be faster
    assert avg_time < 30, f"plot() too slow: {avg_time:.2f}ms (target: <30ms)"


def test_plot_performance_empty_result(test_image):
    """Test plot() performance with empty Result (T035)."""
    result = Result(boxes=None, orig_img=test_image, orig_shape=(640, 640))

    # Warmup
    _ = result.plot(annotator_preset='standard')

    # Measure time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        annotated = result.plot(annotator_preset='standard')
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"\nAverage plot() time (empty result): {avg_time:.2f}ms")

    # Empty result should be very fast (just array copy)
    assert avg_time < 5, f"plot() too slow for empty result: {avg_time:.2f}ms (target: <5ms)"

    # Verify returns copy of original
    np.testing.assert_array_equal(annotated, test_image)


def test_plot_scalability(test_image):
    """Test plot() scalability with varying object counts (T035)."""
    object_counts = [1, 5, 10, 20, 50]
    times_by_count = {}

    for num_objects in object_counts:
        # Generate detections
        boxes = np.random.rand(num_objects, 4) * 640
        boxes = boxes.astype(np.float32)
        boxes[:, 2] = np.maximum(boxes[:, 0] + 10, boxes[:, 2])
        boxes[:, 3] = np.maximum(boxes[:, 1] + 10, boxes[:, 3])

        scores = np.random.rand(num_objects).astype(np.float32)
        class_ids = np.random.randint(0, 2, num_objects, dtype=np.int32)

        result = Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=test_image,
            orig_shape=(640, 640),
            names={0: 'vehicle', 1: 'plate'}
        )

        # Warmup
        _ = result.plot(annotator_preset='standard')

        # Measure time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            annotated = result.plot(annotator_preset='standard')
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        times_by_count[num_objects] = avg_time
        print(f"\nAverage plot() time ({num_objects} objects): {avg_time:.2f}ms")

    # Verify scalability: time should increase roughly linearly
    # 50 objects should be < 5x time of 10 objects
    time_ratio = times_by_count[50] / times_by_count[10]
    assert time_ratio < 5.0, f"Poor scalability: 50 objects is {time_ratio:.1f}x slower than 10 objects"
