"""Shared fixtures for integration tests.

Provides common test data (images, detections) used across annotator
and visualization integration tests.
"""

import numpy as np
import pytest
import supervision as sv


@pytest.fixture
def test_image():
    """Create 640x640 test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create test detections with 5 objects."""
    xyxy = np.array(
        [[100, 100, 250, 200], [300, 150, 450, 280], [500, 100, 620, 220], [100, 350, 240, 480], [350, 400, 500, 550]],
        dtype=np.float32,
    )

    return sv.Detections(
        xyxy=xyxy, confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]), class_id=np.array([0, 1, 0, 1, 0])
    )
