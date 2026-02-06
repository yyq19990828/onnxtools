"""Shared fixtures for contract tests.

Provides common test data (images, detections) used across API contract tests.
"""

import numpy as np
import pytest
import supervision as sv


@pytest.fixture
def test_image():
    """Create a 640x640 test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create test detections with 5 objects."""
    xyxy = np.array(
        [[100, 100, 200, 200], [250, 150, 350, 250], [400, 100, 500, 200], [100, 350, 200, 450], [300, 400, 400, 500]],
        dtype=np.float32,
    )

    return sv.Detections(
        xyxy=xyxy, confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]), class_id=np.array([0, 1, 0, 1, 0])
    )
