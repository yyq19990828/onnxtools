"""Shared fixtures for contract tests.

Provides common test data (images, detections) used across API contract tests.
"""

import numpy as np
import pytest
import supervision as sv

_RNG_SEED = 42


@pytest.fixture(scope="session")
def test_image() -> np.ndarray:
    """Create a deterministic 640x640 test image.

    Returns:
        np.ndarray: BGR image with shape (640, 640, 3), dtype uint8
    """
    rng = np.random.RandomState(_RNG_SEED)
    return rng.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections() -> sv.Detections:
    """Create test detections with 5 non-overlapping objects.

    Returns:
        sv.Detections: 5 detections with confidence and class_id
    """
    xyxy = np.array(
        [
            [100, 100, 250, 200],
            [300, 150, 450, 280],
            [500, 100, 620, 220],
            [100, 350, 240, 480],
            [350, 400, 500, 550],
        ],
        dtype=np.float32,
    )

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]),
        class_id=np.array([0, 1, 0, 1, 0]),
    )
