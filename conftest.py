"""Global pytest configuration and fixtures for supervision integration tests."""

import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :] = [100, 100, 100]  # Gray background
    return image

@pytest.fixture
def sample_detections():
    """Create sample detection data in current format."""
    return [
        [
            [100.0, 100.0, 200.0, 150.0, 0.95, 0],  # vehicle
            [150.0, 120.0, 180.0, 135.0, 0.88, 1],  # plate
        ]
    ]

@pytest.fixture
def sample_plate_results():
    """Create sample OCR results for plates."""
    return [
        None,  # vehicle (no OCR)
        {
            "plate_text": "京A12345",
            "color": "蓝色",
            "layer": "单层",
            "should_display_ocr": True
        }  # plate OCR
    ]

@pytest.fixture
def sample_class_names():
    """Sample class names for detection."""
    return {0: "vehicle", 1: "plate"}

@pytest.fixture
def sample_colors():
    """Sample colors for drawing."""
    return [(255, 0, 0), (0, 255, 0)]  # Red for vehicle, Green for plate

@pytest.fixture
def font_path():
    """Font path for testing."""
    candidates = [
        "SourceHanSans-VF.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/simhei.ttf"
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None

@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "target_time_ms": 30.0,  # Target: <30ms for 20 objects (more realistic for development)
        "iterations": 100,
        "max_objects": 20
    }
