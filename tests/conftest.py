"""
Pytest configuration and fixtures for OCR and color classification model testing.

This module provides shared fixtures for OcrORT and ColorLayerORT testing,
including model instances, test data paths, and configuration loaders.
"""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
import yaml

# Test data paths
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
PLATES_DIR = FIXTURES_DIR / "plates"
GOLDEN_DIR = FIXTURES_DIR / "golden"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return TEST_DIR.parent


@pytest.fixture(scope="session")
def plate_config(project_root: Path) -> Dict[str, Any]:
    """
    Load plate.yaml configuration.

    Returns:
        Dict containing plate configuration including:
        - plate_dict['character']: OCR character dictionary
        - color_map: Color index to name mapping
        - layer_map: Layer index to name mapping
    """
    config_path = project_root / "configs" / "plate.yaml"
    if not config_path.exists():
        pytest.skip(f"Plate config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture(scope="session")
def ocr_character(plate_config: Dict[str, Any]) -> List[str]:
    """Extract OCR character dictionary from config."""
    return plate_config.get('ocr_dict', [])


@pytest.fixture(scope="session")
def color_map(plate_config: Dict[str, Any]) -> Dict[int, str]:
    """Extract color mapping from config."""
    return plate_config.get('color_dict', {
        0: 'blue',
        1: 'yellow',
        2: 'white',
        3: 'black',
        4: 'green'
    })


@pytest.fixture(scope="session")
def layer_map(plate_config: Dict[str, Any]) -> Dict[int, str]:
    """Extract layer mapping from config."""
    return plate_config.get('layer_dict', {
        0: 'single',
        1: 'double'
    })


@pytest.fixture(scope="session")
def ocr_model_path(project_root: Path) -> Path:
    """Return ONNX OCR model path."""
    model_path = project_root / "models" / "ocr.onnx"
    if not model_path.exists():
        pytest.skip(f"OCR model not found: {model_path}")
    return model_path


@pytest.fixture(scope="session")
def color_layer_model_path(project_root: Path) -> Path:
    """Return ONNX color/layer classification model path."""
    model_path = project_root / "models" / "color_layer.onnx"
    if not model_path.exists():
        pytest.skip(f"Color/Layer model not found: {model_path}")
    return model_path


@pytest.fixture
def sample_single_layer_plate() -> np.ndarray:
    """
    Provide a sample single-layer plate image.

    Returns:
        BGR image as numpy array [H, W, 3] uint8
    """
    # Check for real test image
    test_image_path = PLATES_DIR / "single_layer_sample.jpg"
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            return img

    # Fallback: generate synthetic test image
    return _generate_synthetic_plate(is_double=False)


@pytest.fixture
def sample_double_layer_plate() -> np.ndarray:
    """
    Provide a sample double-layer plate image.

    Returns:
        BGR image as numpy array [H, W, 3] uint8
    """
    # Check for real test image
    test_image_path = PLATES_DIR / "double_layer_sample.jpg"
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            return img

    # Fallback: generate synthetic test image
    return _generate_synthetic_plate(is_double=True)


@pytest.fixture
def sample_blue_plate() -> np.ndarray:
    """Provide a sample blue plate image."""
    test_image_path = PLATES_DIR / "blue_plate.jpg"
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            return img

    return _generate_synthetic_plate(color='blue')


@pytest.fixture
def sample_yellow_plate() -> np.ndarray:
    """Provide a sample yellow plate image."""
    test_image_path = PLATES_DIR / "yellow_plate.jpg"
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            return img

    return _generate_synthetic_plate(color='yellow')


@pytest.fixture
def golden_ocr_outputs() -> Dict[str, Any]:
    """
    Load golden OCR test outputs.

    Returns:
        Dict with structure:
        {
            'single_layer': [
                {'image': 'filename.jpg', 'text': '京A12345', 'confidence': 0.95, ...},
                ...
            ],
            'double_layer': [...]
        }
    """
    golden_path = GOLDEN_DIR / "golden_ocr_outputs.json"
    if not golden_path.exists():
        return {'single_layer': [], 'double_layer': []}

    import json
    with open(golden_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def golden_color_layer_outputs() -> Dict[str, Any]:
    """
    Load golden color/layer classification test outputs.

    Returns:
        Dict with structure:
        {
            'blue': [...],
            'yellow': [...],
            'single': [...],
            'double': [...]
        }
    """
    golden_path = GOLDEN_DIR / "golden_color_layer_outputs.json"
    if not golden_path.exists():
        return {}

    import json
    with open(golden_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Helper functions

def _generate_synthetic_plate(is_double: bool = False, color: str = 'blue') -> np.ndarray:
    """
    Generate a synthetic plate image for testing when real images are unavailable.

    Args:
        is_double: Whether to generate double-layer plate
        color: Plate color ('blue', 'yellow', etc.)

    Returns:
        Synthetic plate image [H, W, 3] uint8 BGR
    """
    if is_double:
        # Double-layer plate: 140x440
        height, width = 140, 440
    else:
        # Single-layer plate: 140x440
        height, width = 140, 440

    # Create colored background
    color_bgr = {
        'blue': (255, 100, 0),    # BGR
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (50, 50, 50),
        'green': (0, 255, 0)
    }
    bg_color = color_bgr.get(color, (255, 100, 0))

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add some text-like noise
    import random
    random.seed(42)
    for _ in range(20):
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)
        cv2.rectangle(img, (x, y), (x + 30, y + 40), (0, 0, 0), -1)

    return img


@pytest.fixture(scope="session")
def enable_gpu() -> bool:
    """
    Check if GPU is available for testing.

    Returns:
        True if CUDA GPU is available, False otherwise
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'CUDAExecutionProvider' in providers
    except Exception:
        return False
