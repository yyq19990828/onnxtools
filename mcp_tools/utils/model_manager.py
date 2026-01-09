"""
Model management utilities for MCP tools.

Provides lazy loading and caching of ONNX models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, MAX_CACHED_MODELS

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Dict[str, Any] = {}


def get_detector(
    model_path: str,
    model_type: str,
    conf_thres: float = DEFAULT_CONF_THRESHOLD,
    iou_thres: float = DEFAULT_IOU_THRESHOLD,
) -> "BaseORT":
    """Get or create detector instance with lazy loading.

    Args:
        model_path: Path to ONNX model file
        model_type: Model architecture type (yolo, rtdetr, rfdetr)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS

    Returns:
        BaseORT detector instance

    Raises:
        FileNotFoundError: If model file does not exist
        ValueError: If model type is not supported
    """
    cache_key = f"detector:{model_path}:{model_type}"

    if cache_key not in _model_cache:
        _check_cache_size()

        from onnxtools import create_detector

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Detection model file not found: {model_path}")

        supported_types = ["yolo", "rtdetr", "rfdetr"]
        if model_type not in supported_types:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {supported_types}"
            )

        _model_cache[cache_key] = create_detector(
            model_type=model_type,
            onnx_path=model_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        logger.info(f"Loaded detector: {model_path} (type={model_type})")

    return _model_cache[cache_key]


def get_ocr_model(
    model_path: str,
    conf_thres: float = DEFAULT_CONF_THRESHOLD,
    plate_config_path: Optional[str] = None,
) -> "OcrORT":
    """Get or create OCR model instance with lazy loading.

    Args:
        model_path: Path to OCR ONNX model file
        conf_thres: Confidence threshold
        plate_config_path: Optional path to plate config YAML

    Returns:
        OcrORT instance

    Raises:
        FileNotFoundError: If model file does not exist
    """
    cache_key = f"ocr:{model_path}"

    if cache_key not in _model_cache:
        _check_cache_size()

        from onnxtools import OcrORT

        if not Path(model_path).exists():
            raise FileNotFoundError(f"OCR model file not found: {model_path}")

        kwargs = {"onnx_path": model_path, "conf_thres": conf_thres}
        if plate_config_path:
            kwargs["plate_config_path"] = plate_config_path

        _model_cache[cache_key] = OcrORT(**kwargs)
        logger.info(f"Loaded OCR model: {model_path}")

    return _model_cache[cache_key]


def get_color_layer_classifier(
    model_path: str,
    conf_thres: float = DEFAULT_CONF_THRESHOLD,
) -> "ColorLayerORT":
    """Get or create color/layer classifier instance with lazy loading.

    Args:
        model_path: Path to classification ONNX model file
        conf_thres: Confidence threshold

    Returns:
        ColorLayerORT instance

    Raises:
        FileNotFoundError: If model file does not exist
    """
    cache_key = f"color_layer:{model_path}"

    if cache_key not in _model_cache:
        _check_cache_size()

        from onnxtools import ColorLayerORT

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Classification model file not found: {model_path}")

        _model_cache[cache_key] = ColorLayerORT(
            onnx_path=model_path,
            conf_thres=conf_thres,
        )
        logger.info(f"Loaded color/layer classifier: {model_path}")

    return _model_cache[cache_key]


def _check_cache_size() -> None:
    """Check cache size and remove oldest entries if necessary."""
    if len(_model_cache) >= MAX_CACHED_MODELS:
        # Remove first (oldest) entry
        oldest_key = next(iter(_model_cache))
        del _model_cache[oldest_key]
        logger.warning(f"Model cache full, removed: {oldest_key}")


def clear_cache() -> None:
    """Clear all cached models."""
    _model_cache.clear()
    logger.info("Model cache cleared")


def get_cache_info() -> Dict[str, Any]:
    """Get information about cached models.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "cached_models": len(_model_cache),
        "max_cache_size": MAX_CACHED_MODELS,
        "model_keys": list(_model_cache.keys()),
    }
