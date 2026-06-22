"""Utility exports for onnxtools.

Submodules are loaded on demand so lightweight installs can use logging or
tracking helpers without importing visualization/image-processing dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "draw_detections": ("onnxtools.utils.drawing", "draw_detections"),
    "convert_to_supervision_detections": ("onnxtools.utils.drawing", "convert_to_supervision_detections"),
    "UltralyticsLetterBox": ("onnxtools.utils.image_processing", "UltralyticsLetterBox"),
    "setup_logger": ("onnxtools.utils.logger", "setup_logger"),
    "non_max_suppression": ("onnxtools.utils.nms", "non_max_suppression"),
    "create_ocr_labels": ("onnxtools.utils.supervision_labels", "create_ocr_labels"),
    "get_chinese_font_path": ("onnxtools.utils.font_utils", "get_chinese_font_path"),
    "get_fallback_font_path": ("onnxtools.utils.font_utils", "get_fallback_font_path"),
}


def __getattr__(name: str) -> Any:
    """Lazily load utility exports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "draw_detections",
    "UltralyticsLetterBox",
    "non_max_suppression",
    "convert_to_supervision_detections",
    "create_ocr_labels",
    "get_chinese_font_path",
    "get_fallback_font_path",
    "setup_logger",
]
