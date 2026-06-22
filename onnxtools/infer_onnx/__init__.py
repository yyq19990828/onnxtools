"""
The infer_onnx package provides classes for ONNX Runtime-based inference.

Detection Models (inherit BaseORT):
- YoloORT: YOLO series detection
- RtdetrORT: RT-DETR detection
- RfdetrORT: RF-DETR detection

Classification Models (inherit BaseClsORT):
- ColorLayerORT: Vehicle plate color and layer classification

OCR Models (independent):
- OcrORT: Optical Character Recognition
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

RUN = "runs"

_LAZY_EXPORTS = {
    # Detection base and implementations
    "BaseORT": ("onnxtools.infer_onnx.onnx_base", "BaseORT"),
    "YoloORT": ("onnxtools.infer_onnx.onnx_yolo", "YoloORT"),
    "RtdetrORT": ("onnxtools.infer_onnx.onnx_rtdetr", "RtdetrORT"),
    "RfdetrORT": ("onnxtools.infer_onnx.onnx_rfdetr", "RfdetrORT"),
    "RfdetrUnifiedORT": ("onnxtools.infer_onnx.experiment", "RfdetrUnifiedORT"),
    # Classification base and implementations
    "BaseClsORT": ("onnxtools.infer_onnx.onnx_cls", "BaseClsORT"),
    "ClsResult": ("onnxtools.infer_onnx.onnx_cls", "ClsResult"),
    "ColorLayerORT": ("onnxtools.infer_onnx.onnx_cls", "ColorLayerORT"),
    "HelmetORT": ("onnxtools.infer_onnx.onnx_cls", "HelmetORT"),
    "VehicleAttributeORT": ("onnxtools.infer_onnx.onnx_cls", "VehicleAttributeORT"),
    # OCR
    "OcrORT": ("onnxtools.infer_onnx.onnx_ocr", "OcrORT"),
    # Result classes
    "Result": ("onnxtools.infer_onnx.result", "Result"),
}


def __getattr__(name: str) -> Any:
    """Lazily load inference exports so Result can be used without ONNX deps."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Detection base and implementations
    "BaseORT",
    "YoloORT",
    "RtdetrORT",
    "RfdetrORT",
    "RfdetrUnifiedORT",
    # Classification base and implementations
    "BaseClsORT",
    "ClsResult",
    "ColorLayerORT",
    "HelmetORT",
    "VehicleAttributeORT",
    # OCR
    "OcrORT",
    # Result classes
    "Result",
    # Constants
    "RUN",
]
