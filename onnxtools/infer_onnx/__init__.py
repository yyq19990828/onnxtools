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

# Import detection classes from separate modules
from .onnx_base import BaseORT

# Import classification classes (new architecture)
from .onnx_cls import BaseClsORT, ClsResult, ColorLayerORT, VehicleAttributeORT

# Import OCR class
from .onnx_ocr import OcrORT
from .onnx_rfdetr import RfdetrORT
from .onnx_rtdetr import RtdetrORT
from .onnx_yolo import YoloORT

# Import result class
from .result import Result

RUN = "runs"

# This makes `from onnxtools.infer_onnx import *` behave nicely, exporting only these names.
__all__ = [
    # Detection base and implementations
    'BaseORT',
    'YoloORT',
    'RtdetrORT',
    'RfdetrORT',
    # Classification base and implementations
    'BaseClsORT',
    'ClsResult',
    'ColorLayerORT',
    'VehicleAttributeORT',
    # OCR
    'OcrORT',
    # Result classes
    'Result',
    # Constants
    'RUN'
]
