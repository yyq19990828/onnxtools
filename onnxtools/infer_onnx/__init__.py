"""
The infer_onnx package provides classes for ONNX Runtime-based inference.
"""

# Import classes from separate modules
from .onnx_base import BaseORT
from .onnx_yolo import YoloORT
from .onnx_rtdetr import RtdetrORT
from .onnx_rfdetr import RfdetrORT
from .onnx_ocr import ColorLayerORT, OcrORT
from .result import Result

RUN = "runs"

# This makes `from onnxtools.infer_onnx import *` behave nicely, exporting only these names.
__all__ = [
    'BaseORT',
    'YoloORT',
    'RtdetrORT',
    'RfdetrORT',
    'ColorLayerORT',
    'OcrORT',
    'Result',
    'RUN'
]