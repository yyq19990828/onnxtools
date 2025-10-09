"""
The infer_onnx package provides classes for ONNX-based inference.
"""

# Import classes from separate modules
from .onnx_base import BaseOnnx
from .onnx_yolo import YoloOnnx
from .onnx_rtdetr import RTDETROnnx
from .onnx_rfdetr import RFDETROnnx
from .infer_models import (
    create_detector,
    DetONNX,  # 向后兼容
    YoloRTDETROnnx,  # 向后兼容
)
from .onnx_ocr import ColorLayerONNX, OCRONNX
from .eval_coco import DatasetEvaluator
RUN = "runs"

# This makes `from infer_onnx import *` behave nicely, exporting only these names.
__all__ = [
    'BaseOnnx',
    'YoloOnnx', 
    'RTDETROnnx', 
    'RFDETROnnx', 
    'DetONNX',  # 向后兼容
    'YoloRTDETROnnx',  # 向后兼容
    'ColorLayerONNX', 
    'OCRONNX',
    'create_detector',
    'DatasetEvaluator',
    'RUN'
]