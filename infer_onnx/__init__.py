"""
The infer_onnx package provides classes for ONNX-based inference.
"""

# Import classes from separate modules
from .base_onnx import BaseOnnx
from .yolo_onnx import YoloOnnx
from .rtdetr_onnx import RTDETROnnx
from .rfdetr_onnx import RFDETROnnx
from .infer_models import (
    create_detector,
    DetONNX,  # 向后兼容
    YoloRTDETROnnx,  # 向后兼容
)
from .ocr_onnx import ColorLayerONNX, OCRONNX
from .eval_coco import DatasetEvaluator

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
    'DatasetEvaluator'
]