"""
ONNX模型推理工厂函数

包含:
- create_detector: 根据模型类型创建相应的检测器的工厂函数
- 向后兼容的别名导入

统一API设计，减少代码冗余，提高可维护性
"""

from .base_onnx import BaseOnnx
from .yolo_onnx import YoloOnnx
from .rtdetr_onnx import RTDETROnnx  
from .rfdetr_onnx import RFDETROnnx


def create_detector(model_type: str, onnx_path: str, **kwargs) -> BaseOnnx:
    """
    工厂函数：根据模型类型创建相应的检测器（支持Polygraphy懒加载）
    
    Args:
        model_type (str): 模型类型，支持 'yolo', 'rtdetr', 'rfdetr'
        onnx_path (str): ONNX模型路径
        **kwargs: 其他参数，包括providers等
        
    Returns:
        BaseOnnx: 相应的检测器实例
        
    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = model_type.lower()
    
    if model_type in ['yolo', 'yolov5', 'yolov8']:
        return YoloOnnx(onnx_path, **kwargs)
    elif model_type in ['rtdetr', 'rt-detr']:
        return RTDETROnnx(onnx_path, **kwargs)
    elif model_type in ['rfdetr', 'rf-detr']:
        return RFDETROnnx(onnx_path, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 为了向后兼容，保留旧的类名
DetONNX = YoloOnnx
YoloRTDETROnnx = RTDETROnnx