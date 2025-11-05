"""
onnxtools - ONNX模型推理工具集

提供多种目标检测模型架构的统一推理接口，包括YOLO、RT-DETR、RF-DETR等。
"""

# 从 infer_onnx 子模块导入推理引擎类
from .infer_onnx import (
    BaseORT,
    YoloORT,
    RtdetrORT,
    RfdetrORT,
    ColorLayerORT,
    OcrORT,
    Result,
    DatasetEvaluator,
    OCRDatasetEvaluator,
    SampleEvaluation,
    RUN,
)

# 从 utils 子模块导入工具函数
from .utils import (
    setup_logger,
)


# ============================================================================
# 工厂函数：根据模型类型创建相应的检测器
# ============================================================================

def create_detector(model_type: str, onnx_path: str, **kwargs) -> BaseORT:
    """
    工厂函数：根据模型类型创建相应的检测器（支持Polygraphy懒加载）

    Args:
        model_type (str): 模型类型，支持 'yolo', 'rtdetr', 'rfdetr'
        onnx_path (str): ONNX模型路径
        **kwargs: 其他参数，包括providers等

    Returns:
        BaseORT: 相应的检测器实例

    Raises:
        ValueError: 不支持的模型类型

    Examples:
        >>> detector = create_detector('yolo', 'models/yolo11n.onnx', conf_thres=0.5)
        >>> detector = create_detector('rtdetr', 'models/rtdetr.onnx', iou_thres=0.7)
    """
    model_type = model_type.lower()

    if model_type in ['yolo', 'yolov5', 'yolov8', 'yolov11']:
        return YoloORT(onnx_path, **kwargs)
    elif model_type in ['rtdetr', 'rt-detr']:
        return RtdetrORT(onnx_path, **kwargs)
    elif model_type in ['rfdetr', 'rf-detr']:
        return RfdetrORT(onnx_path, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: yolo, rtdetr, rfdetr")



__version__ = "0.1.0"

__all__ = [
    # 推理引擎
    'BaseORT',
    'YoloORT',
    'RtdetrORT',
    'RfdetrORT',
    'ColorLayerORT',
    'OcrORT',
    'Result',
    'create_detector',
    # 评估工具
    'DatasetEvaluator',
    'OCRDatasetEvaluator',
    'SampleEvaluation',
    # 工具函数
    'setup_logger',
    # 常量
    'RUN',
]
