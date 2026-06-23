"""
onnxtools - ONNX模型推理工具集

提供多种目标检测模型架构的统一推理接口，包括YOLO、RT-DETR、RF-DETR等。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    # Detection base and implementations
    "BaseORT": ("onnxtools.infer_onnx", "BaseORT"),
    "YoloORT": ("onnxtools.infer_onnx", "YoloORT"),
    "RtdetrORT": ("onnxtools.infer_onnx", "RtdetrORT"),
    "RfdetrORT": ("onnxtools.infer_onnx", "RfdetrORT"),
    "RfdetrUnifiedORT": ("onnxtools.infer_onnx.experiment", "RfdetrUnifiedORT"),
    # Classification base and implementations
    "BaseClsORT": ("onnxtools.infer_onnx", "BaseClsORT"),
    "ClsResult": ("onnxtools.infer_onnx", "ClsResult"),
    "ColorLayerORT": ("onnxtools.infer_onnx", "ColorLayerORT"),
    "HelmetORT": ("onnxtools.infer_onnx", "HelmetORT"),
    "VehicleAttributeORT": ("onnxtools.infer_onnx", "VehicleAttributeORT"),
    # OCR
    "OcrORT": ("onnxtools.infer_onnx", "OcrORT"),
    # Result classes
    "Result": ("onnxtools.infer_onnx.result", "Result"),
    # Inference pipeline
    "InferencePipeline": ("onnxtools.pipeline", "InferencePipeline"),
    "VehicleAttributePipeline": ("onnxtools.pipeline", "VehicleAttributePipeline"),
    # Evaluation tools
    "DetDatasetEvaluator": ("onnxtools.eval", "DetDatasetEvaluator"),
    "OCRDatasetEvaluator": ("onnxtools.eval", "OCRDatasetEvaluator"),
    "SampleEvaluation": ("onnxtools.eval", "SampleEvaluation"),
    "ClsDatasetEvaluator": ("onnxtools.eval", "ClsDatasetEvaluator"),
    "ClsSampleEvaluation": ("onnxtools.eval", "ClsSampleEvaluation"),
    "BranchConfig": ("onnxtools.eval", "BranchConfig"),
    "MOTEvaluator": ("onnxtools.eval", "MOTEvaluator"),
    "MOTResult": ("onnxtools.eval", "MOTResult"),
    "run_tracker_on_gt": ("onnxtools.eval", "run_tracker_on_gt"),
    # Utilities
    "setup_logger": ("onnxtools.utils", "setup_logger"),
    # Constants
    "RUN": ("onnxtools.infer_onnx", "RUN"),
}


def __getattr__(name: str) -> Any:
    """Lazily load optional subsystems so tracking-only installs avoid ONNX imports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def create_detector(model_type: str, onnx_path: str, **kwargs: Any) -> Any:
    """根据模型类型创建相应的检测器实例(工厂函数)。

    Args:
        model_type (str): 模型类型,大小写不敏感。支持以下别名:

            - YOLO 系列: ``'yolo'`` / ``'yolov5'`` / ``'yolov8'`` / ``'yolov11'``
              -> :class:`YoloORT`
            - RT-DETR: ``'rtdetr'`` / ``'rt-detr'`` -> :class:`RtdetrORT`
            - RF-DETR: ``'rfdetr'`` / ``'rf-detr'`` -> :class:`RfdetrORT`
            - RF-DETR Unified (实验性): ``'rfdetr_unified'`` / ``'rfdetr-unified'``

        onnx_path (str): ONNX 模型文件路径。
        **kwargs: 透传给具体检测器构造函数,常用项:

            - ``conf_thres`` (float): 置信度阈值
            - ``iou_thres`` (float): NMS IoU 阈值(仅 YOLO 实际使用)
            - ``input_shape`` (Tuple[int, int]): 输入尺寸 (H, W)
            - ``providers`` (List[str]): ONNX Runtime 执行提供程序
            - ``det_config`` (str | Dict[int, str]): 类别配置文件或字典

    Returns:
        BaseORT: 检测器实例。调用 ``detector(image)`` 得到 :class:`Result` 对象。

    Raises:
        ValueError: 当 ``model_type`` 不在支持列表中时抛出。

    Examples:
        >>> from onnxtools import create_detector
        >>> det = create_detector('yolo', 'models/yolo11n.onnx', conf_thres=0.5)
        >>> det = create_detector('rtdetr', 'models/rtdetr.onnx', iou_thres=0.7)
        >>> result = det(image)               # Result 实例
    """
    model_type = model_type.lower()

    if model_type in ["yolo", "yolov5", "yolov8", "yolov11"]:
        from .infer_onnx import YoloORT

        return YoloORT(onnx_path, **kwargs)
    if model_type in ["rtdetr", "rt-detr"]:
        from .infer_onnx import RtdetrORT

        return RtdetrORT(onnx_path, **kwargs)
    if model_type in ["rfdetr", "rf-detr"]:
        from .infer_onnx import RfdetrORT

        return RfdetrORT(onnx_path, **kwargs)
    if model_type in ["rfdetr_unified", "rfdetr-unified"]:
        from .infer_onnx.experiment import RfdetrUnifiedORT

        return RfdetrUnifiedORT(onnx_path, **kwargs)
    raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: yolo, rtdetr, rfdetr")


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
    # Factory function
    "create_detector",
    # Inference pipeline
    "InferencePipeline",
    "VehicleAttributePipeline",
    # Evaluation tools
    "DetDatasetEvaluator",
    "OCRDatasetEvaluator",
    "SampleEvaluation",
    "ClsDatasetEvaluator",
    "ClsSampleEvaluation",
    "BranchConfig",
    "MOTEvaluator",
    "MOTResult",
    "run_tracker_on_gt",
    # Utilities
    "setup_logger",
    # Constants
    "RUN",
]
