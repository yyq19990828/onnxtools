"""
onnxtools - ONNX模型推理工具集

提供多种目标检测模型架构的统一推理接口，包括YOLO、RT-DETR、RF-DETR等。
"""

# 从 eval 子模块导入评估工具
from .eval import (
    BranchConfig,
    ClsDatasetEvaluator,
    ClsSampleEvaluation,
    DetDatasetEvaluator,
    OCRDatasetEvaluator,
    SampleEvaluation,
)

# 从 infer_onnx 子模块导入推理引擎类
from .infer_onnx import (
    RUN,
    BaseClsORT,
    BaseORT,
    ClsResult,
    ColorLayerORT,
    HelmetORT,
    OcrORT,
    Result,
    RfdetrORT,
    RtdetrORT,
    VehicleAttributeORT,
    YoloORT,
)
from .infer_onnx.experiment import RfdetrUnifiedORT

# 从 pipeline 子模块导入推理管道类
from .pipeline import InferencePipeline

# 从 utils 子模块导入工具函数
from .utils import setup_logger

# ============================================================================
# 工厂函数：根据模型类型创建相应的检测器
# ============================================================================


def create_detector(model_type: str, onnx_path: str, **kwargs) -> BaseORT:
    """根据模型类型创建相应的检测器实例(工厂函数)。

    Args:
        model_type (str): 模型类型,大小写不敏感。支持以下别名:

            - YOLO 系列: ``'yolo'`` / ``'yolov5'`` / ``'yolov8'`` / ``'yolov11'``
              → :class:`YoloORT`
            - RT-DETR: ``'rtdetr'`` / ``'rt-detr'`` → :class:`RtdetrORT`
            - RF-DETR: ``'rfdetr'`` / ``'rf-detr'`` → :class:`RfdetrORT`
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
        return YoloORT(onnx_path, **kwargs)
    elif model_type in ["rtdetr", "rt-detr"]:
        return RtdetrORT(onnx_path, **kwargs)
    elif model_type in ["rfdetr", "rf-detr"]:
        return RfdetrORT(onnx_path, **kwargs)
    elif model_type in ["rfdetr_unified", "rfdetr-unified"]:
        return RfdetrUnifiedORT(onnx_path, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: yolo, rtdetr, rfdetr")


__version__ = "0.1.0"

__all__ = [
    # Detection base and implementations
    "BaseORT",
    "YoloORT",
    "RtdetrORT",
    "RfdetrORT",
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
    # Evaluation tools
    "DetDatasetEvaluator",
    "OCRDatasetEvaluator",
    "SampleEvaluation",
    "ClsDatasetEvaluator",
    "ClsSampleEvaluation",
    "BranchConfig",
    # Utilities
    "setup_logger",
    # Constants
    "RUN",
]
