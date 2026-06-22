"""评估模块。

提供 COCO 检测、OCR、分类与 MOT（多目标跟踪）数据集的评估工具。

各评估器按需惰性加载（PEP 562 ``__getattr__``）：检测/OCR/分类评估器依赖 ONNX
推理栈，而 MOT 评估器（:class:`MOTEvaluator`）仅依赖 numpy + motmetrics/trackeval
（``[mot]`` extra），因此在纯 ``[tracking]`` / ``[mot]`` 安装下也能导入，不会触发
``onnxruntime`` 导入。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    # ONNX 推理栈相关评估器
    "DetDatasetEvaluator": ("onnxtools.eval.eval_coco", "DetDatasetEvaluator"),
    "OCRDatasetEvaluator": ("onnxtools.eval.eval_ocr", "OCRDatasetEvaluator"),
    "SampleEvaluation": ("onnxtools.eval.eval_ocr", "SampleEvaluation"),
    "ClsDatasetEvaluator": ("onnxtools.eval.eval_cls", "ClsDatasetEvaluator"),
    "ClsSampleEvaluation": ("onnxtools.eval.eval_cls", "ClsSampleEvaluation"),
    "BranchConfig": ("onnxtools.eval.eval_cls", "BranchConfig"),
    # MOT 评估器（无需 ONNX）
    "MOTEvaluator": ("onnxtools.eval.eval_mot", "MOTEvaluator"),
    "MOTResult": ("onnxtools.eval.eval_mot", "MOTResult"),
    "run_tracker_on_gt": ("onnxtools.eval.eval_mot", "run_tracker_on_gt"),
}


def __getattr__(name: str) -> Any:
    """惰性加载评估器，使纯 tracking/MOT 安装避免 ONNX 导入。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DetDatasetEvaluator",
    "OCRDatasetEvaluator",
    "SampleEvaluation",
    "ClsDatasetEvaluator",
    "ClsSampleEvaluation",
    "BranchConfig",
    "MOTEvaluator",
    "MOTResult",
    "run_tracker_on_gt",
]
