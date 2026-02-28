"""
评估模块

提供COCO数据集、OCR数据集和分类数据集的评估工具。
"""

from .eval_cls import BranchConfig, ClsDatasetEvaluator, ClsSampleEvaluation
from .eval_coco import DetDatasetEvaluator
from .eval_ocr import OCRDatasetEvaluator, SampleEvaluation

__all__ = [
    'DetDatasetEvaluator',
    'OCRDatasetEvaluator',
    'SampleEvaluation',
    'ClsDatasetEvaluator',
    'ClsSampleEvaluation',
    'BranchConfig',
]
