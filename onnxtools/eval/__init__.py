"""
评估模块

提供COCO数据集和OCR数据集的评估工具。
"""

from .eval_coco import DatasetEvaluator
from .eval_ocr import OCRDatasetEvaluator, SampleEvaluation

__all__ = [
    'DatasetEvaluator',
    'OCRDatasetEvaluator',
    'SampleEvaluation',
]
