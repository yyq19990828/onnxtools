"""
RT-DETR模型ORT推理类

包含:
- RtdetrORT: RT-DETR模型ORT推理类 (原YoloRTDETROnnx，已废弃)
完全复刻ultralytics RTDETRValidator的后处理逻辑，端到端检测，无需NMS
模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .infer_utils import xywh2xyxy
from .onnx_base import BaseORT


def apply_softmax(scores: np.ndarray) -> np.ndarray:
    """
    对分类输出应用softmax归一化

    Args:
        scores (np.ndarray): 原始分数 [batch, num_queries, num_classes]

    Returns:
        np.ndarray: softmax归一化后的概率
    """
    # 为了数值稳定性，减去每行的最大值
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)

    # 计算exp
    exp_scores = np.exp(scores_shifted)

    # 计算softmax
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return softmax_scores


def smart_normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    智能检测并处理分类输出的归一化状态

    RT-DETR模型的分类输出可能是：
    1. 未归一化的logits (需要softmax归一化)
    2. 已归一化的概率值 (0-1之间，各类别概率和为1)
    3. 已sigmoid处理的概率值 (多标签情况)

    Args:
        scores (np.ndarray): 分类输出 [batch, num_queries, num_classes]

    Returns:
        np.ndarray: 归一化后的分数
    """
    # 检查输入是否有效
    if scores.size == 0:
        return scores

    # 获取一些统计信息来判断归一化状态
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    scores_mean = np.mean(scores)

    # 检查每个query的分数和（用于判断是否已经softmax归一化）
    scores_sum_per_query = np.sum(scores, axis=-1)  # [batch, num_queries]
    sum_mean = np.mean(scores_sum_per_query)
    sum_std = np.std(scores_sum_per_query)

    logging.debug(f"分类输出统计: min={scores_min:.4f}, max={scores_max:.4f}, "
                 f"mean={scores_mean:.4f}, sum_mean={sum_mean:.4f}, sum_std={sum_std:.4f}")

    # 判断归一化状态的启发式规则
    is_softmax_normalized = (
        0.95 <= sum_mean <= 1.05 and  # 和接近1
        sum_std < 0.1 and             # 标准差很小
        scores_min >= -0.1 and        # 最小值接近0或稍微负数
        scores_max <= 1.1              # 最大值接近1或稍微超过
    )

    is_sigmoid_normalized = (
        scores_min >= -0.1 and        # 最小值接近0
        scores_max <= 1.1 and         # 最大值接近1
        not is_softmax_normalized     # 不是softmax归一化
    )

    is_raw_logits = (
        scores_min < -1.0 or          # 有较大负值
        scores_max > 5.0 or           # 有较大正值
        (scores_max - scores_min) > 10.0  # 值域范围很大
    )

    if is_softmax_normalized:
        logging.debug("检测到已softmax归一化的分类输出，直接使用")
        return scores
    elif is_sigmoid_normalized:
        logging.debug("检测到已sigmoid处理的分类输出，直接使用")
        return scores
    elif is_raw_logits:
        logging.debug("检测到未归一化的logits，应用softmax归一化")
        return apply_softmax(scores)
    else:
        # 边界情况：尝试softmax，如果结果合理就使用，否则直接返回原值
        logging.debug("分类输出状态不明确，尝试softmax归一化")
        softmax_scores = apply_softmax(scores)

        # 检查softmax后的结果是否合理
        softmax_max = np.max(softmax_scores)
        if softmax_max > 0.01:  # softmax后至少有一些有意义的概率值
            logging.debug("应用softmax归一化")
            return softmax_scores
        else:
            logging.warning("softmax归一化后概率值过小，使用原始分数")
            return scores


class RtdetrORT(BaseORT):
    """
    RT-DETR模型ORT推理类 (原YoloRTDETROnnx，已废弃)

    完全复刻ultralytics RTDETRValidator的后处理逻辑，端到端检测，无需NMS
    模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.001, iou_thres: float = 0.5,
                 providers: Optional[List[str]] = None, **kwargs):
        """
        初始化RT-DETR检测器

        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值，默认0.001
            iou_thres (float): IoU阈值，RT-DETR不使用，保持接口统一
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
            **kwargs: 其他参数（如 det_config）
        """
        # 调用BaseOnnx初始化
        super().__init__(onnx_path, input_shape, conf_thres, providers, **kwargs)

        # RT-DETR输出格式验证延迟到模型初始化时进行

    @staticmethod
    def preprocess(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
        """
        RT-DETR预处理静态方法（复刻ultralytics风格，直接resize不保持长宽比）

        Source: ultralytics/data/base.py::BaseDataset.load_image
        参考: ultralytics/models/rtdetr/val.py::RTDETRDataset.build_transforms

        Args:
            image (np.ndarray): 输入图像，BGR格式
            input_shape (Tuple[int, int]): 输入尺寸

        Returns:
            Tuple: (预处理后的tensor, scale, 原始形状)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_h, target_w = input_shape

        # 直接resize到目标尺寸，不保持长宽比（与ultralytics一致）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 转换为RGB（ultralytics通常使用RGB）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化到[0,1]
        normalized = resized_rgb.astype(np.float32) / 255.0

        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

        # 计算缩放因子（用于最终的坐标还原）
        scale_h = target_h / h
        scale_w = target_w / w

        # RT-DETR直接拉伸，需要返回实际的缩放因子
        # 返回(scale_w, scale_h)作为scale，在后处理中使用
        scale = (scale_w, scale_h)

        return tensor, scale, original_shape

    @staticmethod
    def postprocess(preds: np.ndarray, input_shape: Tuple[int, int], conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        RT-DETR后处理（优化版本，支持自适应分类输出归一化处理）

        Source: ultralytics/models/rtdetr/val.py::RTDETRValidator.postprocess
        原函数路径: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L157-L188

        Args:
            preds (np.ndarray): 模型原始输出 [batch, 300, 19]
            conf_thres (float): 置信度阈值
            **kwargs: 其他参数（RT-DETR不使用scale等）

        Returns:
            List[np.ndarray]: 检测结果列表，坐标在输入图像尺寸上
        """
        # 处理预测格式（复刻 ultralytics/models/rtdetr/val.py#L173-L174）
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, _ = preds[0].shape  # batch_size, num_queries(300), num_features(19)

        # 分割bbox和scores（复刻 ultralytics/models/rtdetr/val.py#L177）
        bboxes = preds[0][:, :, :4]    # [batch, 300, 4] - bbox坐标
        scores = preds[0][:, :, 4:]    # [batch, 300, 15] - 类别分数

        # 智能检测和处理分类输出的归一化状态
        scores = smart_normalize_scores(scores)

        # 缩放bbox到原图尺寸（复刻 ultralytics/models/rtdetr/val.py#L178）
        # RT-DETR输出的bbox是归一化坐标[0,1]，需要转换为像素坐标
        # 如果有orig_shape参数，直接缩放到原图；否则缩放到输入尺寸
        orig_shape = kwargs.get('orig_shape', None)
        if orig_shape is not None:
            # 直接缩放到原图尺寸
            orig_h, orig_w = orig_shape  # (H, W)
            # bboxes格式是[cx, cy, w, h]，需要按宽高分别缩放
            bboxes = bboxes.copy()
            bboxes[:, :, [0, 2]] *= orig_w  # cx, w 按宽度缩放
            bboxes[:, :, [1, 3]] *= orig_h  # cy, h 按高度缩放
        else:
            # 回退到输入尺寸
            imgsz = input_shape[0]  # ultralytics假设输入是正方形
            bboxes = bboxes * imgsz

        # 初始化输出
        outputs = []

        # 为每个batch中的图像处理
        for i in range(bs):
            bbox = bboxes[i]  # [300, 4]
            score_matrix = scores[i]  # [300, 15]

            # 坐标转换从xywh到xyxy（复刻 ultralytics/models/rtdetr/val.py#L181）
            bbox = xywh2xyxy(bbox)

            # 获取每个query的最大类别分数和索引（复刻 ultralytics/models/rtdetr/val.py#L182）
            score = np.max(score_matrix, axis=-1)  # [300,] - 最大分数
            cls = np.argmax(score_matrix, axis=-1)  # [300,] - 类别索引

            # 组合预测结果（复刻 ultralytics/models/rtdetr/val.py#L183）
            pred = np.column_stack([bbox, score, cls])  # [300, 6]

            # 按置信度排序（复刻 ultralytics/models/rtdetr/val.py#L185）
            sorted_indices = np.argsort(score)[::-1]  # 降序排序
            pred = pred[sorted_indices]
            score_sorted = score[sorted_indices]

            # 置信度过滤（复刻 ultralytics/models/rtdetr/val.py#L186）
            mask = score_sorted > conf_thres
            pred = pred[mask]

            outputs.append(pred)

        # 重要：RT-DETR返回的坐标是在输入图像尺寸上的，与ultralytics一致
        # 坐标缩放在后续的evaluate阶段进行
        return outputs
