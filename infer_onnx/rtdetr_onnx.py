"""
RT-DETR模型ONNX推理类

包含:
- RTDETROnnx: RT-DETR模型ONNX推理类 (原YoloRTDETROnnx)
完全复刻ultralytics RTDETRValidator的后处理逻辑，端到端检测，无需NMS
模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
"""

import numpy as np
import logging
import cv2
from typing import List, Tuple, Optional

from .base_onnx import BaseOnnx
from .infer_utils import xywh2xyxy


class RTDETROnnx(BaseOnnx):
    """
    RT-DETR模型ONNX推理类 (原YoloRTDETROnnx)
    
    完全复刻ultralytics RTDETRValidator的后处理逻辑，端到端检测，无需NMS
    模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.001, iou_thres: float = 0.5, 
                 providers: Optional[List[str]] = None):
        """
        初始化RT-DETR检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值，默认0.001
            iou_thres (float): IoU阈值，RT-DETR不使用，保持接口统一
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
        """
        # 调用BaseOnnx初始化
        super().__init__(onnx_path, input_shape, conf_thres, providers)
        
        # RT-DETR输出格式验证延迟到模型初始化时进行
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """
        RT-DETR预处理（实例方法，向后兼容）
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (预处理后的tensor, scale, 原始形状)
        """
        return self._preprocess_static(image, self.input_shape)
    
    @staticmethod
    def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
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
    
    def _postprocess(self, preds: np.ndarray, conf_thres: float, **_kwargs) -> List[np.ndarray]:
        """
        RT-DETR后处理（完全复刻ultralytics RTDETRValidator.postprocess）
        
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
        
        # 缩放bbox到输入图像尺寸（复刻 ultralytics/models/rtdetr/val.py#L178）
        # RT-DETR输出的bbox是归一化坐标[0,1]，需要乘以输入尺寸转换为像素坐标
        imgsz = self.input_shape[0]  # ultralytics假设输入是正方形
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


# 为了向后兼容，保留旧的类名
YoloRTDETROnnx = RTDETROnnx