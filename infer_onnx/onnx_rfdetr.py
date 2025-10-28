"""
RF-DETR模型ONNX推理类

包含:
- RFDETROnnx: RF-DETR模型类
支持RF-DETR (ResNet-based Feature Pyramid + DETR) 模型
输出格式：两个独立的输出 - pred_boxes 和 pred_logits
"""

import numpy as np
import logging
import cv2
from typing import List, Tuple, Optional

from .onnx_base import BaseOnnx


class RFDETROnnx(BaseOnnx):
    """
    RF-DETR模型ONNX推理类
    
    支持RF-DETR (ResNet-based Feature Pyramid + DETR) 模型
    输出格式：两个独立的输出 - pred_boxes 和 pred_logits
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (576, 576), 
                 conf_thres: float = 0.001, iou_thres: float = 0.5,
                 providers: Optional[List[str]] = None):
        """
        初始化RF-DETR检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸，默认576x576
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值，RF-DETR不使用，保持接口统一
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
        """
        super().__init__(onnx_path, input_shape, conf_thres, providers)
        
        # RF-DETR输出格式验证延迟到模型初始化时进行
    
    def _validate_rf_detr_format(self):
        """验证RF-DETR输出格式"""
        with self._runner:
            # 检查模型期望的batch维度
            input_metadata = self._runner.get_input_metadata()
            input_shape = input_metadata[self.input_name].shape
            expected_batch_size = input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else 1
            
            dummy_input = np.random.randn(expected_batch_size, 3, self.input_shape[0], self.input_shape[1]).astype(np.float32)
            
            feed_dict = {self.input_name: dummy_input}
            outputs_dict = self._runner.infer(feed_dict)
            outputs = [outputs_dict[name] for name in self.output_names]
            
            if len(outputs) != 2:
                raise ValueError(f"RF-DETR模型应该有2个输出（pred_boxes, pred_logits），但实际有{len(outputs)}个")
            
            pred_boxes_shape, pred_logits_shape = outputs[0].shape, outputs[1].shape
            logging.info(f"RF-DETR输出格式 - pred_boxes: {pred_boxes_shape}, pred_logits: {pred_logits_shape}")
            
            if len(pred_boxes_shape) != 3 or pred_boxes_shape[2] != 4:
                logging.warning(f"警告: pred_boxes形状 {pred_boxes_shape} 可能不符合标准RF-DETR格式")
            
            if len(pred_logits_shape) != 3 or pred_logits_shape[1] != pred_boxes_shape[1]:
                logging.warning(f"警告: pred_logits形状 {pred_logits_shape} 与pred_boxes不匹配")
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """
        RF-DETR预处理（实例方法，向后兼容）
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (预处理后的tensor, scale, 原始形状)
        """
        return self._preprocess_static(image, self.input_shape)
    
    @staticmethod
    def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
        """
        RF-DETR预处理静态方法（对齐原始实现）
        
        基于third_party/rfdetr/datasets/coco.py中的make_coco_transforms实现
        使用ImageNet标准化参数：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            input_shape (Tuple[int, int]): 输入尺寸
            
        Returns:
            Tuple: (预处理后的tensor, scale, 原始形状)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_h, target_w = input_shape
        
        # 直接resize到目标尺寸（对齐SquareResize）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB（PyTorch通常使用RGB）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为[0,1]范围（对齐ToTensor）
        normalized = resized_rgb.astype(np.float32) / 255.0
        
        # 应用ImageNet标准化（对齐Normalize）
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (normalized - imagenet_mean) / imagenet_std
        
        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        # 计算缩放因子
        scale_h = target_h / h
        scale_w = target_w / w
        # RF-DETR直接拉伸，需要返回实际的缩放因子
        scale = (scale_w, scale_h)
        
        return tensor, scale, original_shape
    
    def _postprocess(self, outputs: List[np.ndarray], conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        RF-DETR后处理逻辑（对齐原始实现）
        
        基于third_party/rfdetr/models/lwdetr.py中的PostProcess类实现
        
        Args:
            outputs (List[np.ndarray]): 模型原始输出 [pred_boxes, pred_logits]
            conf_thres (float): 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            List[np.ndarray]: 后处理后的检测结果
        """
        # 首次调用时验证RF-DETR格式
        try:
            self._validate_rf_detr_format()
        except Exception as e:
            logging.warning(f"RF-DETR格式验证失败: {e}")
        
        # 根据实际输出形状确定pred_logits和pred_boxes
        # 输出0: (4, 300, 4) - 应该是pred_boxes
        # 输出1: (4, 300, 15) - 应该是pred_logits
        if outputs[0].shape[2] == 4 and outputs[1].shape[2] > 4:
            pred_boxes, pred_logits = outputs[0], outputs[1]
        else:
            pred_logits, pred_boxes = outputs[0], outputs[1]
        
        bs = pred_boxes.shape[0]
        num_queries = pred_boxes.shape[1]
        num_classes = pred_logits.shape[2]
        
        results = []
        
        # 为每个batch中的图像处理
        for i in range(bs):
            out_logits = pred_logits[i]  # [300, 15]
            out_bbox = pred_boxes[i]     # [300, 4]
            
            # 使用sigmoid激活（对齐原始实现）
            prob = 1.0 / (1.0 + np.exp(-out_logits))  # sigmoid
            
            # topk选择（对齐原始PostProcess.forward）
            # 将prob展平并选择top-k个值
            prob_flat = prob.flatten()  # [300*15]
            
            # 选择top 300个检测（num_select=300）
            num_select = min(300, len(prob_flat))
            topk_indices = np.argpartition(prob_flat, -num_select)[-num_select:]
            topk_indices = topk_indices[np.argsort(prob_flat[topk_indices])][::-1]  # 降序排序
            
            scores = prob_flat[topk_indices]
            
            # 计算对应的box和label索引
            topk_boxes_idx = topk_indices // num_classes  # 对应的query索引
            labels = topk_indices % num_classes           # 对应的类别索引
            
            # 提取对应的bbox
            boxes = out_bbox[topk_boxes_idx]  # [num_select, 4]
            
            # 坐标转换：从cxcywh到xyxy（对齐box_ops.box_cxcywh_to_xyxy）
            x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            w = np.clip(w, a_min=0.0, a_max=None)  # clamp(min=0.0)
            h = np.clip(h, a_min=0.0, a_max=None)
            
            x1 = x_c - 0.5 * w
            y1 = y_c - 0.5 * h  
            x2 = x_c + 0.5 * w
            y2 = y_c + 0.5 * h
            
            boxes_xyxy = np.column_stack([x1, y1, x2, y2])
            
            # 缩放到输入图像尺寸（从归一化坐标[0,1]到像素坐标）
            imgsz_w, imgsz_h = self.input_shape[1], self.input_shape[0]  # (width, height)
            scale_fct = np.array([imgsz_w, imgsz_h, imgsz_w, imgsz_h])
            boxes_xyxy = boxes_xyxy * scale_fct
            
            # 置信度过滤
            mask = scores > conf_thres
            if np.any(mask):
                final_scores = scores[mask]
                final_labels = labels[mask]
                final_boxes = boxes_xyxy[mask]
                
                # 组合结果：[x1, y1, x2, y2, conf, class]
                pred = np.column_stack([final_boxes, final_scores, final_labels])
            else:
                pred = np.zeros((0, 6))
            
            results.append(pred)
        
        return results