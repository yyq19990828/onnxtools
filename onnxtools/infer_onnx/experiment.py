"""
实验性推理类

存放经过 ONNX 图改造后的模型对应推理类，用于快速验证和原型开发。
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .onnx_base import BaseORT


class RfdetrUnifiedORT(BaseORT):
    """
    RF-DETR Unified 推理类

    适用于经 modify_rfdetr.py 改造后的模型，特征:
    - 预处理: 直接 resize + /255 (与 RT-DETR 一致，ImageNet 归一化在模型内部)
    - 后处理: sigmoid 激活 + 全局 topk 选择 (保留 RF-DETR 原始逻辑)
    - 单输出: [batch, 300, 4+num_classes] = concat(pred_boxes, pred_logits)
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.001, iou_thres: float = 0.5,
                 providers: Optional[List[str]] = None, **kwargs):
        super().__init__(onnx_path, input_shape, conf_thres, providers, **kwargs)

    @staticmethod
    def preprocess(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
        """
        预处理: 直接 resize + /255 归一化 (与 RT-DETR 一致)

        ImageNet mean/std 归一化已烧入 ONNX 模型，外部无需处理。

        Args:
            image: 输入图像 BGR 格式
            input_shape: 目标尺寸 (H, W)

        Returns:
            (tensor, scale, original_shape)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_h, target_w = input_shape

        # 直接 resize（不保持宽高比，与 RT-DETR/RF-DETR 一致）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化到 [0, 1]（模型内部会做 ImageNet mean/std）
        normalized = resized_rgb.astype(np.float32) / 255.0

        # HWC → CHW，添加 batch 维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

        scale = (target_w / w, target_h / h)
        return tensor, scale, original_shape

    @staticmethod
    def postprocess(outputs: list, input_shape: Tuple[int, int],
                    conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        后处理: sigmoid 激活 + 全局 topk (保留 RF-DETR 原始逻辑)

        输出格式: [batch, 300, 4+num_classes] 其中 [:4]=bbox(cxcywh 归一化), [4:]=logits

        Args:
            outputs: 模型输出列表，outputs[0] 为 [batch, 300, 9]
            input_shape: 输入尺寸
            conf_thres: 置信度阈值
            **kwargs: orig_shape 等

        Returns:
            List[np.ndarray]: 每个 batch 的检测结果 [N, 6] = [x1,y1,x2,y2,conf,cls]
        """
        # 统一取第一个输出 (可能是 list 或单个 ndarray)
        preds = outputs[0] if isinstance(outputs, list) else outputs

        bs = preds.shape[0]

        # 分离 bbox 和 logits
        pred_boxes = preds[:, :, :4]    # [batch, 300, 4] cxcywh 归一化
        pred_logits = preds[:, :, 4:]   # [batch, 300, num_classes] raw logits
        num_classes = pred_logits.shape[2]

        results = []

        for i in range(bs):
            out_bbox = pred_boxes[i]      # [300, 4]
            out_logits = pred_logits[i]   # [300, num_classes]

            # sigmoid 激活
            prob = 1.0 / (1.0 + np.exp(-out_logits))

            # 全局 topk 选择 (展平 query×class)
            prob_flat = prob.flatten()
            num_select = min(300, len(prob_flat))
            topk_indices = np.argpartition(prob_flat, -num_select)[-num_select:]
            topk_indices = topk_indices[np.argsort(prob_flat[topk_indices])][::-1]

            scores = prob_flat[topk_indices]
            topk_boxes_idx = topk_indices // num_classes
            labels = topk_indices % num_classes

            boxes = out_bbox[topk_boxes_idx]  # [num_select, 4]

            # cxcywh → xyxy
            x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            w = np.clip(w, a_min=0.0, a_max=None)
            h = np.clip(h, a_min=0.0, a_max=None)

            boxes_xyxy = np.column_stack([
                x_c - 0.5 * w,
                y_c - 0.5 * h,
                x_c + 0.5 * w,
                y_c + 0.5 * h,
            ])

            # 缩放到原图尺寸
            orig_shape = kwargs.get('orig_shape', None)
            if orig_shape is not None:
                orig_h, orig_w = orig_shape
                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h])
            else:
                imgsz_h, imgsz_w = input_shape
                scale_fct = np.array([imgsz_w, imgsz_h, imgsz_w, imgsz_h])
            boxes_xyxy = boxes_xyxy * scale_fct

            # 置信度过滤
            mask = scores > conf_thres
            if np.any(mask):
                pred = np.column_stack([
                    boxes_xyxy[mask],
                    scores[mask],
                    labels[mask],
                ])
            else:
                pred = np.zeros((0, 6))

            results.append(pred)

        return results

    def _prepare_inference(self, image: np.ndarray):
        """重写: 返回 3-tuple (与 RT-DETR/RF-DETR 一致)"""
        input_tensor, scale, original_shape = self.preprocess(image, self.input_shape)
        return input_tensor, scale, original_shape, None

    def _finalize_inference(self, outputs, expected_batch_size, scale,
                            ratio_pad, conf_thres, orig_shape=None, **kwargs):
        """重写: 使用自身的 postprocess (单输出 + sigmoid + topk)"""
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres

        detections = self.postprocess(
            outputs, self.input_shape, effective_conf_thres,
            orig_shape=orig_shape, **kwargs
        )

        if expected_batch_size > 1 and len(detections) > 1:
            detections = [detections[0]]

        return detections
