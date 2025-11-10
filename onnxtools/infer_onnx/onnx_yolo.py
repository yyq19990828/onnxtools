"""
YOLO模型ORT推理类

包含:
- YoloORT: 传统YOLO模型ORT推理类 (原DetONNX)
支持YOLOv5、YOLOv8等使用NMS后处理的YOLO模型
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

from .onnx_base import BaseORT
from .infer_utils import scale_boxes
from onnxtools.utils.nms import non_max_suppression


class YoloORT(BaseORT):
    """
    传统YOLO模型ORT推理类 (原DetONNX)
    
    支持YOLOv5、YOLOv8等使用NMS后处理的YOLO模型
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.5, iou_thres: float = 0.5,
                 multi_label: bool = True, has_objectness: bool = False,
                 providers: Optional[List[str]] = None, **kwargs):
        """
        初始化YOLO检测器

        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            multi_label (bool): 是否允许多标签检测，默认True
            has_objectness (bool): 模型是否有objectness分支，默认False（适应现代YOLO）
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
            **kwargs: 其他参数（如 det_config）

        Note:
            统一使用Ultralytics风格的预处理(LetterBox)
        """
        super().__init__(onnx_path, input_shape, conf_thres, providers, **kwargs)
        self.iou_thres = iou_thres
        self.multi_label = multi_label
        self.has_objectness = has_objectness
        # YoloORT类固定使用Ultralytics预处理方式
    
    @staticmethod
    def preprocess(
        image: np.ndarray,
        input_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, tuple, Optional[tuple]]:
        """
        YOLO预处理静态方法 (统一使用Ultralytics风格)

        Args:
            image (np.ndarray): 输入图像，BGR格式
            input_shape (Tuple[int, int]): 输入尺寸

        Returns:
            Tuple: (input_tensor, scale, original_shape, ratio_pad)
                - input_tensor: 预处理后的张量 [1, 3, H, W]
                - scale: 缩放因子
                - original_shape: 原始图像形状 (H, W)
                - ratio_pad: ((scale_w, scale_h), (pad_w, pad_h))

        Note:
            使用Ultralytics LetterBox预处理,保持宽高比并填充
        """
        from onnxtools.utils.image_processing import UltralyticsLetterBox
        letterbox = UltralyticsLetterBox(new_shape=input_shape)
        input_tensor, scale, original_shape, pad = letterbox(image)

        # 构造scale_boxes期望的ratio_pad格式: ((ratio_w, ratio_h), (pad_w, pad_h))
        ratio_pad = ((scale, scale), pad)
        return input_tensor, scale, original_shape, ratio_pad
    
    @staticmethod
    def postprocess(prediction: np.ndarray, input_shape: Tuple[int, int], conf_thres: float,
                   iou_thres: float, multi_label: bool, has_objectness: bool,
                   scale: float = 1.0, ratio_pad: Optional[tuple] = None,
                   orig_shape: Optional[tuple] = None) -> List[np.ndarray]:
        """
        YOLO模型后处理，包含NMS

        Args:
            prediction (np.ndarray): 模型原始输出
            conf_thres (float): 置信度阈值
            scale (float): 图像缩放因子
            ratio_pad (Optional[tuple]): Ratio和padding信息 (((ratio_w, ratio_h), (pad_w, pad_h)))
            iou_thres (Optional[float]): IoU阈值
            orig_shape (Optional[tuple]): 原始图像尺寸 (height, width)

        Returns:
            List[np.ndarray]: 后处理后的检测结果
        """
        # 自适应处理YOLO输出格式
        # YOLO官方库输出: [B, bbox+C, N] 例如 (1, 84, 8400)
        # 我们的处理逻辑: [B, N, bbox+C] 例如 (1, 8400, 84)
        original_shape = prediction.shape
        
        # 鲁棒的维度判断逻辑：
        # 1. 如果第二维小于第三维，且第二维在合理的特征数范围内(4-200)，则可能是[B, C, N]格式
        # 2. 特征数应该是4+类别数，一般在80-200之间比较合理
        if (prediction.shape[1] < prediction.shape[2] and 
            4 <= prediction.shape[1] <= 200 and
            prediction.shape[2] > prediction.shape[1]):
            # 转换为我们期望的格式: [B, N, C]
            prediction = prediction.transpose(0, 2, 1)
            logging.info(f"YOLO输出格式自适应转换: {original_shape} -> {prediction.shape}")
        
        # 验证最终格式的合理性
        if prediction.shape[2] < 4:
            raise ValueError(f"YOLO输出格式异常: {prediction.shape}，特征维度应该至少包含4个bbox坐标")
        
        # 检查坐标格式并转换为像素坐标（对齐Ultralytics实现）
        bbox_coords = prediction[..., :4]
        max_coord = np.max(np.abs(bbox_coords))
        
        if max_coord <= 1.0:
            # 归一化坐标，需要转换为像素坐标
            prediction[..., 0] *= input_shape[1]  # x_center
            prediction[..., 1] *= input_shape[0]  # y_center
            prediction[..., 2] *= input_shape[1]  # width
            prediction[..., 3] *= input_shape[0]  # height
        else:
            # 已经是像素坐标，无需转换
            pass

        # NMS后处理，传递multi_label和objectness信息
        detections = non_max_suppression(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=multi_label,
            has_objectness=has_objectness
        )

        # 坐标还原到原图尺寸 (统一使用Ultralytics风格)
        if detections and len(detections[0]) > 0:
            if orig_shape is not None and ratio_pad is not None:
                # 使用orig_shape + ratio_pad进行精确的Ultralytics风格还原
                boxes = detections[0][:, :4].copy()

                # 使用scale_boxes函数进行坐标还原
                scaled_boxes = scale_boxes(
                    img1_shape=input_shape,        # 输入尺寸
                    boxes=boxes,                   # 检测框坐标
                    img0_shape=orig_shape,         # 原图尺寸
                    ratio_pad=ratio_pad,           # ratio和padding信息
                    padding=True                   # 使用padding
                )

                # 更新检测结果中的坐标
                detections[0][:, :4] = scaled_boxes
            elif orig_shape is not None:
                # 回退: 有orig_shape但没有ratio_pad,使用简单缩放
                h0, w0 = orig_shape
                h1, w1 = input_shape
                # 从letterbox/resize坐标缩放到原图坐标
                detections[0][:, 0] *= w0 / w1  # x1
                detections[0][:, 1] *= h0 / h1  # y1
                detections[0][:, 2] *= w0 / w1  # x2
                detections[0][:, 3] *= h0 / h1  # y2
            else:
                # 回退: 使用scale因子
                detections[0][:, :4] /= scale

        return detections
    
    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, 
                 iou_thres: Optional[float] = None) -> Tuple[List[np.ndarray], tuple]:
        """
        YOLO推理接口
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            iou_thres (Optional[float]): IoU阈值
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
        """
        return super().__call__(image, conf_thres=conf_thres, iou_thres=iou_thres)


