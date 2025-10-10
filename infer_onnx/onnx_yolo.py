"""
YOLO模型ONNX推理类

包含:
- YoloOnnx: 传统YOLO模型ONNX推理类 (原DetONNX)
支持YOLOv5、YOLOv8等使用NMS后处理的YOLO模型
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

from .onnx_base import BaseOnnx
from .infer_utils import scale_boxes
from utils.image_processing import preprocess_image
from utils.nms import non_max_suppression


class YoloOnnx(BaseOnnx):
    """
    传统YOLO模型ONNX推理类 (原DetONNX)
    
    支持YOLOv5、YOLOv8等使用NMS后处理的YOLO模型
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.5, iou_thres: float = 0.5, 
                 multi_label: bool = True, use_ultralytics_preprocess: bool = True,
                 has_objectness: bool = False, providers: Optional[List[str]] = None):
        """
        初始化YOLO检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            multi_label (bool): 是否允许多标签检测，默认True
            use_ultralytics_preprocess (bool): 是否使用Ultralytics兼容的预处理
            has_objectness (bool): 模型是否有objectness分支，默认False（适应现代YOLO）
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
        """
        super().__init__(onnx_path, input_shape, conf_thres, providers)
        self.iou_thres = iou_thres
        self.multi_label = multi_label
        self.use_ultralytics_preprocess = use_ultralytics_preprocess
        self.has_objectness = has_objectness
        # YoloOnnx类固定使用yolo模型类型
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple, Optional[tuple]]:
        """
        YOLO预处理，支持Ultralytics兼容模式（实例方法，向后兼容）
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (input_tensor, scale, original_shape, ratio_pad)
        """
        return self._preprocess_static(image, self.input_shape, self.use_ultralytics_preprocess)
    
    @staticmethod
    def _preprocess_static(
        image: np.ndarray, 
        input_shape: Tuple[int, int], 
        use_ultralytics_preprocess: bool = False
    ) -> Tuple[np.ndarray, float, tuple, Optional[tuple]]:
        """
        YOLO预处理静态方法
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            input_shape (Tuple[int, int]): 输入尺寸
            use_ultralytics_preprocess (bool): 是否使用Ultralytics兼容预处理
            
        Returns:
            Tuple: (input_tensor, scale, original_shape, ratio_pad)
        """
        if use_ultralytics_preprocess:
            # 使用Ultralytics兼容的预处理，返回ratio_pad信息
            from utils.image_processing import UltralyticsLetterBox
            letterbox = UltralyticsLetterBox(new_shape=input_shape)
            input_tensor, scale, original_shape, ratio_pad = letterbox(image)
            return input_tensor, scale, original_shape, (((scale, scale), ratio_pad))
        else:
            # 使用原始预处理方法
            input_tensor, scale, original_shape = preprocess_image(image, input_shape)
            return input_tensor, scale, original_shape, None
    
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, scale: float = 1.0, 
                    ratio_pad: Optional[tuple] = None, iou_thres: Optional[float] = None) -> List[np.ndarray]:
        """
        YOLO模型后处理，包含NMS
        
        Args:
            prediction (np.ndarray): 模型原始输出
            conf_thres (float): 置信度阈值
            scale (float): 图像缩放因子
            ratio_pad (Optional[tuple]): Ratio和padding信息 (((ratio_w, ratio_h), (pad_w, pad_h)))
            iou_thres (Optional[float]): IoU阈值
            
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
            prediction[..., 0] *= self.input_shape[1]  # x_center
            prediction[..., 1] *= self.input_shape[0]  # y_center
            prediction[..., 2] *= self.input_shape[1]  # width
            prediction[..., 3] *= self.input_shape[0]  # height
        else:
            # 已经是像素坐标，无需转换
            pass
        
        # NMS后处理，传递multi_label和objectness信息
        effective_iou_thres = iou_thres if iou_thres is not None else self.iou_thres
        detections = non_max_suppression(
            prediction, 
            conf_thres=conf_thres, 
            iou_thres=effective_iou_thres,
            multi_label=self.multi_label,
            has_objectness=self.has_objectness
        )
        
        # 使用Ultralytics风格的坐标还原
        if detections and len(detections[0]) > 0:
            if ratio_pad is not None and self.use_ultralytics_preprocess:
                # 使用scale_boxes进行精确的坐标还原
                # 从输入尺寸坐标还原到原图坐标
                boxes = detections[0][:, :4].copy()
                original_shape = (int(self.input_shape[0] / ratio_pad[0][0]), 
                                int(self.input_shape[1] / ratio_pad[0][1]))
                
                # 使用scale_boxes函数进行坐标还原
                scaled_boxes = scale_boxes(
                    img1_shape=self.input_shape,  # 输入尺寸
                    boxes=boxes,                   # 检测框坐标
                    img0_shape=original_shape,     # 原图尺寸
                    ratio_pad=ratio_pad,           # ratio和padding信息
                    padding=True                   # 使用padding
                )
                
                # 更新检测结果中的坐标
                detections[0][:, :4] = scaled_boxes
            else:
                # 回退到简单的缩放方法
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


# 为了向后兼容，保留旧的类名
DetONNX = YoloOnnx