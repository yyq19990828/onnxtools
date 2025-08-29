#!/usr/bin/env python3
"""
自定义引擎比较数据加载器

使用静态预处理方法，避免嵌套调用，支持真实数据集进行ONNX vs TensorRT引擎比较
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple, TYPE_CHECKING
from collections import OrderedDict

from polygraphy.comparator.data_loader import DataLoader
from polygraphy.logger import G_LOGGER

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from .infer_models import BaseOnnx, YoloOnnx, RTDETROnnx, RFDETROnnx


class CustomEngineDataLoader(DataLoader):
    """
    自定义引擎比较数据加载器，继承自Polygraphy DataLoader
    
    直接使用检测器类的静态预处理方法，避免嵌套调用和代码冗余
    支持使用真实图像数据集进行模型比较
    """
    
    def __init__(
        self,
        detector_class,  # 检测器类（非实例）
        input_shape: Tuple[int, int],
        input_name: str,
        image_paths: Optional[List[Union[str, Path]]] = None,
        images: Optional[List[np.ndarray]] = None,
        iterations: int = 10,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        初始化自定义数据加载器
        
        Args:
            detector_class: 检测器类（如YoloOnnx, RTDETROnnx等）
            input_shape (Tuple[int, int]): 输入尺寸 (height, width)
            input_name (str): 输入张量名称
            image_paths (Optional[List[Union[str, Path]]]): 图像文件路径列表
            images (Optional[List[np.ndarray]]): 图像数组列表
            iterations (int): 迭代次数，如果使用随机数据则有效
            preprocess_kwargs (Optional[Dict[str, Any]]): 预处理函数的额外参数
            **kwargs: 传递给父类DataLoader的其他参数
        """
        # 调用父类初始化
        super().__init__(iterations=iterations, **kwargs)
        
        self.detector_class = detector_class
        self.input_shape = input_shape
        self.input_name = input_name
        self.image_paths = image_paths or ['data/sample.jpg']
        self.images = images or []
        self.preprocess_kwargs = preprocess_kwargs or {}
        self._cached_images = {}  # 缓存加载的图像
        
        # 确定数据来源
        if self.image_paths:
            self.data_source = "paths"
            self.data_length = len(self.image_paths)
        elif self.images:
            self.data_source = "arrays"
            self.data_length = len(self.images)
        else:
            self.data_source = "synthetic"
            self.data_length = iterations
        
        G_LOGGER.info(f"CustomEngineDataLoader 初始化: 检测器={detector_class.__name__}, 数据源={self.data_source}, 数据长度={self.data_length}")
    
    def _load_image(self, index: int) -> np.ndarray:
        """
        根据索引加载图像
        
        Args:
            index (int): 图像索引
            
        Returns:
            np.ndarray: 加载的图像 (BGR格式)
        """
        if self.data_source == "paths":
            # 从文件路径加载
            if index in self._cached_images:
                return self._cached_images[index]
            
            image_path = self.image_paths[index % len(self.image_paths)]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            self._cached_images[index] = image
            return image
            
        elif self.data_source == "arrays":
            # 从图像数组列表获取
            return self.images[index % len(self.images)]
            
        else:
            # 生成合成数据 (随机图像)
            height, width = self.input_shape
            # 生成随机RGB图像，然后转为BGR
            random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            return random_image
    
    def __getitem__(self, index: int) -> OrderedDict:
        """
        重写__getitem__方法，使用检测器类的静态预处理方法
        
        Args:
            index (int): 数据索引
            
        Returns:
            OrderedDict: 输入数据字典 {input_name: preprocessed_tensor}
        """
        if index >= self.iterations:
            raise IndexError(f"索引 {index} 超出迭代次数 {self.iterations}")
        
        # 获取或生成图像
        if self.data_source == "synthetic":
            # 使用父类的随机数据生成
            return super().__getitem__(index)
        else:
            # 使用真实图像数据
            image = self._load_image(index)
            
            # 使用检测器类的静态预处理方法
            preprocess_result = self.detector_class._preprocess_static(
                image, self.input_shape, **self.preprocess_kwargs
            )
            
            # 处理不同预处理方法的返回值格式
            if len(preprocess_result) == 3:
                input_tensor, _, _ = preprocess_result
            else:
                input_tensor, _, _, _ = preprocess_result
            
            # 构造输出字典
            feed_dict = OrderedDict()
            feed_dict[self.input_name] = input_tensor
            
            return feed_dict
    
    def __len__(self) -> int:
        """返回数据加载器长度"""
        return self.iterations


def create_engine_dataloader(
    detector_class,
    input_shape: Tuple[int, int],
    input_name: str,
    image_paths: Optional[List[Union[str, Path]]] = None,
    images: Optional[List[np.ndarray]] = None,
    iterations: int = 10,
    **kwargs
) -> CustomEngineDataLoader:
    """
    工厂函数：创建引擎比较数据加载器
    
    Args:
        detector_class: 检测器类（如YoloOnnx, RTDETROnnx等）
        input_shape (Tuple[int, int]): 输入尺寸 (height, width)
        input_name (str): 输入张量名称
        image_paths (Optional[List[Union[str, Path]]]): 图像路径列表
        images (Optional[List[np.ndarray]]): 图像数组列表
        iterations (int): 迭代次数
        **kwargs: 其他参数，包括preprocess_kwargs
        
    Returns:
        CustomEngineDataLoader: 数据加载器实例
    """
    return CustomEngineDataLoader(
        detector_class=detector_class,
        input_shape=input_shape,
        input_name=input_name,
        image_paths=image_paths,
        images=images,
        iterations=iterations,
        **kwargs
    )


# 便捷函数，从检测器实例创建DataLoader
def create_dataloader_from_detector(
    detector_instance,
    image_paths: Optional[List[Union[str, Path]]] = None,
    images: Optional[List[np.ndarray]] = None,
    iterations: int = 10,
    **kwargs
) -> CustomEngineDataLoader:
    """
    从检测器实例创建数据加载器（便捷函数）
    
    Args:
        detector_instance: 检测器实例
        image_paths (Optional[List[Union[str, Path]]]): 图像路径列表
        images (Optional[List[np.ndarray]]): 图像数组列表
        iterations (int): 迭代次数
        **kwargs: 其他参数
        
    Returns:
        CustomEngineDataLoader: 数据加载器实例
    """
    # 确保检测器已初始化
    detector_instance._ensure_initialized()
    
    # 提取特定于不同检测器类的预处理参数
    preprocess_kwargs = {}
    if hasattr(detector_instance, 'use_ultralytics_preprocess'):
        preprocess_kwargs['use_ultralytics_preprocess'] = detector_instance.use_ultralytics_preprocess
    
    return CustomEngineDataLoader(
        detector_class=detector_instance.__class__,
        input_shape=detector_instance.input_shape,
        input_name=detector_instance.input_name,
        image_paths=image_paths,
        images=images,
        iterations=iterations,
        preprocess_kwargs=preprocess_kwargs,
        **kwargs
    )