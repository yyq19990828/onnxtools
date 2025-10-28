#!/usr/bin/env python3
"""
自定义引擎比较数据加载器

使用静态预处理方法，避免嵌套调用，支持真实数据集进行ONNX vs TensorRT引擎比较
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple, TYPE_CHECKING
from collections import OrderedDict
import glob

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
        self.image_paths = self._process_image_paths(image_paths or [''])
        self.images = images or []
        self.preprocess_kwargs = preprocess_kwargs or {}
        self._cached_images = {}  # 缓存加载的图像
        
        # 确定数据来源
        if self.image_paths and len(self.image_paths) > 0:
            self.data_source = "paths"
            self.data_length = len(self.image_paths)
        elif self.images and len(self.images) > 0:
            self.data_source = "arrays"
            self.data_length = len(self.images)
        else:
            self.data_source = "synthetic"
            self.data_length = iterations
        
        G_LOGGER.info(f"CustomEngineDataLoader 初始化: 检测器={detector_class.__name__}, 数据源={self.data_source}, 数据长度={self.data_length}")
    
    def _process_image_paths(self, paths: List[Union[str, Path]]) -> List[Path]:
        """
        处理输入路径列表，支持文件夹路径和具体图片路径
        
        Args:
            paths: 输入路径列表，可以是文件夹或具体图片路径
            
        Returns:
            List[Path]: 处理后的图片文件路径列表
        """
        processed_paths = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for path in paths:
            path = Path(path)
            
            if path.is_file():
                # 如果是文件，检查是否是支持的图片格式
                if path.suffix.lower() in supported_extensions:
                    processed_paths.append(path)
                else:
                    G_LOGGER.warning(f"跳过不支持的文件格式: {path}")
            elif path.is_dir():
                # 如果是文件夹，扫描其中的图片文件
                found_images = []
                for ext in supported_extensions:
                    pattern = str(path / f"*{ext}")
                    found_images.extend(glob.glob(pattern))
                    pattern = str(path / f"*{ext.upper()}")
                    found_images.extend(glob.glob(pattern))
                
                if found_images:
                    # 排序确保一致性
                    found_images.sort()
                    processed_paths.extend([Path(img) for img in found_images])
                    G_LOGGER.info(f"从文件夹 {path} 中找到 {len(found_images)} 张图片")
                else:
                    G_LOGGER.warning(f"文件夹 {path} 中未找到支持的图片文件")
            else:
                G_LOGGER.warning(f"路径不存在: {path}")
        
        if not processed_paths:
            G_LOGGER.warning("未找到任何有效的图片文件，将使用合成数据")
        
        return processed_paths
    
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
            
            # 检测模型期望的batch size并调整输入张量
            expected_batch_size = self._get_expected_batch_size()
            if expected_batch_size > 1 and input_tensor.shape[0] == 1:
                # 重复数据以匹配期望的batch size
                input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
                logging.debug(f"数据加载器：调整输入batch维度从1到{expected_batch_size}")
            
            # 构造输出字典
            feed_dict = OrderedDict()
            feed_dict[self.input_name] = input_tensor
            
            return feed_dict
    
    def __len__(self) -> int:
        """返回数据加载器长度"""
        return self.iterations
    
    def _get_expected_batch_size(self) -> int:
        """
        获取模型期望的batch size
        
        Returns:
            int: 期望的batch size
        """
        try:
            # 检查是否有已缓存的batch size信息
            if hasattr(self, '_cached_batch_size'):
                return self._cached_batch_size
                
            # 尝试通过检测器类的静态方法获取batch size信息
            if hasattr(self.detector_class, '_get_batch_size_from_onnx'):
                batch_size = self.detector_class._get_batch_size_from_onnx()
                self._cached_batch_size = batch_size
                return batch_size
                
            # # 对于RFDETR模型，已知batch size是4
            # if self.detector_class.__name__ == 'RFDETROnnx':
            #     self._cached_batch_size = 4
            #     logging.debug(f"数据加载器使用RFDETR已知batch size: 4")
            #     return 4
                
            # 默认情况
            self._cached_batch_size = 1
            return 1
            
        except Exception as e:
            logging.debug(f"无法检测期望batch size，使用默认值1: {e}")
            return 1


def create_engine_dataloader(
    detector_class,
    input_shape: Tuple[int, int],
    input_name: str,
    image_paths: Optional[List[Union[str, Path]]] = None,
    images: Optional[List[np.ndarray]] = None,
    iterations: int = 4,
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
    iterations: int = 2,
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