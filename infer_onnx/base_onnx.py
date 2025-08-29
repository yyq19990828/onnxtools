"""
ONNX模型推理基类

包含:
- BaseOnnx: ONNX模型推理基类，使用Polygraphy懒加载
- 通用的工具函数：xywh2xyxy, clip_boxes, scale_boxes
"""

import numpy as np
import logging
import yaml
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Polygraphy懒加载导入
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import (TrtRunner, 
                                    EngineFromNetwork, EngineFromPath,
                                    NetworkFromOnnxPath, 
                                    SaveEngine)
from polygraphy.comparator import Comparator, CompareFunc

from .infer_utils import get_model_info
from utils.image_processing import preprocess_image


class BaseOnnx(ABC):
    """ONNX模型推理基类 - 使用Polygraphy懒加载"""
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.5, providers: Optional[List[str]] = None):
        """
        初始化ONNX模型推理器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸 (height, width)
            conf_thres (float): 置信度阈值
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.input_shape = input_shape
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建Polygraphy懒加载器
        self._session_loader = SessionFromOnnx(self.onnx_path, providers=self.providers)
        self._runner = None
        self._is_initialized = False
        
        # 延迟初始化的属性
        self.input_name = None
        self.output_names = None
        self._class_names = None
        
        logging.info(f"创建懒加载ONNX推理器: {self.onnx_path}")
    
    @property
    def class_names(self) -> Dict[int, str]:
        """懒加载的类别名称属性"""
        if not self._is_initialized:
            self._ensure_initialized()
        return self._class_names or {}
    
    def _ensure_initialized(self):
        """确保模型已初始化（懒加载）"""
        if not self._is_initialized:
            # 创建Polygraphy运行器
            self._runner = OnnxrtRunner(self._session_loader)
            
            # 激活运行器以获取元数据
            with self._runner:
                # 获取输入输出信息
                input_metadata = self._runner.get_input_metadata()
                self.input_name = list(input_metadata.keys())[0]
                
                # 通过临时会话获取输出名称
                session = self._session_loader()
                self.output_names = [output.name for output in session.get_outputs()]
                
                # 获取类别名称（如果存在配置文件）
                self._class_names = self._load_class_names()
                
                logging.info(f"模型已初始化 - 输入: {self.input_name}, 输出: {self.output_names}")
            
            self._is_initialized = True
    
    def _load_class_names(self) -> Dict[int, str]:
        """使用get_model_info加载类别名称（包括ONNX metadata和配置文件）"""
        try:
            # 使用get_model_info获取模型信息，包括从metadata和配置文件的类别名称
            model_info = get_model_info(self.onnx_path, self.input_shape)
            if model_info and model_info.get('class_names'):
                logging.info("从get_model_info获取到类别名称")
                return model_info['class_names']
        except Exception as e:
            logging.warning(f"get_model_info获取类别名称失败: {e}")
        
        # 回退到原始方法：从配置文件加载
        model_dir = Path(self.onnx_path).parent
        config_files = ['det_config.yaml', 'classes.yaml']
        
        for config_file in config_files:
            config_path = model_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if 'names' in config:
                            names = config['names']
                            if isinstance(names, list):
                                logging.info(f"从配置文件 {config_file} 加载类别名称")
                                return {i: name for i, name in enumerate(names)}
                            elif isinstance(names, dict):
                                logging.info(f"从配置文件 {config_file} 加载类别名称")
                                return names
                except Exception as e:
                    logging.warning(f"无法加载配置文件 {config_path}: {e}")
        
        logging.warning("未找到类别名称，返回空字典")
        return {}
    
    
    @abstractmethod
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[np.ndarray]:
        """后处理抽象方法，子类需要实现"""
        pass
    
    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, **kwargs) -> Tuple[List[np.ndarray], tuple]:
        """
        对图像进行推理（使用Polygraphy懒加载）
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
        """
        # 确保模型已初始化
        self._ensure_initialized()
        
        # 预处理
        preprocess_result = self._preprocess(image)
        if len(preprocess_result) == 3:
            # 兼容旧版本返回值 (input_tensor, scale, original_shape)
            input_tensor, scale, original_shape = preprocess_result
            ratio_pad = None
        else:
            # 新版本返回值 (input_tensor, scale, original_shape, ratio_pad)
            input_tensor, scale, original_shape, ratio_pad = preprocess_result
        
        # 使用Polygraphy运行器进行推理
        with self._runner:
            # 获取输入元数据来检查batch维度
            input_metadata = self._runner.get_input_metadata()
            input_shape = input_metadata[self.input_name].shape
            expected_batch_size = input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else 1
            
            if expected_batch_size > 1 and input_tensor.shape[0] == 1:
                # 如果模型期望batch>1，但输入是batch=1，则复制输入以满足要求
                input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
                logging.debug(f"调整输入batch维度从1到{expected_batch_size}")
            
            # 构造feed_dict并执行推理
            feed_dict = {self.input_name: input_tensor}
            outputs_dict = self._runner.infer(feed_dict)
            
            # 将字典转换为列表格式以保持兼容性
            outputs = [outputs_dict[name] for name in self.output_names]
        
        # 后处理 - 根据子类不同传递不同参数
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        
        # RF-DETR需要完整的outputs，其他模型使用第一个输出
        if type(self).__name__ == 'RFDETROnnx':
            detections = self._postprocess(outputs, effective_conf_thres, **kwargs)
        else:
            prediction = outputs[0]
            detections = self._postprocess(prediction, effective_conf_thres, scale=scale, ratio_pad=ratio_pad, **kwargs)
        
        # 如果输入是多batch但只处理一张图片，只返回第一个batch的结果
        if (expected_batch_size > 1 and len(detections) > 1):
            detections = [detections[0]]
            logging.debug(f"只返回第一个batch的检测结果")
        
        return detections, original_shape
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """预处理图像（实例方法，向后兼容）"""
        return self._preprocess_static(image, self.input_shape)
    
    @staticmethod
    def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
        """预处理图像（静态方法）"""
        return preprocess_image(image, input_shape)
    
    @property
    def engine_dataloader(self):
        """
        为引擎比较创建适配的数据加载器
        
        Returns:
            CustomEngineDataLoader: 使用静态预处理方法的自定义数据加载器
        """
        from .engine_dataloader import create_dataloader_from_detector
        return create_dataloader_from_detector(self)
    
    def compare_engine(
        self, 
        engine_path: Optional[str] = None,
        save_engine: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-3
    ) -> bool:
        """
        使用Polygraphy比较ONNX模型和TensorRT引擎的推理结果
        
        Args:
            engine_path (Optional[str]): TensorRT引擎文件路径。如果为None，则从ONNX构建引擎
            save_engine (bool): 当从ONNX构建引擎时，是否保存引擎文件，默认False
            rtol (float): 相对容差，默认1e-3
            atol (float): 绝对容差，默认1e-3
            
        Returns:
            bool: 比较结果，True表示精度匹配
        """
        # 确保模型已初始化
        self._ensure_initialized()
        
        # 创建ONNX Runner
        onnx_runner = OnnxrtRunner(self._session_loader)
        
        # 创建TensorRT Runner
        if engine_path is not None:
            # 使用现有引擎文件
            from polygraphy import config
            config.USE_TENSORRT_RTX = True  # 启用RTX加速（不启用的trt 8.6.1 会报错
            build_engine = EngineFromPath(engine_path)
            trt_runner = TrtRunner(build_engine)
        else:
            # 从ONNX构建引擎
            build_engine = EngineFromNetwork(NetworkFromOnnxPath(self.onnx_path))
            
            if save_engine:
                # 保存引擎文件
                engine_save_path = str(Path(self.onnx_path).with_suffix('.engine'))
                build_engine = SaveEngine(build_engine, path=engine_save_path)
            
            trt_runner = TrtRunner(build_engine)
        
        # 运行推理比较，使用适配的数据加载器
        run_results = Comparator.run([onnx_runner, trt_runner], data_loader=self.engine_dataloader)
        
        # 比较精度
        accuracy_results = Comparator.compare_accuracy(
            run_results,
            compare_func=CompareFunc.simple(rtol=rtol, atol=atol)
        )
        
        # 返回比较结果
        return bool(accuracy_results), run_results