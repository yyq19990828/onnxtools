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
        self._requested_input_shape = input_shape  # 用户请求的输入形状
        self.input_shape = None  # 实际输入形状，将从模型中读取
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建Polygraphy懒加载器
        self._session_loader = SessionFromOnnx(self.onnx_path, providers=self.providers)
        self._runner = None
        self._is_initialized = False
        
        # 延迟初始化的属性
        self.input_name = None
        self.output_names = None
        self._class_names = None

        self.engine_dataloader = None  # 用于引擎比较的数据加载器
        
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
                
                # 获取实际输入形状
                input_metadata = self._runner.get_input_metadata()
                input_shape_from_model = input_metadata[self.input_name].shape
                if (len(input_shape_from_model) >= 4 and 
                    isinstance(input_shape_from_model[2], int) and input_shape_from_model[2] > 0 and
                    isinstance(input_shape_from_model[3], int) and input_shape_from_model[3] > 0):
                    self.input_shape = (input_shape_from_model[2], input_shape_from_model[3])
                    logging.info(f"从ONNX模型读取到固定输入形状: {self.input_shape}")
                else:
                    self.input_shape = self._requested_input_shape
                    logging.info(f"使用用户指定的输入形状: {self.input_shape}")
                
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
    def _postprocess(self, prediction: List[np.ndarray], conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        Post-process model outputs into final detection/classification results.

        This method must be implemented by all subclasses. It is responsible for
        converting raw model outputs into a standardized detection format.

        Args:
            prediction: Raw model outputs, list of numpy arrays. Format varies by model:
                - YOLO: [batch, num_boxes, 5+num_classes]
                - RT-DETR: [batch, num_boxes, 6]
            conf_thres: Confidence threshold for filtering low-confidence results
            **kwargs: Additional parameters, commonly:
                - iou_thres (float): IoU threshold for NMS (default: self.iou_thres)
                - max_det (int): Maximum number of detections to keep (default: 300)
                - scale (Tuple): Scaling information from preprocessing
                - ratio_pad (Tuple): Padding information for coordinate transformation

        Returns:
            List of post-processed results, one array per batch. Each array has shape:
                - Detection models: [N, 6] where columns are [x1, y1, x2, y2, confidence, class_id]
                - Classification models: [N, 2] where columns are [class_id, confidence]

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If prediction format is invalid

        Example:
            >>> # In YoloOnnx subclass
            >>> def _postprocess(self, prediction, conf_thres, **kwargs):
            ...     iou_thres = kwargs.get('iou_thres', self.iou_thres)
            ...     results = []
            ...     for pred in prediction:
            ...         detections = non_max_suppression(pred, conf_thres, iou_thres)
            ...         results.append(detections)
            ...     return results
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._postprocess() must be implemented by subclass. "
            "This method is responsible for post-processing model outputs. "
            "See BaseOnnx._postprocess docstring for implementation guidance."
        )
    
    def _prepare_inference(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple, Optional[tuple]]:
        """
        Phase 1: Prepare inference by initializing model and preprocessing image.

        This method ensures the model is initialized and performs image preprocessing
        to prepare the input tensor for inference.

        Args:
            image: Input image in BGR format, shape [H, W, C]

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor, shape [1, 3, H, W]
                - scale: Scaling information from preprocessing
                - original_shape: Original image shape (H, W, C)
                - ratio_pad: Padding information (optional, for letterbox resize)

        Note:
            This method can be overridden by subclasses to customize the
            preparation phase behavior.
        """
        # Ensure model is initialized
        self._ensure_initialized()

        # Preprocess image
        preprocess_result = self._preprocess(image)
        if len(preprocess_result) == 3:
            # Compatible with old version return value (input_tensor, scale, original_shape)
            input_tensor, scale, original_shape = preprocess_result
            ratio_pad = None
        else:
            # New version return value (input_tensor, scale, original_shape, ratio_pad)
            input_tensor, scale, original_shape, ratio_pad = preprocess_result

        return input_tensor, scale, original_shape, ratio_pad

    def _execute_inference(self, input_tensor: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """
        Phase 2: Execute model inference with input tensor.

        This method performs the actual model inference using Polygraphy runner.
        It handles batch dimension adjustments and executes the ONNX model.

        Args:
            input_tensor: Preprocessed input tensor, shape [1, 3, H, W]

        Returns:
            Tuple containing:
                - outputs: List of model output arrays
                - expected_batch_size: Expected batch size from model metadata

        Note:
            This method can be overridden by subclasses to customize the
            inference execution behavior.
        """
        # Execute inference using Polygraphy runner
        with self._runner:
            # Get input metadata to check batch dimension
            input_metadata = self._runner.get_input_metadata()
            input_shape = input_metadata[self.input_name].shape
            expected_batch_size = input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else 1

            if expected_batch_size > 1 and input_tensor.shape[0] == 1:
                # If model expects batch>1 but input is batch=1, repeat input to meet requirement
                input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
                logging.debug(f"调整输入batch维度从1到{expected_batch_size}")

            # Construct feed_dict and execute inference
            feed_dict = {self.input_name: input_tensor}
            outputs_dict = self._runner.infer(feed_dict)

            # Convert dictionary to list format to maintain compatibility
            outputs = [outputs_dict[name] for name in self.output_names]

        return outputs, expected_batch_size

    def _finalize_inference(
        self,
        outputs: List[np.ndarray],
        expected_batch_size: int,
        scale: float,
        ratio_pad: Optional[tuple],
        conf_thres: Optional[float],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Phase 3: Finalize inference by post-processing outputs and filtering results.

        This method performs post-processing on model outputs and filters results
        based on batch size and confidence threshold.

        Args:
            outputs: List of model output arrays from inference
            expected_batch_size: Expected batch size from model metadata
            scale: Scaling information from preprocessing
            ratio_pad: Padding information (optional, for letterbox resize)
            conf_thres: Confidence threshold for filtering results
            **kwargs: Additional parameters for post-processing

        Returns:
            List of final detection results, one array per batch

        Note:
            This method can be overridden by subclasses to customize the
            finalization phase behavior.
        """
        # Post-process - pass different parameters based on subclass
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres

        # RF-DETR needs full outputs, other models use first output
        if type(self).__name__ == 'RFDETROnnx':
            detections = self._postprocess(outputs, effective_conf_thres, **kwargs)
        else:
            prediction = outputs[0]
            detections = self._postprocess(prediction, effective_conf_thres, scale=scale, ratio_pad=ratio_pad, **kwargs)

        # If input is multi-batch but only processing one image, return only first batch result
        if (expected_batch_size > 1 and len(detections) > 1):
            detections = [detections[0]]
            logging.debug(f"只返回第一个batch的检测结果")

        return detections

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
        # Phase 1: Prepare inference
        input_tensor, scale, original_shape, ratio_pad = self._prepare_inference(image)

        # Phase 2: Execute inference
        outputs, expected_batch_size = self._execute_inference(input_tensor)

        # Phase 3: Finalize inference
        detections = self._finalize_inference(outputs, expected_batch_size, scale, ratio_pad, conf_thres, **kwargs)

        return detections, original_shape
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """预处理图像（实例方法，向后兼容）"""
        return self._preprocess_static(image, self.input_shape)
    
    @staticmethod
    @abstractmethod
    def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, tuple]:
        """
        Static preprocessing method for image transformation.

        This static method must be implemented by all subclasses. It performs
        image preprocessing independent of instance state.

        Args:
            image: Input image in BGR format (OpenCV default), shape [H, W, C]
            input_shape: Target input size (height, width), e.g., (640, 640)

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor, shape [1, 3, H, W], range [0, 1]
                  Format: NCHW (batch, channels, height, width), RGB order
                - scale: Scaling information for coordinate transformation, format varies:
                  * Letterbox: dict with 'scale', 'pad_w', 'pad_h' keys
                  * Simple resize: tuple (scale_x, scale_y)

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If image dimensions are invalid

        Example:
            >>> # In RTDETROnnx subclass
            >>> @staticmethod
            >>> @abstractmethod
            >>> def _preprocess_static(image, input_shape):
            ...     # Letterbox resize (keep aspect ratio)
            ...     resized, scale = letterbox_resize(image, input_shape)
            ...     # BGR to RGB
            ...     rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ...     # Normalize to [0, 1]
            ...     normalized = rgb_image.astype(np.float32) / 255.0
            ...     # NCHW format
            ...     input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
            ...     return input_tensor, scale
        """
        raise NotImplementedError(
            f"BaseOnnx._preprocess_static() must be implemented by subclass. "
            "This static method is responsible for image preprocessing. "
            "See BaseOnnx._preprocess_static docstring for implementation guidance."
        )
    
    def create_engine_dataloader(self, **kwargs):
        """
        为引擎比较创建适配的数据加载器，并自动赋值给engine_dataloader属性
        
        Args:
            image_paths: 图片路径列表，可以是文件夹路径或具体图片路径
            iterations: 迭代次数
        
        Returns:
            CustomEngineDataLoader: 使用静态预处理方法的自定义数据加载器
        """
        from .engine_dataloader import create_dataloader_from_detector
        dataloader = create_dataloader_from_detector(self, **kwargs)
        # 直接赋值给私有属性，这样外部脚本调用后就会自动设置好
        self.engine_dataloader = dataloader
        return dataloader
    
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
            save_engine (bool): 当从ONNX构建引擎时，是否保存引擎文件(fp32)，默认False
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
        # 如果没有设置自定义数据加载器，则创建默认的
        # if self.engine_dataloader is None:
        #     self.create_engine_dataloader()  # 这会自动设置self.engine_dataloader
        
        run_results = Comparator.run([onnx_runner, trt_runner], data_loader=self.engine_dataloader)
        
        # 比较精度
        accuracy_results = Comparator.compare_accuracy(
            run_results,
            compare_func=CompareFunc.simple(check_error_stat='quantile',
                                            error_quantile=0.95,
                                            rtol=rtol, atol=atol)
                                            )
        
        # 返回比较结果
        return bool(accuracy_results), run_results