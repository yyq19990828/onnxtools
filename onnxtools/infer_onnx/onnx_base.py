"""
ONNX Runtime 模型推理基类

包含:
- BaseOrt: ONNX Runtime模型推理基类，使用Polygraphy懒加载
- 通用的工具函数：xywh2xyxy, clip_boxes, scale_boxes
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from .result import Result
from pathlib import Path

# Polygraphy懒加载导入
from polygraphy.backend.onnxrt import OnnxrtRunner
from polygraphy.backend.trt import (TrtRunner, 
                                    EngineFromNetwork, EngineFromPath,
                                    NetworkFromOnnxPath, 
                                    SaveEngine)
from polygraphy.comparator import Comparator, CompareFunc


class BaseORT(ABC):
    """ONNX Runtime模型推理基类 - 使用Polygraphy懒加载"""
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.5, providers: Optional[List[str]] = None,
                 det_config_path: Optional[str] = None):
        """
        初始化ONNX模型推理器

        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸 (height, width)
            conf_thres (float): 置信度阈值
            providers (Optional[List[str]]): ONNX Runtime执行提供程序
            det_config_path (Optional[str]): 检测配置文件路径(可选)
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self._requested_input_shape = input_shape  # 用户请求的输入形状
        self.input_shape = None  # 实际输入形状，将从模型中读取
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.det_config_path = det_config_path  # 保存配置文件路径
        
        # 创建ONNX Runtime会话（立即创建并缓存，后续复用）
        import onnxruntime
        self._onnx_session = onnxruntime.InferenceSession(self.onnx_path, providers=self.providers)
        logging.info(f"ONNX Runtime会话已创建: {self._onnx_session.get_providers()}")
        
        # 使用纯onnx库获取模型信息（不创建ORT会话，轻量级）
        model_info = self._get_model_info()

        # 从模型信息中提取属性
        self.input_name = model_info.get('input_name')
        self.output_names = model_info.get('output_names')
        self.class_names = model_info.get('class_names', {})

        # 从模型信息确定input_shape和expected_batch_size
        model_input_shape = model_info.get('input_shape')
        if model_input_shape and len(model_input_shape) >= 4:
            # 提取batch size (第0维)
            if isinstance(model_input_shape[0], int) and model_input_shape[0] > 0:
                self._expected_batch_size = model_input_shape[0]
            else:
                self._expected_batch_size = 1

            # 提取height和width (第2、3维)
            if (isinstance(model_input_shape[2], int) and model_input_shape[2] > 0 and
                isinstance(model_input_shape[3], int) and model_input_shape[3] > 0):
                self.input_shape = (model_input_shape[2], model_input_shape[3])
                logging.info(f"从ONNX模型metadata读取到固定输入形状: {self.input_shape}, batch_size: {self._expected_batch_size}")
            else:
                self.input_shape = self._requested_input_shape
                logging.info(f"模型输入形状为动态，使用用户指定形状: {self.input_shape}, batch_size: {self._expected_batch_size}")
        else:
            self.input_shape = self._requested_input_shape
            self._expected_batch_size = 1
            logging.info(f"使用默认配置: input_shape={self.input_shape}, batch_size={self._expected_batch_size}")

        self.engine_dataloader = None  # 用于引擎比较的数据加载器

        logging.info(f"创建ONNX推理器: {self.onnx_path}")

    def _get_model_info(self) -> Dict[str, any]:
        """
        使用纯onnx库获取模型信息（不创建ORT会话，轻量级）

        Returns:
            Dict包含:
            - input_name: str
            - output_names: List[str]
            - input_shape: List (可能包含动态维度)
            - class_names: Dict[int, str]
        """
        import onnx
        import json

        result = {
            'input_name': None,
            'output_names': [],
            'input_shape': None,
            'class_names': {}
        }

        try:
            # 加载ONNX模型（只读取结构，不创建推理会话）
            model = onnx.load(self.onnx_path)
            graph = model.graph

            # 获取输入信息
            if graph.input:
                input_info = graph.input[0]
                result['input_name'] = input_info.name

                # 解析输入shape
                input_shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        input_shape.append(dim.dim_param)  # 动态维度
                    else:
                        input_shape.append(dim.dim_value)  # 固定维度
                result['input_shape'] = input_shape

            # 获取输出信息
            result['output_names'] = [output.name for output in graph.output]

            # 读取类别名称从metadata
            custom_metadata = {}
            for prop in model.metadata_props:
                custom_metadata[prop.key] = prop.value

            # 尝试解析names字段
            if 'names' in custom_metadata:
                try:
                    # 首先尝试JSON解析
                    names_data = json.loads(custom_metadata['names'])
                except (json.JSONDecodeError, TypeError):
                    try:
                        # 如果JSON解析失败，尝试Python字典的eval解析（Ultralytics格式）
                        names_data = eval(custom_metadata['names'])
                    except Exception:
                        names_data = None

                # 处理不同的names格式
                if isinstance(names_data, dict):
                    try:
                        result['class_names'] = {int(k): str(v) for k, v in names_data.items()}
                        logging.info(f"从ONNX模型metadata读取到类别名称 (纯onnx库)")
                    except (ValueError, TypeError):
                        result['class_names'] = {i: str(v) for i, v in enumerate(names_data.values())}
                        logging.info(f"从ONNX模型metadata读取到类别名称 (纯onnx库)")
                elif isinstance(names_data, list):
                    result['class_names'] = {i: str(name) for i, name in enumerate(names_data)}
                    logging.info(f"从ONNX模型metadata读取到类别名称 (纯onnx库)")

            logging.info(f"从ONNX模型读取信息: input={result['input_name']}, outputs={result['output_names']}")

        except Exception as e:
            logging.warning(f"从ONNX模型读取信息失败: {e}")

        # 如果metadata中没有类别名称，回退到配置文件
        if not result['class_names']:
            result['class_names'] = self._load_class_names_from_config()

        return result

    def _load_class_names_from_config(self) -> Dict[int, str]:
        """
        从配置加载类别名称（回退方案）

        优先级:
        1. self.det_config_path显式指定的外部YAML文件
        2. onnxtools.config.DET_CLASS_NAMES 硬编码常量（默认）

        Returns:
            Dict[int, str]: 类别ID到类别名称的映射
        """
        try:
            # 如果指定了外部配置文件，使用load_det_config加载
            if self.det_config_path:
                from onnxtools.config import load_det_config
                config = load_det_config(self.det_config_path)
                class_names = config.get('class_names')

                if not class_names:
                    logging.warning("外部配置中没有class_names字段")
                    return {}

                if isinstance(class_names, list):
                    logging.info(f"从外部配置加载 {len(class_names)} 个类别")
                    return {i: name for i, name in enumerate(class_names)}
                else:
                    logging.warning(f"class_names字段格式错误: {type(class_names)}")
                    return {}

            # 默认直接使用硬编码常量
            from onnxtools.config import DET_CLASS_NAMES
            logging.info(f"使用硬编码类别名称: {len(DET_CLASS_NAMES)} 个类别")
            return {i: name for i, name in enumerate(DET_CLASS_NAMES)}

        except Exception as e:
            logging.error(f"加载配置失败: {e}")
            return {}

    @staticmethod
    @abstractmethod
    def preprocess(image: np.ndarray, input_shape: Tuple[int, int], **kwargs) -> Tuple:
        """
        静态预处理方法，将图像转换为模型输入格式

        Args:
            image: 输入图像，BGR格式，shape [H, W, C]
            input_shape: 目标输入尺寸，例如 (640, 640)
            **kwargs: 额外参数（子类特定）

        Returns:
            Tuple: 预处理结果，至少包含 (input_tensor, scale, original_shape)
                   可选返回 (input_tensor, scale, original_shape, ratio_pad)

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"{__class__.__name__}.preprocess() must be implemented by subclass"
        )

    @staticmethod
    @abstractmethod
    def postprocess(prediction: np.ndarray, input_shape: Tuple[int, int],
                   conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        静态后处理方法，将模型输出转换为检测结果

        Args:
            prediction: 模型原始输出
            input_shape: 输入图像尺寸
            conf_thres: 置信度阈值
            **kwargs: 额外参数（子类特定，如 iou_thres, orig_shape 等）

        Returns:
            List[np.ndarray]: 检测结果列表，每个元素为 [N, 6] 数组
                             格式: [x1, y1, x2, y2, confidence, class_id]

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"{__class__.__name__}.postprocess() must be implemented by subclass"
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
        # Call subclass's static preprocess method
        preprocess_result = self.preprocess(image, self.input_shape)

        # Use subclass name to determine return value format
        class_name = self.__class__.__name__
        if class_name in ['RtdetrORT', 'RfdetrORT']:
            # RT-DETR and RF-DETR return 3-tuple: (input_tensor, scale, original_shape)
            input_tensor, scale, original_shape = preprocess_result
            ratio_pad = None
        else:
            # YOLO class return 4-tuple: (input_tensor, scale, original_shape, ratio_pad)
            input_tensor, scale, original_shape, ratio_pad = preprocess_result

        return input_tensor, scale, original_shape, ratio_pad

    def _execute_inference(self, input_tensor: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """
        Phase 2: Execute model inference with input tensor.

        This method performs the actual model inference using Polygraphy runner
        with proper context management to avoid state leakage.

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
        # 使用缓存的 batch size，避免每次推理都查询元数据
        expected_batch_size = self._expected_batch_size

        if expected_batch_size > 1 and input_tensor.shape[0] == 1:
            # If model expects batch>1 but input is batch=1, repeat input to meet requirement
            input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
            logging.debug(f"调整输入batch维度从1到{expected_batch_size}")

        # 使用缓存的ONNX Runtime会话执行推理（直接调用，无需激活/停用）
        feed_dict = {self.input_name: input_tensor}
        outputs = self._onnx_session.run(self.output_names, feed_dict)

        return outputs, expected_batch_size

    def _finalize_inference(
        self,
        outputs: List[np.ndarray],
        expected_batch_size: int,
        scale: float,
        ratio_pad: Optional[tuple],
        conf_thres: Optional[float],
        orig_shape: Optional[tuple] = None,
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
        # Post-process - call static methods with explicit parameters
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        class_name = self.__class__.__name__

        # Different models have different postprocess signatures
        if class_name == 'RfdetrORT':
            # RF-DETR needs full outputs (List[np.ndarray])
            detections = self.postprocess(
                outputs, self.input_shape, effective_conf_thres,
                orig_shape=orig_shape, **kwargs
            )
        elif class_name == 'RtdetrORT':
            # RT-DETR uses first output only
            detections = self.postprocess(
                outputs[0], self.input_shape, effective_conf_thres,
                orig_shape=orig_shape, **kwargs
            )
        else:
            # YOLO and other models need additional NMS parameters
            detections = self.postprocess(
                outputs[0], self.input_shape, effective_conf_thres,
                self.iou_thres, self.multi_label, self.has_objectness,
                scale=scale, ratio_pad=ratio_pad, orig_shape=orig_shape
            )

        # If input is multi-batch but only processing one image, return only first batch result
        if (expected_batch_size > 1 and len(detections) > 1):
            detections = [detections[0]]
            logging.debug(f"只返回第一个batch的检测结果")

        return detections

    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, **kwargs) -> Result:
        """
        对图像进行推理（使用Polygraphy懒加载）

        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            **kwargs: 其他参数

        Returns:
            Result: 包装的检测结果对象
        """
        # Phase 1: Prepare inference
        input_tensor, scale, original_shape, ratio_pad = self._prepare_inference(image)

        # Phase 2: Execute inference
        outputs, expected_batch_size = self._execute_inference(input_tensor)

        # Phase 3: Finalize inference
        detections = self._finalize_inference(outputs, expected_batch_size, scale, ratio_pad, conf_thres, orig_shape=original_shape, **kwargs)

        # Convert list format to Result object (T014)
        # detections is List[np.ndarray] where each array has shape [N, 6] (x1,y1,x2,y2,conf,class_id)
        if len(detections) > 0 and len(detections[0]) > 0:
            det = detections[0]  # Take first batch
            boxes = det[:, :4].astype(np.float32)
            scores = det[:, 4].astype(np.float32)
            class_ids = det[:, 5].astype(np.int32)
        else:
            # Empty detections
            boxes = None
            scores = None
            class_ids = None

        return Result(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            orig_img=image,
            orig_shape=original_shape,
            names=self.class_names
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
        
        # 创建ONNX Runner
        onnx_runner = OnnxrtRunner(sess=self._onnx_session)
        
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