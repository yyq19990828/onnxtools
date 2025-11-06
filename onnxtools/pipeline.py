"""Pipeline module for vehicle and license plate detection.

This module provides both the new InferencePipeline class (recommended)
and legacy functions for backward compatibility.

Recommended usage:
    from onnxtools import InferencePipeline
    pipeline = InferencePipeline(model_type='rtdetr', model_path='model.onnx')
    result_img, output_data = pipeline(image)
"""

import numpy as np
import yaml
import logging
import warnings
from typing import Tuple, List, Dict, Any, Optional

from onnxtools.utils.drawing import draw_detections

# Import annotator functionality
try:
    from .utils.annotator_factory import AnnotatorType, AnnotatorPipeline
    from .utils.visualization_preset import VisualizationPreset
    ANNOTATOR_AVAILABLE = True
except ImportError:
    ANNOTATOR_AVAILABLE = False
    logging.warning("Annotator functionality not available. Using legacy drawing.")

def create_annotator_pipeline(args):
    """Create annotator pipeline based on command line arguments.

    .. deprecated:: 0.2.0
        Use InferencePipeline class instead. This function will be removed in v0.3.0.

    Args:
        args: Command line arguments containing annotator configuration

    Returns:
        AnnotatorPipeline or None if annotators not available/configured
    """
    warnings.warn(
        "create_annotator_pipeline() is deprecated and will be removed in v0.3.0. "
        "Use InferencePipeline class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not ANNOTATOR_AVAILABLE:
        return None

    # Use preset if specified
    if hasattr(args, 'annotator_preset') and args.annotator_preset:
        try:
            preset = VisualizationPreset.from_yaml(args.annotator_preset)
            pipeline = preset.create_pipeline()
            logging.info(f"Using annotator preset: {args.annotator_preset}")
            return pipeline
        except Exception as e:
            logging.warning(f"Failed to load preset '{args.annotator_preset}': {e}")
            return None

    # Use custom annotator types if specified
    if hasattr(args, 'annotator_types') and args.annotator_types:
        pipeline = AnnotatorPipeline()
        for ann_type_str in args.annotator_types:
            try:
                ann_type = AnnotatorType(ann_type_str)

                # Build config based on annotator type and args
                if ann_type == AnnotatorType.ROUND_BOX:
                    config = {
                        'thickness': args.box_thickness if hasattr(args, 'box_thickness') else 2,
                        'roundness': args.roundness if hasattr(args, 'roundness') else 0.3
                    }
                elif ann_type == AnnotatorType.BLUR:
                    config = {
                        'kernel_size': args.blur_kernel_size if hasattr(args, 'blur_kernel_size') else 15
                    }
                elif ann_type in [AnnotatorType.BOX, AnnotatorType.BOX_CORNER]:
                    config = {
                        'thickness': args.box_thickness if hasattr(args, 'box_thickness') else 2
                    }
                else:
                    config = {}

                pipeline.add(ann_type, config)
                logging.debug(f"Added annotator: {ann_type.value}")
            except ValueError:
                logging.warning(f"Unknown annotator type: {ann_type_str}")

        # Check for conflicts
        warnings = pipeline.check_conflicts()
        if warnings:
            for warning in warnings:
                logging.warning(warning)

        logging.info(f"Using custom annotators: {args.annotator_types}")
        return pipeline

    # No annotator configuration specified
    return None


# ==============================================================================
# DEPRECATED LEGACY FUNCTIONS (v0.2.0)
# These functions are kept for backward compatibility with existing code.
# They will be removed in v0.3.0. Use InferencePipeline class instead.
# ==============================================================================

def initialize_models(args):
    """Initialize all the models required for the pipeline.

    .. deprecated:: 0.2.0
        Use InferencePipeline class instead. This function will be removed in v0.3.0.

    Args:
        args: Argument namespace with model configuration

    Returns:
        Tuple of (detector, color_layer_classifier, ocr_model, character,
                  class_names, colors, annotator_pipeline)

    Example:
        >>> # Old way (deprecated)
        >>> models = initialize_models(args)
        >>>
        >>> # New way (recommended)
        >>> from onnxtools import InferencePipeline
        >>> pipeline = InferencePipeline(
        ...     model_type=args.model_type,
        ...     model_path=args.model_path,
        ...     conf_thres=args.conf_thres
        ... )
    """
    warnings.warn(
        "initialize_models() is deprecated and will be removed in v0.3.0. "
        "Use InferencePipeline class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Initialize the detector based on model type
    try:
        from onnxtools import create_detector
        detector = create_detector(
            model_type=args.model_type,
            onnx_path=args.model_path,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
    except Exception as e:
        logging.error(f"Error initializing detector: {e}")
        logging.error("Please ensure the ONNX model path is correct and onnxruntime is installed.")
        return None

    # Initialize color/layer and OCR models (使用新API,自动加载配置)
    from onnxtools import ColorLayerORT, OcrORT
    color_layer_model_path = getattr(args, "color_layer_model", "models/color_layer.onnx")
    ocr_model_path = getattr(args, "ocr_model", "models/ocr.onnx")

    # 新API:不再需要手动加载配置,由类内部自动加载
    color_layer_classifier = ColorLayerORT(color_layer_model_path)
    ocr_model = OcrORT(ocr_model_path)

    # 为了保持向后兼容,提供character访问
    character = ocr_model.character

    # Load class names and colors from config
    # 优先从detector模型的class_names属性获取（已在BaseOnnx初始化时从metadata读取）
    if detector.class_names:
        # 从ONNX模型metadata成功读取到类别名称
        logging.info(f"从ONNX模型metadata读取到类别名称: {detector.class_names}")
        max_class_id = max(detector.class_names.keys())
        class_names = [detector.class_names.get(i, f"class_{i}") for i in range(max_class_id + 1)]
    else:
        # 回退到YAML配置文件
        logging.info("ONNX模型metadata中未找到names字段，回退到YAML配置文件")
        with open("configs/det_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        class_names = config["class_names"]

    # colors始终从YAML配置文件读取
    with open("configs/det_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    colors = config["visual_colors"]

    # Initialize annotator pipeline if configured
    annotator_pipeline = None
    if hasattr(args, 'annotator_preset') or hasattr(args, 'annotator_types'):
        annotator_pipeline = create_annotator_pipeline(args)
        if annotator_pipeline:
            logging.info("Annotator pipeline initialized successfully")

    return detector, color_layer_classifier, ocr_model, character, class_names, colors, annotator_pipeline


def process_frame(frame, detector, color_layer_classifier, ocr_model, character,
                  class_names, colors, args, annotator_pipeline=None):
    """Process a single frame for vehicle and plate detection and recognition.

    .. deprecated:: 0.2.0
        Use InferencePipeline class instead. This function will be removed in v0.3.0.

    Args:
        frame: Input image frame
        detector: Detection model
        color_layer_classifier: Color/layer classification model
        ocr_model: OCR model
        character: Character dictionary for OCR
        class_names: List of class names
        colors: List of colors for visualization
        args: Command line arguments
        annotator_pipeline: Optional pre-initialized annotator pipeline

    Returns:
        Tuple of (annotated_frame, output_data)

    Example:
        >>> # Old way (deprecated)
        >>> result_frame, output_data = process_frame(
        ...     frame, detector, color_classifier, ocr_model,
        ...     character, class_names, colors, args
        ... )
        >>>
        >>> # New way (recommended)
        >>> from onnxtools import InferencePipeline
        >>> pipeline = InferencePipeline(
        ...     model_type='rtdetr',
        ...     model_path='models/rtdetr.onnx'
        ... )
        >>> result_frame, output_data = pipeline(frame)
    """
    warnings.warn(
        "process_frame() is deprecated and will be removed in v0.3.0. "
        "Use InferencePipeline class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # 1. Object Detection - now returns Result object
    result = detector(frame)

    output_data = []
    plate_results = []

    # 2. Process all detections to gather data for JSON and prepare for drawing
    if len(result) > 0:
        h_img, w_img, _ = frame.shape
        roi_top_pixel = int(h_img * args.roi_top_ratio)

        # Access detection data from Result object
        boxes = result.boxes  # [N, 4] xyxy format
        scores = result.scores
        class_ids = result.class_ids

        # NOTE: 坐标变换已在检测器的_postprocess方法中完成
        # RT-DETR/RF-DETR的_postprocess会接收orig_shape参数并直接缩放到原图尺寸
        # 所以这里boxes已经是原图坐标,不需要再次缩放
        # 详见: onnxtools/infer_onnx/onnx_rtdetr.py:222-232
        #       onnxtools/infer_onnx/onnx_base.py:354-358

        # NOTE: 边界框裁剪已在Result类的__init__中自动完成
        # 详见: onnxtools/infer_onnx/result.py:129-137
        # Result对象创建时会自动将boxes裁剪到[0, width]和[0, height]范围内

        plate_conf_thres = args.plate_conf_thres if args.plate_conf_thres is not None else args.conf_thres

        # Iterate through detections
        for detection_idx in range(len(result)):
            xyxy = boxes[detection_idx]
            conf = float(scores[detection_idx])
            cls = int(class_ids[detection_idx])
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else "unknown"

            # Apply specific confidence threshold for plates
            if class_name == 'plate' and conf < plate_conf_thres:
                plate_results.append(None) # Keep lists in sync
                continue

            # Keep float values for JSON output
            float_xyxy = [float(c) for c in xyxy]
            x1, y1, x2, y2 = map(int, float_xyxy)
            w, h = x2 - x1, y2 - y1

            plate_text, color_str, layer_str = "", "", ""
            plate_info = None

            if class_name == 'plate':
                exp_x1 = int(max(0, x1 - w * 0.1))
                exp_y1 = int(max(0, y1 - h * 0.1))
                exp_x2 = int(min(w_img, x2 + w * 0.1))
                exp_y2 = int(min(h_img, y2 + h * 0.1))
                plate_img = frame[exp_y1:exp_y2, exp_x1:exp_x2]

                if plate_img.size > 0:
                    # Use new API: ColorLayerONNX.__call__() returns (color, layer, confidence)
                    color_str, layer_str, color_conf = color_layer_classifier(plate_img)

                    # Use new API: OCRONNX.__call__() returns Optional[(text, avg_conf, char_confs)]
                    is_double = (layer_str == "double")
                    ocr_result = ocr_model(plate_img, is_double_layer=is_double)
                    plate_text = ocr_result[0] if ocr_result else ""
                    
                    # Determine if OCR text should be displayed based on ROI and width
                    should_display_ocr = (y1 >= roi_top_pixel) and (w > 50)
                    
                    plate_info = {
                        "plate_text": plate_text, "color": color_str, "layer": layer_str,
                        "should_display_ocr": should_display_ocr
                    }
            
            plate_results.append(plate_info)

            # Populate JSON data regardless of display logic
            if class_name == 'plate':
                output_data.append({
                    "plate_box2d": float_xyxy, "plate_name": plate_text,
                    "plate_color": color_str, "plate_layer": layer_str,
                    "width": w, "height": h
                })
            else:
                output_data.append({
                    "type": class_name, "box2d": float_xyxy, "color": "unknown",
                    "width": w, "height": h
                })

    # 3. Draw detections
    # Use annotator pipeline if available, otherwise fall back to legacy drawing
    if annotator_pipeline and ANNOTATOR_AVAILABLE:
        # Use Result object's to_supervision() method
        if len(result) > 0:
            # Create a modified Result with clipped boxes for visualization
            from onnxtools.infer_onnx import Result
            vis_result = Result(
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                orig_shape=result.orig_shape,
                names=result.names,
                path=result.path,
                orig_img=frame
            )
            sv_detections = vis_result.to_supervision()

            # Import label creation function
            from .utils.supervision_labels import create_ocr_labels

            # Create labels with OCR information (adapted for Result API)
            labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)

            result_frame = annotator_pipeline.annotate(frame.copy(), sv_detections, labels=labels)
            logging.debug(f"Annotated frame with {len(sv_detections)} detections using annotator pipeline")
        else:
            result_frame = frame.copy()
    else:
        # Fall back to legacy drawing system
        if len(result) > 0:
            # Convert boxes to old format [N, 6] with conf and cls
            detections_array = np.concatenate([
                boxes,
                scores.reshape(-1, 1),
                class_ids.reshape(-1, 1)
            ], axis=1)
            scaled_detections_for_drawing = [detections_array]
        else:
            scaled_detections_for_drawing = []
        result_frame = draw_detections(frame.copy(), scaled_detections_for_drawing, class_names, colors, plate_results=plate_results)

    return result_frame, output_data


# ==============================================================================
# RECOMMENDED API - InferencePipeline Class (v0.2.0+)
# This is the recommended way to perform inference. It encapsulates all
# functionality in a single, easy-to-use class.
# ==============================================================================

class InferencePipeline:
    """完整的推理管道类,封装检测、OCR识别和可视化功能。

    这个类整合了目标检测、颜色/层级分类、OCR识别和可视化标注的完整流程,
    提供简单的API接口用于图像推理。

    Attributes:
        detector: 目标检测模型
        color_layer_classifier: 颜色和层级分类模型
        ocr_model: OCR识别模型
        character: OCR字符字典
        class_names: 检测类别名称列表
        colors: 可视化颜色配置
        annotator_pipeline: 可视化标注管道

    Example:
        >>> pipeline = InferencePipeline(
        ...     model_type='rtdetr',
        ...     model_path='models/rtdetr.onnx',
        ...     conf_thres=0.5
        ... )
        >>> result_img, output_data = pipeline(image)
    """

    def __init__(
        self,
        model_type: str,
        model_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        roi_top_ratio: float = 0.5,
        plate_conf_thres: Optional[float] = None,
        color_layer_model: str = 'models/color_layer.onnx',
        ocr_model: str = 'models/ocr.onnx',
        plate_yaml_path: str = 'configs/plate.yaml',
        det_config_path: str = 'configs/det_config.yaml',
        annotator_preset: Optional[str] = None,
        annotator_types: Optional[List[str]] = None,
        **kwargs
    ):
        """初始化推理管道。

        Args:
            model_type: 检测模型类型 ('yolo', 'rtdetr', 'rfdetr')
            model_path: ONNX检测模型路径
            conf_thres: 检测置信度阈值
            iou_thres: NMS的IoU阈值
            roi_top_ratio: ROI区域上边界比例 (0.0-1.0)
            plate_conf_thres: 车牌特定置信度阈值,默认使用conf_thres
            color_layer_model: 颜色/层级分类模型路径
            ocr_model: OCR模型路径
            plate_yaml_path: 车牌配置文件路径
            det_config_path: 检测配置文件路径
            annotator_preset: 可视化预设名称
            annotator_types: 自定义annotator类型列表
            **kwargs: 其他参数
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.roi_top_ratio = roi_top_ratio
        self.plate_conf_thres = plate_conf_thres if plate_conf_thres is not None else conf_thres

        # 初始化检测器
        try:
            from onnxtools import create_detector
            self.detector = create_detector(
                model_type=model_type,
                onnx_path=model_path,
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )
            logging.info(f"Initialized {model_type} detector from {model_path}")
        except Exception as e:
            logging.error(f"Error initializing detector: {e}")
            raise

        # 初始化颜色/层级分类器和OCR模型(使用新API,自动加载配置)
        from onnxtools import ColorLayerORT, OcrORT
        self.color_layer_classifier = ColorLayerORT(
            color_layer_model,
            plate_config_path=plate_yaml_path
        )
        self.ocr_model = OcrORT(
            ocr_model,
            plate_config_path=plate_yaml_path
        )
        logging.info("Initialized color/layer classifier and OCR model")

        # 保存character用于向后兼容
        self.character = self.ocr_model.character

        # 加载类别名称和颜色(使用新API)
        from onnxtools.config import load_det_config
        det_config = load_det_config(det_config_path if det_config_path != 'configs/det_config.yaml' else None)

        if self.detector.class_names:
            logging.info(f"从ONNX模型metadata读取到类别名称: {self.detector.class_names}")
            max_class_id = max(self.detector.class_names.keys())
            self.class_names = [self.detector.class_names.get(i, f"class_{i}") for i in range(max_class_id + 1)]
        else:
            logging.info("ONNX模型metadata中未找到names字段,回退到配置")
            self.class_names = det_config.get("class_names", [])

        # colors始终从配置读取
        self.colors = det_config.get("visual_colors", [])

        # 初始化annotator管道
        self.annotator_pipeline = None
        if annotator_preset or annotator_types:
            self.annotator_pipeline = self._create_annotator_pipeline(
                annotator_preset, annotator_types, **kwargs
            )
            if self.annotator_pipeline:
                logging.info("Annotator pipeline initialized successfully")

    def _create_annotator_pipeline(
        self,
        preset: Optional[str],
        types: Optional[List[str]],
        **kwargs
    ) -> Optional[AnnotatorPipeline]:
        """创建annotator管道。"""
        if not ANNOTATOR_AVAILABLE:
            return None

        # 优先使用预设
        if preset:
            try:
                preset_obj = VisualizationPreset.from_yaml(preset)
                pipeline = preset_obj.create_pipeline()
                logging.info(f"Using annotator preset: {preset}")
                return pipeline
            except Exception as e:
                logging.warning(f"Failed to load preset '{preset}': {e}")
                return None

        # 使用自定义类型
        if types:
            pipeline = AnnotatorPipeline()
            for ann_type_str in types:
                try:
                    ann_type = AnnotatorType(ann_type_str)
                    config = {}
                    # 根据类型配置参数
                    if ann_type == AnnotatorType.ROUND_BOX:
                        config = {
                            'thickness': kwargs.get('box_thickness', 2),
                            'roundness': kwargs.get('roundness', 0.3)
                        }
                    elif ann_type == AnnotatorType.BLUR:
                        config = {'kernel_size': kwargs.get('blur_kernel_size', 15)}
                    elif ann_type in [AnnotatorType.BOX, AnnotatorType.BOX_CORNER]:
                        config = {'thickness': kwargs.get('box_thickness', 2)}

                    pipeline.add(ann_type, config)
                except ValueError:
                    logging.warning(f"Unknown annotator type: {ann_type_str}")

            return pipeline

        return None

    def __call__(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """执行完整的推理管道。

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            Tuple[np.ndarray, List[Dict]]:
                - 标注后的图像 (BGR格式)
                - 输出数据列表,包含检测框、车牌文本、颜色、层级等信息

        Example:
            >>> result_img, output_data = pipeline(image)
            >>> cv2.imwrite('output.jpg', result_img)
            >>> print(output_data[0]['plate_name'])  # 打印第一个车牌号
        """
        # 创建一个伪args对象用于process_frame
        class Args:
            def __init__(self, roi_top_ratio, plate_conf_thres, conf_thres):
                self.roi_top_ratio = roi_top_ratio
                self.plate_conf_thres = plate_conf_thres
                self.conf_thres = conf_thres

        args = Args(self.roi_top_ratio, self.plate_conf_thres, self.conf_thres)

        # 使用现有的process_frame函数
        result_frame, output_data = process_frame(
            frame,
            self.detector,
            self.color_layer_classifier,
            self.ocr_model,
            self.character,
            self.class_names,
            self.colors,
            args,
            self.annotator_pipeline
        )

        return result_frame, output_data