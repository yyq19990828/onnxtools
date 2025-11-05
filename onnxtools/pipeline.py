import cv2
import numpy as np
import yaml
import logging

from .utils.drawing import draw_detections

# Import new annotator functionality (optional)
try:
    from .utils.annotator_factory import AnnotatorFactory, AnnotatorType, AnnotatorPipeline
    from .utils.visualization_preset import VisualizationPreset, Presets
    from .utils.supervision_converter import convert_to_supervision_detections
    import supervision as sv
    ANNOTATOR_AVAILABLE = True
except ImportError:
    ANNOTATOR_AVAILABLE = False
    logging.warning("Annotator functionality not available. Using legacy drawing.")

def create_annotator_pipeline(args):
    """
    Create annotator pipeline based on command line arguments.

    Args:
        args: Command line arguments containing annotator configuration

    Returns:
        AnnotatorPipeline or None if annotators not available/configured
    """
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


#TODO use a class to wrap pipeline
def initialize_models(args):
    """
    Initialize all the models required for the pipeline.
    """
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

    # Initialize color/layer and OCR models
    from onnxtools import ColorLayerORT, OcrORT
    color_layer_model_path = getattr(args, "color_layer_model", "models/color_layer.onnx")
    ocr_model_path = getattr(args, "ocr_model", "models/ocr.onnx")
    plate_yaml_path = "configs/plate.yaml"

    with open(plate_yaml_path, "r", encoding="utf-8") as f:
        plate_yaml = yaml.safe_load(f)
        character = ["blank"] + plate_yaml["ocr_dict"] + [" "]
        color_dict = plate_yaml["color_dict"]
        layer_dict = plate_yaml["layer_dict"]

    color_layer_classifier = ColorLayerORT(
        color_layer_model_path,
        color_map=color_dict,
        layer_map=layer_dict
    )
    ocr_model = OcrORT(ocr_model_path, character=character)

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


def process_frame(frame, detector, color_layer_classifier, ocr_model, character, class_names, colors, args, annotator_pipeline=None):
    """
    Process a single frame for vehicle and plate detection and recognition.

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
    """
    # 1. Object Detection
    detections, original_shape = detector(frame)

    output_data = []
    plate_results = []

    # 2. Process all detections to gather data for JSON and prepare for drawing
    if detections and len(detections[0]) > 0:
        h_img, w_img, _ = frame.shape
        roi_top_pixel = int(h_img * args.roi_top_ratio)

        # Scale coordinates to original image size
        scaled_detections = detections[0].copy()
        if hasattr(detector, '__class__') and detector.__class__.__name__ in ['RtdetrORT', 'RfdetrORT']:
            # RT-DETR and RF-DETR models直接拉伸图像，坐标需要从输入尺寸缩放到原始尺寸
            # 坐标从输入尺寸变换回原始尺寸需要乘以缩放比例
            scale_x = w_img / detector.input_shape[1]  # original_width / input_width
            scale_y = h_img / detector.input_shape[0]  # original_height / input_height
            scaled_detections[:, [0, 2]] *= scale_x  # x1, x2坐标缩放
            scaled_detections[:, [1, 3]] *= scale_y  # y1, y2坐标缩放

        # Ensure detections are clipped within frame boundaries
        clipped_detections = scaled_detections
        clipped_detections[:, 0] = np.clip(clipped_detections[:, 0], 0, w_img)
        clipped_detections[:, 1] = np.clip(clipped_detections[:, 1], 0, h_img)
        clipped_detections[:, 2] = np.clip(clipped_detections[:, 2], 0, w_img)
        clipped_detections[:, 3] = np.clip(clipped_detections[:, 3], 0, h_img)
        
        plate_conf_thres = args.plate_conf_thres if args.plate_conf_thres is not None else args.conf_thres

        for detection_idx, (*xyxy, conf, cls) in enumerate(clipped_detections):
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
        # Use new annotator system
        sv_detections = convert_to_supervision_detections(
            [clipped_detections] if detections and len(detections[0]) > 0 else [],
            class_names
        )

        if sv_detections is not None:
            # Import label creation function
            from .utils.supervision_labels import create_ocr_labels
            
            # Create labels with OCR information
            labels = None
            if detections and len(detections[0]) > 0:
                labels = create_ocr_labels(clipped_detections, plate_results, class_names)
            
            result_frame = annotator_pipeline.annotate(frame.copy(), sv_detections, labels=labels)
            logging.debug(f"Annotated frame with {len(sv_detections)} detections using annotator pipeline")
        else:
            result_frame = frame.copy()
    else:
        # Fall back to legacy drawing system
        scaled_detections_for_drawing = [clipped_detections] if detections and len(detections[0]) > 0 else []
        result_frame = draw_detections(frame.copy(), scaled_detections_for_drawing, class_names, colors, plate_results=plate_results)

    return result_frame, output_data