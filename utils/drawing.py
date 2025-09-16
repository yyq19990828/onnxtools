import cv2
import numpy as np
import os
import logging
import time
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Union

try:
    import supervision as sv
    from .supervision_converter import convert_to_supervision_detections
    from .supervision_labels import create_ocr_labels
    from .supervision_config import create_box_annotator, create_rich_label_annotator
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logging.warning("Supervision library not available, falling back to PIL implementation")

def draw_detections(image, detections, class_names, colors, plate_results=None, font_path="SourceHanSans-VF.ttf", use_supervision=True):
    """
    Draws detection boxes on the image. Handles Chinese characters gracefully.

    Args:
        image: Input image as numpy array
        detections: Detection results
        class_names: Class name mapping
        colors: Colors for different classes
        plate_results: Optional OCR results for plates
        font_path: Path to font file
        use_supervision: Whether to use supervision library (True) or fallback to PIL (False)

    Returns:
        Annotated image as numpy array
    """
    # Try supervision implementation first if enabled and available
    if use_supervision and SUPERVISION_AVAILABLE:
        try:
            return draw_detections_supervision(image, detections, class_names, colors, plate_results, font_path)
        except Exception as e:
            logging.warning(f"Supervision drawing failed, falling back to PIL: {e}")
            # Fall through to PIL implementation

    # PIL implementation (original code)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    img_h, img_w = pil_img.height, pil_img.width
    
    font_found = os.path.exists(font_path)
    if font_found:
        try:
            # 尝试加载粗体字重
            # Removed layout_engine for compatibility with older Pillow versions.
            font = ImageFont.truetype(font_path, 20)
            font.set_variation_by_name('Bold')
        except (IOError, ValueError, AttributeError):
            try:
                # 如果失败，回退到常规字体
                font = ImageFont.truetype(font_path, 20)
                logging.warning(f"Could not load bold variation for {font_path}. Using regular weight.")
            except IOError:
                logging.warning(f"Could not load font file at {font_path}. Using default font.")
                font = ImageFont.load_default()
                font_found = False # Treat as not found if loading fails
    else:
        logging.warning(f"Font file not found at {font_path}. Chinese characters will not be displayed. Using default font.")
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):
        for j, (*xyxy, conf, cls) in enumerate(det):
            class_id = int(cls)
            label = f'{class_names[class_id] if class_id < len(class_names) else "unknown"} {conf:.2f}'
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 根据类别ID选择颜色
            box_color = colors[class_id % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3) # 加粗边框
            
            # 绘制类别标签
            label_y_pos = y1 - 25
            if label_y_pos < 0: label_y_pos = y1 + 5 # 如果标签超出顶部，则移到框内
            draw.text((x1, label_y_pos), label, fill=box_color, font=font)
            
            # 绘制OCR结果 - 只对车牌类别的检测框绘制OCR信息
            if (class_id < len(class_names) and class_names[class_id] == 'plate' and plate_results and
                j < len(plate_results) and plate_results[j] is not None):
                plate_info = plate_results[j]
                
                # 检查是否应该显示OCR信息
                if not plate_info.get("should_display_ocr", False) or not font_found:
                    continue

                ocr_str = plate_info.get("plate_text", "")
                color_str = plate_info.get("color", "")
                layer_str = plate_info.get("layer", "")

                # 构建第二行的信息文本
                info_parts = []
                if color_str and color_str != "unknown": info_parts.append(color_str)
                if layer_str and layer_str != "unknown": info_parts.append(layer_str)
                info_str = ' '.join(info_parts)

                # 只有在车牌号存在时才绘制
                if ocr_str:
                    try:
                        # 使用 textbbox 获取更准确的尺寸
                        line1_bbox = draw.textbbox((0, 0), ocr_str, font=font)
                        line1_w, line1_h = line1_bbox[2] - line1_bbox[0], line1_bbox[3] - line1_bbox[1]
                        
                        line2_w, line2_h = 0, 0
                        if info_str:
                            line2_bbox = draw.textbbox((0, 0), info_str, font=font)
                            line2_w, line2_h = line2_bbox[2] - line2_bbox[0], line2_bbox[3] - line2_bbox[1]

                    except AttributeError:
                        # 兼容旧版 Pillow
                        line1_w, line1_h = draw.textsize(ocr_str, font=font)
                        line2_w, line2_h = (draw.textsize(info_str, font=font) if info_str else (0, 0))

                    total_text_h = line1_h + (line2_h + 5 if info_str else 0) # 两行文本的总高度，带间距
                    max_text_w = max(line1_w, line2_w)

                    # 垂直位置：优先放在框下方
                    text_y = y2 + 8 # 框下方8px
                    if text_y + total_text_h > img_h: # 如果下方空间不足，移到上方
                        text_y = y1 - total_text_h - 8
                        if text_y < 0: # 如果上方也超出
                            text_y = max(0, y1 + 5) # 放在框内顶部

                    # 水平位置：居中对齐
                    box_center_x = (x1 + x2) // 2
                    text_x1 = box_center_x - line1_w // 2
                    text_x2 = box_center_x - line2_w // 2 if info_str else 0
                    
                    # 确保背景框不超出图像边界
                    padding = 3
                    bg_x1 = max(0, box_center_x - max_text_w // 2 - padding)
                    bg_y1 = max(0, text_y - padding)
                    bg_x2 = min(img_w, box_center_x + max_text_w // 2 + padding)
                    bg_y2 = min(img_h, text_y + total_text_h + padding)
                    
                    # 绘制文本背景
                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 180))

                    # 绘制第一行：车牌号
                    draw.text((text_x1, text_y), ocr_str, fill="blue", font=font)
                    
                    # 绘制第二行：颜色和层信息
                    if info_str:
                        draw.text((text_x2, text_y + line1_h + 5), info_str, fill="black", font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_detections_supervision(image: np.ndarray,
                               detections: List[List[List[float]]],
                               class_names: Union[Dict[int, str], List[str]],
                               colors: List[tuple],
                               plate_results: Optional[List[Optional[Dict[str, Any]]]] = None,
                               font_path: str = "SourceHanSans-VF.ttf") -> np.ndarray:
    """
    Draw detection boxes using supervision library for enhanced performance and visuals.

    Args:
        image: Input image as BGR numpy array
        detections: Detection results in format [[[x1, y1, x2, y2, confidence, class_id], ...]]
        class_names: Dict mapping class_id to class_name or list of class names
        colors: List of colors for different classes (not used directly in supervision)
        plate_results: Optional OCR results for plate detections
        font_path: Path to font file for text rendering

    Returns:
        Annotated image as BGR numpy array
    """
    if not SUPERVISION_AVAILABLE:
        raise ImportError("Supervision library not available")

    # Convert to supervision format
    sv_detections = convert_to_supervision_detections(detections, class_names)

    if len(sv_detections.xyxy) == 0:
        return image.copy()

    # Create annotators
    box_annotator = create_box_annotator(thickness=3)
    label_annotator = create_rich_label_annotator(font_path=font_path, font_size=16)

    # Start with copy of input image
    annotated_image = image.copy()

    # Draw bounding boxes
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=sv_detections
    )

    # Create labels with OCR information
    if len(detections) > 0 and len(detections[0]) > 0:
        labels = create_ocr_labels(detections[0], plate_results or [], class_names)

        # Draw labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=sv_detections,
            labels=labels
        )

    return annotated_image


def benchmark_drawing_performance(image: np.ndarray,
                                 detections_data: List[List[List[float]]],
                                 iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark drawing performance comparing PIL and supervision implementations.

    Args:
        image: Test image
        detections_data: Detection data for testing
        iterations: Number of test iterations

    Returns:
        Dictionary with performance metrics
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    # Mock class names and colors for testing
    class_names = {0: "vehicle", 1: "plate"}
    colors = [(255, 0, 0), (0, 255, 0)]

    # Test PIL implementation (current)
    start_time = time.time()
    for _ in range(iterations):
        result1 = draw_detections(image.copy(), detections_data, class_names, colors, use_supervision=False)
    pil_time = (time.time() - start_time) / iterations * 1000  # ms

    # Test supervision implementation (if available)
    if SUPERVISION_AVAILABLE:
        start_time = time.time()
        for _ in range(iterations):
            result2 = draw_detections_supervision(image.copy(), detections_data, class_names, colors)
        supervision_time = (time.time() - start_time) / iterations * 1000  # ms
    else:
        supervision_time = pil_time  # Fallback

    improvement_ratio = pil_time / supervision_time if supervision_time > 0 else 1.0

    return {
        'pil_avg_time': pil_time,
        'supervision_avg_time': supervision_time,
        'improvement_ratio': improvement_ratio
    }