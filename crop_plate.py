import cv2
import numpy as np
import json
import os
import argparse
import logging

from onnxtools.pipeline import initialize_models
from onnxtools import setup_logger


def infer_source_type(input_path):
    """
    推断输入源类型
    """
    input_path_lower = input_path.lower()
    if os.path.isdir(input_path):
        return 'folder'
    elif any(input_path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        return 'image'
    elif any(input_path_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        return 'video'
    elif input_path_lower.startswith('rtsp://'):
        return 'rtsp'
    elif input_path.isdigit():
        return 'camera'
    else:
        return 'unknown'


def crop_plate_from_frame(frame, detector, color_layer_classifier, ocr_model, class_names, args, frame_name="frame"):
    """
    从单帧图像中检测并裁剪车牌小图

    Args:
        frame: 输入图像帧
        detector: 检测模型
        color_layer_classifier: 颜色/层级分类模型
        ocr_model: OCR模型
        class_names: 类别名称列表
        args: 命令行参数
        frame_name: 帧标识名称(用于保存文件名)

    Returns:
        Tuple of (cropped_images_count, detection_data)
    """
    # 目标类别ID (车牌类别)
    target_classes = ['plate']  # 车牌类别
    target_class_ids = [i for i, name in enumerate(class_names) if name in target_classes]

    # 1. 目标检测
    detections, original_shape = detector(frame)

    output_data = []
    cropped_count = 0

    # 2. 处理检测结果
    if detections and len(detections[0]) > 0:
        h_img, w_img, _ = frame.shape

        # 缩放坐标到原始图像尺寸
        scaled_detections = detections[0].copy()
        if hasattr(detector, '__class__') and detector.__class__.__name__ in ['RtdetrORT', 'RfdetrORT']:
            # RT-DETR和RF-DETR模型坐标缩放
            scale_x = w_img / detector.input_shape[1]
            scale_y = h_img / detector.input_shape[0]
            scaled_detections[:, [0, 2]] *= scale_x
            scaled_detections[:, [1, 3]] *= scale_y

        # 裁剪坐标到图像边界内
        clipped_detections = scaled_detections
        clipped_detections[:, 0] = np.clip(clipped_detections[:, 0], 0, w_img)
        clipped_detections[:, 1] = np.clip(clipped_detections[:, 1], 0, h_img)
        clipped_detections[:, 2] = np.clip(clipped_detections[:, 2], 0, w_img)
        clipped_detections[:, 3] = np.clip(clipped_detections[:, 3], 0, h_img)

        # 遍历所有检测结果
        for detection_idx, (*xyxy, conf, cls) in enumerate(clipped_detections):
            class_id = int(cls)
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"

            # 只处理目标类别(车牌)
            if class_id not in target_class_ids:
                continue

            # 应用置信度阈值
            if conf < args.conf_thres:
                continue

            # 获取边界框坐标
            float_xyxy = [float(c) for c in xyxy]
            x1, y1, x2, y2 = map(int, float_xyxy)
            w, h = x2 - x1, y2 - y1

            # 过滤过小的检测框
            if w < args.min_width or h < args.min_height:
                logging.debug(f"跳过过小的检测框: {class_name} w={w}, h={h}")
                continue

            # 扩展边界框(可选)
            if args.expand_ratio > 0:
                expand_w = int(w * args.expand_ratio)
                expand_h = int(h * args.expand_ratio)
                exp_x1 = max(0, x1 - expand_w)
                exp_y1 = max(0, y1 - expand_h)
                exp_x2 = min(w_img, x2 + expand_w)
                exp_y2 = min(h_img, y2 + expand_h)
            else:
                exp_x1, exp_y1, exp_x2, exp_y2 = x1, y1, x2, y2

            # 裁剪车牌区域
            cropped_plate = frame[exp_y1:exp_y2, exp_x1:exp_x2]

            if cropped_plate.size == 0:
                logging.warning(f"裁剪区域为空: {class_name} box=[{exp_x1},{exp_y1},{exp_x2},{exp_y2}]")
                continue

            # 3. 对车牌进行OCR识别和颜色/层级分类
            plate_text = "unknown"
            plate_color = "unknown"
            plate_layer = "unknown"  # single or double

            try:
                # 颜色/层级分类
                if color_layer_classifier:
                    color_name, layer_name, color_conf = color_layer_classifier(cropped_plate)
                    plate_color = color_name
                    plate_layer = layer_name
                    logging.debug(f"颜色/层级分类: color={color_name}, layer={layer_name}, conf={color_conf:.3f}")

                # OCR识别 (根据层级决定是否为双层车牌)
                if ocr_model:
                    is_double_layer = (plate_layer == "double")
                    ocr_result = ocr_model(cropped_plate, is_double_layer=is_double_layer)
                    # ocr_result 是 (text, avg_conf, char_confs) 或 None
                    if ocr_result:
                        plate_text = ocr_result[0]  # 第一个元素是识别的文本
                        logging.debug(f"OCR识别: text={plate_text}")
            except Exception as e:
                logging.warning(f"车牌OCR/颜色识别失败: {e}")

            # 使用从color_layer_classifier获取的layer信息
            plate_type = plate_layer  # 'single' or 'double'

            # 保存裁剪图像，格式: 车牌号_single/double_颜色_宽x高_原图名.jpg
            # 如果OCR失败，使用"unknown"作为车牌号
            safe_plate_text = plate_text if plate_text != "unknown" else f"plate_{detection_idx:03d}"
            crop_filename = f"{safe_plate_text}_{plate_type}_{plate_color}_{w}x{h}_{frame_name}.jpg"
            crop_path = os.path.join(args.output_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_plate)
            cropped_count += 1

            # 记录检测信息
            detection_info = {
                "frame_name": frame_name,
                "class": class_name,
                "plate_type": plate_type,
                "plate_text": plate_text,
                "plate_color": plate_color,
                "confidence": float(conf),
                "bbox": float_xyxy,
                "bbox_expanded": [float(exp_x1), float(exp_y1), float(exp_x2), float(exp_y2)],
                "width": w,
                "height": h,
                "cropped_file": crop_filename
            }
            output_data.append(detection_info)

            logging.info(f"裁剪车牌: {crop_filename} (text={plate_text}, color={plate_color}, conf={conf:.3f}, size={w}x{h})")

    return cropped_count, output_data


def main(args):
    # 设置日志
    setup_logger(args.log_level)

    # 检查输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"创建输出目录: {args.output_dir}")

    # 初始化模型
    logging.info("正在初始化检测模型...")
    models = initialize_models(args)
    if models is None:
        return
    detector, color_layer_classifier, ocr_model, _, class_names, _, _ = models

    source_type = infer_source_type(args.input)
    total_cropped = 0
    all_detections = []

    if source_type == 'image':
        # 处理单张图像
        img = cv2.imread(args.input)
        if img is None:
            logging.error(f"无法读取图像: {args.input}")
            return

        frame_name = os.path.splitext(os.path.basename(args.input))[0]
        cropped_count, detection_data = crop_plate_from_frame(
            img, detector, color_layer_classifier, ocr_model,
            class_names, args, frame_name
        )
        total_cropped += cropped_count
        all_detections.extend(detection_data)

        logging.info(f"完成处理,共裁剪 {cropped_count} 个车牌小图")

    elif source_type == 'folder':
        # 处理文件夹中的所有图像
        image_files = [f for f in os.listdir(args.input)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        total_images = len(image_files)
        logging.info(f"在文件夹 '{args.input}' 中找到 {total_images} 张图像")

        for i, image_file in enumerate(image_files):
            logging.info(f"处理图像 {i + 1}/{total_images}: {image_file}")
            image_path = os.path.join(args.input, image_file)
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"无法读取图像 {image_path}, 跳过")
                continue

            frame_name = os.path.splitext(image_file)[0]
            cropped_count, detection_data = crop_plate_from_frame(
                img, detector, color_layer_classifier, ocr_model,
                class_names, args, frame_name
            )
            total_cropped += cropped_count
            all_detections.extend(detection_data)

        logging.info(f"完成处理所有 {total_images} 张图像,共裁剪 {total_cropped} 个车牌小图")

    elif source_type in ['video', 'rtsp', 'camera']:
        # 处理视频流
        if source_type == 'camera':
            cap = cv2.VideoCapture(int(args.input))
        else:
            cap = cv2.VideoCapture(args.input)

        if not cap.isOpened():
            logging.error(f"无法打开视频源: {args.input}")
            return

        # 获取总帧数
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total_frames = 0

        if total_frames > 0:
            logging.info(f"处理视频,共 {total_frames} 帧")
        else:
            logging.info("处理视频流(总帧数未知)")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧处理
            if frame_count % (args.frame_skip + 1) == 0:
                frame_name = f"frame_{frame_count:06d}"
                cropped_count, detection_data = crop_plate_from_frame(
                    frame, detector, color_layer_classifier, ocr_model,
                    class_names, args, frame_name
                )
                total_cropped += cropped_count
                all_detections.extend(detection_data)

                # 每100帧记录一次进度
                if frame_count > 0 and frame_count % 100 == 0:
                    if total_frames > 0:
                        logging.info(f"已处理帧 {frame_count}/{total_frames}, 已裁剪 {total_cropped} 个车牌小图")
                    else:
                        logging.info(f"已处理帧 {frame_count}, 已裁剪 {total_cropped} 个车牌小图")

            frame_count += 1

        cap.release()
        logging.info(f"完成视频处理,共处理 {frame_count} 帧,裁剪 {total_cropped} 个车牌小图")

    else:
        logging.error(f"不支持的输入源类型: {source_type}")
        return

    # 保存检测结果JSON
    if args.save_json and all_detections:
        json_path = os.path.join(args.output_dir, "plate_detections.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, ensure_ascii=False, indent=4)
        logging.info(f"检测结果已保存到: {json_path}")

    logging.info(f"所有裁剪图像已保存到: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX车牌检测与裁剪工具')

    # 必需参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='ONNX检测模型路径')
    parser.add_argument('--model-type', type=str, default='rtdetr',
                       choices=['rtdetr', 'yolo', 'rfdetr'],
                       help='模型类型')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像/视频路径或摄像头ID')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/cropped_plates',
                       help='裁剪图像保存目录')
    parser.add_argument('--save-json', action='store_true',
                       help='保存检测结果JSON文件')

    # 检测参数
    parser.add_argument('--conf-thres', type=float, default=0.5,
                       help='检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                       help='NMS的IoU阈值')

    # 裁剪参数
    parser.add_argument('--expand-ratio', type=float, default=0.0,
                       help='边界框扩展比例(0.0表示不扩展,0.1表示扩展10%%)')
    parser.add_argument('--min-width', type=int, default=20,
                       help='最小检测框宽度(像素)')
    parser.add_argument('--min-height', type=int, default=10,
                       help='最小检测框高度(像素)')

    # 视频处理参数
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='视频处理时跳帧数量(0表示处理所有帧)')

    # 其他参数
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')

    # OCR模型参数
    parser.add_argument('--color-layer-model', type=str,
                       default='models/color_layer.onnx',
                       help='颜色/层级模型路径')
    parser.add_argument('--ocr-model', type=str,
                       default='models/ocr_mobile.onnx',
                       help='OCR模型路径')
    parser.add_argument('--roi-top-ratio', type=float, default=0.0,
                       help='ROI顶部比例(未使用但需要)')
    parser.add_argument('--plate-conf-thres', type=float, default=None,
                       help='车牌置信度阈值(未使用但需要)')

    args = parser.parse_args()

    # 验证模型路径
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        exit(1)

    # 验证输入路径
    if not args.input.isdigit() and not args.input.startswith('rtsp://'):
        if not os.path.exists(args.input):
            logging.error(f"输入路径不存在: {args.input}")
            exit(1)

    main(args)
