"""通用目标裁剪工具 - 使用 Result API

支持裁剪任意类别的目标区域，包括：
- 行人（person/pedestrian）
- 车牌（plate）+ OCR识别
- 车辆（car/truck/bus）
- 其他任意检测类别

使用示例:
    # 裁剪行人
    python demo_crop.py --model-path models/yolo11n.onnx --model-type yolo \\
        --input data/sample.jpg --target-classes person --output-dir runs/person

    # 裁剪车牌并进行OCR识别
    python demo_crop.py --model-path models/yolo11n.onnx --model-type yolo \\
        --input data/sample.jpg --target-classes plate --output-dir runs/plate \\
        --enable-ocr

    # 裁剪多个类别
    python demo_crop.py --model-path models/yolo11n.onnx --model-type yolo \\
        --input data/sample.jpg --target-classes person car truck \\
        --output-dir runs/multi

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-07
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import cv2

from onnxtools import ColorLayerORT, OcrORT, create_detector, setup_logger


def infer_source_type(input_path: str) -> str:
    """推断输入源类型."""
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


def process_ocr_for_plate(
    crop_img,
    color_layer_classifier,
    ocr_model
) -> Tuple[str, str, str]:
    """
    对车牌进行OCR识别和颜色/层级分类

    Args:
        crop_img: 裁剪的车牌图像
        color_layer_classifier: 颜色/层级分类模型
        ocr_model: OCR模型

    Returns:
        Tuple of (plate_text, plate_color, plate_layer)
    """
    plate_text = "unknown"
    plate_color = "unknown"
    plate_layer = "unknown"

    try:
        # 颜色/层级分类
        if color_layer_classifier:
            color_name, layer_name, color_conf = color_layer_classifier(crop_img)
            plate_color = color_name
            plate_layer = layer_name
            logging.debug(f"颜色/层级: color={color_name}, layer={layer_name}, conf={color_conf:.3f}")

        # OCR识别
        if ocr_model:
            is_double_layer = (plate_layer == "double")
            ocr_result = ocr_model(crop_img, is_double_layer=is_double_layer)
            if ocr_result:
                plate_text = ocr_result[0]
                logging.debug(f"OCR识别: text={plate_text}")
    except Exception as e:
        logging.warning(f"车牌OCR/颜色识别失败: {e}")

    return plate_text, plate_color, plate_layer


def crop_from_frame(
    frame,
    detector,
    args,
    color_layer_classifier=None,
    ocr_model=None,
    frame_name="frame"
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    从单帧图像中检测并裁剪目标区域

    Args:
        frame: 输入图像帧
        detector: 检测模型
        args: 命令行参数
        color_layer_classifier: 颜色/层级分类模型（可选）
        ocr_model: OCR模型（可选）
        frame_name: 帧标识名称

    Returns:
        Tuple of (cropped_images_count, detection_data)
    """
    # 1. 目标检测 - 返回 Result 对象
    result = detector(frame)

    output_data = []
    cropped_count = 0

    # 2. 如果没有检测结果
    if len(result) == 0:
        return cropped_count, output_data

    # 3. 查找可用的目标类别
    available_classes = list(result.names.values())
    target_classes = []

    for cls in args.target_classes:
        if cls in available_classes:
            target_classes.append(cls)
        else:
            # 尝试查找类似的类别名称
            similar_classes = [c for c in available_classes if cls.lower() in c.lower() or c.lower() in cls.lower()]
            if similar_classes:
                logging.warning(f"类别 '{cls}' 未找到，使用相似类别: {similar_classes[0]}")
                target_classes.append(similar_classes[0])

    if not target_classes:
        logging.warning(
            f"未找到目标类别。指定类别: {args.target_classes}, "
            f"可用类别: {available_classes[:10]}..."
        )
        return cropped_count, output_data

    logging.debug(f"使用类别: {target_classes}")

    # 4. 使用新 Result API 进行裁剪和过滤
    gain = 1.0 + args.expand_ratio

    crops = result.crop(
        conf_threshold=args.conf_thres,
        classes=target_classes,  # 使用类别名称过滤
        gain=gain,
        pad=args.pad,
        square=args.square
    )

    # 5. 处理裁剪结果
    for crop_data in crops:
        # 获取基本信息
        box = crop_data['box']
        w = int(box[2] - box[0])
        h = int(box[3] - box[1])
        class_name = crop_data['class_name']
        confidence = crop_data['confidence']
        crop_img = crop_data['image']
        crop_box = crop_data['crop_box']

        # 过滤过小的检测框
        if w < args.min_width or h < args.min_height:
            logging.debug(f"跳过过小的检测框: {class_name} w={w}, h={h}")
            continue

        # 构建检测信息
        detection_info = {
            "frame_name": frame_name,
            "class": class_name,
            "confidence": confidence,
            "bbox": box.tolist(),
            "bbox_expanded": crop_box.tolist(),
            "width": w,
            "height": h
        }

        # 6. 对车牌类进行OCR识别（如果启用）
        if args.enable_ocr and class_name == 'plate':
            plate_text, plate_color, plate_layer = process_ocr_for_plate(
                crop_img, color_layer_classifier, ocr_model
            )
            detection_info.update({
                "plate_text": plate_text,
                "plate_color": plate_color,
                "plate_layer": plate_layer
            })

            # 车牌文件名格式: 车牌号_层级_颜色_宽x高_原图名.jpg
            safe_plate_text = plate_text if plate_text != "unknown" else f"plate_{crop_data['index']:03d}"
            crop_filename = f"{safe_plate_text}_{plate_layer}_{plate_color}_{w}x{h}_{frame_name}.jpg"
            log_msg = f"裁剪车牌: {crop_filename} (text={plate_text}, color={plate_color}, conf={confidence:.3f})"
        else:
            # 普通目标文件名格式: 原图名_类别_索引_置信度.jpg
            crop_filename = f"{frame_name}_{class_name}_{crop_data['index']:03d}_conf{confidence:.2f}.jpg"
            log_msg = f"裁剪 {class_name}: {crop_filename} (conf={confidence:.3f}, size={w}x{h})"

        # 7. 保存裁剪图像
        crop_path = os.path.join(args.output_dir, crop_filename)
        cv2.imwrite(crop_path, crop_img)
        cropped_count += 1

        detection_info["cropped_file"] = crop_filename
        output_data.append(detection_info)

        logging.info(log_msg)

    return cropped_count, output_data


def process_source(args, detector, color_layer_classifier=None, ocr_model=None):
    """处理输入源（图像/文件夹/视频/相机）"""
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
        cropped_count, detection_data = crop_from_frame(
            img, detector, args, color_layer_classifier, ocr_model, frame_name
        )
        total_cropped += cropped_count
        all_detections.extend(detection_data)

        logging.info(f"完成处理，共裁剪 {cropped_count} 个目标")

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
                logging.warning(f"无法读取图像 {image_path}，跳过")
                continue

            frame_name = os.path.splitext(image_file)[0]
            cropped_count, detection_data = crop_from_frame(
                img, detector, args, color_layer_classifier, ocr_model, frame_name
            )
            total_cropped += cropped_count
            all_detections.extend(detection_data)

        logging.info(f"完成处理所有 {total_images} 张图像，共裁剪 {total_cropped} 个目标")

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
            logging.info(f"处理视频，共 {total_frames} 帧")
        else:
            logging.info("处理视频流（总帧数未知）")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧处理
            if frame_count % (args.frame_skip + 1) == 0:
                frame_name = f"frame_{frame_count:06d}"
                cropped_count, detection_data = crop_from_frame(
                    frame, detector, args, color_layer_classifier, ocr_model, frame_name
                )
                total_cropped += cropped_count
                all_detections.extend(detection_data)

                # 每100帧记录一次进度
                if frame_count > 0 and frame_count % 100 == 0:
                    if total_frames > 0:
                        logging.info(f"已处理帧 {frame_count}/{total_frames}，已裁剪 {total_cropped} 个目标")
                    else:
                        logging.info(f"已处理帧 {frame_count}，已裁剪 {total_cropped} 个目标")

            frame_count += 1

        cap.release()
        logging.info(f"完成视频处理，共处理 {frame_count} 帧，裁剪 {total_cropped} 个目标")

    else:
        logging.error(f"不支持的输入源类型: {source_type}")
        return

    # 保存检测结果JSON
    if args.save_json and all_detections:
        json_path = os.path.join(args.output_dir, "detections.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, ensure_ascii=False, indent=4)
        logging.info(f"检测结果已保存到: {json_path}")

    logging.info(f"所有裁剪图像已保存到: {args.output_dir}")


def main(args):
    # 设置日志
    setup_logger(args.log_level)

    # 检查输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"创建输出目录: {args.output_dir}")

    # 初始化模型
    logging.info("正在初始化检测模型...")
    try:
        detector = create_detector(
            model_type=args.model_type,
            onnx_path=args.model_path,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
    except Exception as e:
        logging.error(f"检测模型初始化失败: {e}")
        return

    color_layer_classifier = None
    ocr_model = None
    if args.enable_ocr:
        try:
            color_layer_classifier = ColorLayerORT(args.color_layer_model)
            ocr_model = OcrORT(args.ocr_model)
        except Exception as e:
            logging.error(f"OCR模型初始化失败: {e}")
            return

    # 处理输入源
    process_source(args, detector, color_layer_classifier, ocr_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ONNX通用目标裁剪工具 - 使用 Result API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 裁剪行人
  python %(prog)s --model-path models/yolo11n.onnx --model-type yolo \\
      --input data/sample.jpg --target-classes person --output-dir runs/person

  # 裁剪车牌并进行OCR识别
  python %(prog)s --model-path models/yolo11n.onnx --model-type yolo \\
      --input data/sample.jpg --target-classes plate --output-dir runs/plate \\
      --enable-ocr

  # 裁剪多个类别
  python %(prog)s --model-path models/yolo11n.onnx --model-type yolo \\
      --input data/sample.jpg --target-classes person car truck \\
      --output-dir runs/multi
        """
    )

    # 必需参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='ONNX检测模型路径')
    parser.add_argument('--model-type', type=str, default='yolo',
                       choices=['rtdetr', 'yolo', 'rfdetr'],
                       help='模型类型 (默认: yolo)')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像/视频路径、文件夹或摄像头ID')

    # 目标类别参数
    parser.add_argument('--target-classes', type=str, nargs='+', required=True,
                       help='要裁剪的目标类别列表 (例如: person plate car)')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/crops',
                       help='裁剪图像保存目录 (默认: runs/crops)')
    parser.add_argument('--save-json', action='store_true',
                       help='保存检测结果JSON文件')

    # 检测参数
    parser.add_argument('--conf-thres', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                       help='NMS的IoU阈值 (默认: 0.5)')

    # 裁剪参数
    parser.add_argument('--expand-ratio', type=float, default=0.0,
                       help='边界框扩展比例 (0.1表示扩展10%%, 默认: 0.0)')
    parser.add_argument('--pad', type=int, default=0,
                       help='边界框填充像素 (默认: 0)')
    parser.add_argument('--square', action='store_true',
                       help='强制裁剪为正方形')
    parser.add_argument('--min-width', type=int, default=10,
                       help='最小检测框宽度(像素) (默认: 10)')
    parser.add_argument('--min-height', type=int, default=10,
                       help='最小检测框高度(像素) (默认: 10)')

    # OCR参数
    parser.add_argument('--enable-ocr', action='store_true',
                       help='启用车牌OCR识别和颜色/层级分类')
    parser.add_argument('--color-layer-model', type=str,
                       default='models/color_layer.onnx',
                       help='颜色/层级模型路径')
    parser.add_argument('--ocr-model', type=str,
                       default='models/ocr_mobile.onnx',
                       help='OCR模型路径')

    # 视频处理参数
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='视频处理时跳帧数量 (0表示处理所有帧)')

    # 其他参数
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别 (默认: INFO)')

    args = parser.parse_args()

    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        exit(1)

    # 验证输入路径
    if not args.input.isdigit() and not args.input.startswith('rtsp://'):
        if not os.path.exists(args.input):
            print(f"错误: 输入路径不存在: {args.input}")
            exit(1)

    main(args)
