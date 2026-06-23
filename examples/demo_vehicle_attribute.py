#!/usr/bin/env python3
"""Demo: 二阶段车辆属性预标注 (检测 → 机动车 ROI → 车型/颜色)。

使用 VehicleAttributePipeline 对单张图像推理,打印 / 保存每个检测框的几何与
机动车属性 (vehicle_type / color)。

Example:
    python examples/demo_vehicle_attribute.py --input data/sample.jpg
    python examples/demo_vehicle_attribute.py --input data/sample.jpg --output-json out.json
"""

import argparse
import json
import logging

import cv2

from onnxtools import setup_logger
from onnxtools.pipeline import VehicleAttributePipeline


def main(args):
    """Run the two-stage vehicle-attribute pipeline on a single image."""
    setup_logger(args.log_level)

    pipeline = VehicleAttributePipeline(
        model_type=args.model_type,
        model_path=args.model_path,
        va_model_path=args.va_model,
        conf_thres=args.conf_thres,
    )

    img = cv2.imread(args.input)
    if img is None:
        logging.error("无法读取图像: %s", args.input)
        return

    output = pipeline(img)
    motor = sum(1 for d in output if "vehicle_type" in d)
    logging.info("检测 %d 个目标,其中机动车 %d 个", len(output), motor)

    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info("结果已保存到 %s", args.output_json)
    else:
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="二阶段车辆属性预标注 demo")
    parser.add_argument("--input", default="data/sample.jpg", help="输入图像路径")
    parser.add_argument("--model-path", default="models/rtdetr-2024080100.onnx", help="检测 ONNX 模型路径")
    parser.add_argument(
        "--model-type", default="rtdetr", choices=["rtdetr", "yolo", "rfdetr"], help="检测模型类型 (默认 rtdetr)"
    )
    parser.add_argument("--va-model", default="models/va_260612.onnx", help="车辆属性分类 ONNX 模型路径")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--output-json", default=None, help="结果 JSON 输出路径 (缺省打印到 stdout)")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="日志级别"
    )
    main(parser.parse_args())
