import argparse
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目路径到系统路径
project_root = Path(__file__).parent.parent  # 获取父目录作为项目根目录
sys.path.insert(0, str(project_root))
from onnxtools import DetDatasetEvaluator, create_detector, setup_logger


def parse_class_mapping_arg(mapping_str: str) -> Dict[str, List[str]]:
    """
    解析命令行类别映射参数

    格式: "vehicle:car,truck,bus;cyclist:bicycle,motorcycle"
    返回: {"vehicle": ["car", "truck", "bus"], "cyclist": ["bicycle", "motorcycle"]}

    Args:
        mapping_str: 类别映射字符串

    Returns:
        Dict[str, List[str]]: 解析后的映射字典
    """
    result = {}
    for group in mapping_str.split(';'):
        group = group.strip()
        if not group or ':' not in group:
            continue
        target, sources = group.split(':', 1)
        target = target.strip()
        source_list = [s.strip() for s in sources.split(',') if s.strip()]
        if target and source_list:
            result[target] = source_list
    return result


def main():
    parser = argparse.ArgumentParser(description='评估ONNX模型在数据集上的性能')
    parser.add_argument('--model-type', type=str, default='rtdetr',
                        choices=['rtdetr', 'yolo', 'rfdetr'], help='模型类型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='ONNX模型文件路径')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                        help='IoU阈值')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最大处理图片数量')
    parser.add_argument('--exclude-files', type=str, nargs='*', default=None,
                        help='需要排除的文件列表')
    parser.add_argument('--exclude-labels-containing', type=str, nargs='*', default=None,
                        help='需要排除的标签内容关键词列表')
    parser.add_argument('--class-mapping', type=str, default=None,
                        help='类别映射，格式: "vehicle:car,truck,bus;cyclist:bicycle,motorcycle"')

    args = parser.parse_args()

    setup_logger()

    # 使用工厂函数创建检测器
    detector = create_detector(args.model_type, args.model_path)

    # 使用统一的评估器
    evaluator = DetDatasetEvaluator(detector)

    # 构建evaluate_dataset的参数字典
    eval_kwargs = {
        'dataset_path': args.dataset_path,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold
    }

    # 添加可选参数
    if args.max_images is not None:
        eval_kwargs['max_images'] = args.max_images
    if args.exclude_files is not None:
        eval_kwargs['exclude_files'] = args.exclude_files
    if args.exclude_labels_containing is not None:
        eval_kwargs['exclude_labels_containing'] = args.exclude_labels_containing
    if args.class_mapping is not None:
        eval_kwargs['class_mapping'] = parse_class_mapping_arg(args.class_mapping)

    evaluator.evaluate_dataset(**eval_kwargs)

if __name__ == '__main__':
    main()
