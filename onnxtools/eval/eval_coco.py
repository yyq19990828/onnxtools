"""
数据集评估模块

包含:
- DetDatasetEvaluator: 检测数据集评估器
用于评估ONNX模型在YOLO格式数据集上的性能
"""

import numpy as np
import logging
import time
import cv2
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from ..infer_onnx import BaseORT
from ..utils.detection_metrics import evaluate_detection, print_metrics


class DetDatasetEvaluator:
    """检测数据集评估器"""
    
    def __init__(self, detector: BaseORT):
        """
        初始化评估器
        
        Args:
            detector (BaseORT): 检测器实例
        """
        self.detector = detector
    
    def evaluate_dataset(
        self, 
        dataset_path: str,
        output_transform: Optional[Callable] = None,
        conf_threshold: float = 0.25,  # 与Ultralytics验证模式对齐，避免过低阈值产生大量误检
        iou_threshold: float = 0.7,  # 保留参数以保持一致性
        max_images: Optional[int] = None,
        exclude_files: Optional[List[str]] = None,  # 允许用户指定需要排除的文件
        exclude_labels_containing: Optional[List[str]] = None  # 允许用户指定需要排除的标签内容
    ) -> Dict[str, Any]:
        """
        在YOLO格式数据集上评估模型性能
        
        Args:
            dataset_path (str): 数据集路径
            output_transform (Optional[Callable]): 输出转换函数
            conf_threshold (float): 置信度阈值，默认0.25与Ultralytics对齐
                注意：Ultralytics在验证模式下会将默认的0.001重置为0.25
                参考：ultralytics/utils/metrics.py:403 v8.3.179
                conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
            iou_threshold (float): IoU阈值
            max_images (Optional[int]): 最大评估图像数量
            exclude_files (Optional[List[str]]): 需要排除的文件名列表
            exclude_labels_containing (Optional[List[str]]): 排除包含指定内容的标签文件
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 重要说明：置信度阈值默认值更改
        # ==========================================
        # 之前版本使用 conf_threshold=0.001，但发现与Ultralytics结果存在显著差异
        # 原因：Ultralytics在验证过程中会自动将过低的置信度阈值重置为0.25
        # 
        # 具体机制（参考ultralytics/utils/metrics.py:403）：
        # conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
        # 
        # 即：如果传入的conf是默认验证值0.001，会被强制重置为预测模式的0.25
        # 
        # 为了保持与Ultralytics一致的评估结果，现将默认值统一设置为0.25
        # 这样可以避免因置信度阈值差异导致的指标差异（P/R/mAP等）
        # ==========================================
        
        dataset_path = Path(dataset_path)
        
        # 数据集路径检测逻辑
        test_images_dir = dataset_path / "images" / "test"
        test_labels_dir = dataset_path / "labels" / "test"
        val_images_dir = dataset_path / "images" / "val"
        val_labels_dir = dataset_path / "labels" / "val"
        
        if test_images_dir.exists() and test_labels_dir.exists():
            images_dir = test_images_dir
            labels_dir = test_labels_dir
            split_name = "test"
            logging.info("使用test数据集进行评估")
        elif val_images_dir.exists() and val_labels_dir.exists():
            images_dir = val_images_dir
            labels_dir = val_labels_dir
            split_name = "val"
            logging.info("使用val数据集进行评估")
        else:
            # 回退逻辑
            images_dir = dataset_path / "images" / "train"
            labels_dir = dataset_path / "labels" / "train"
            if images_dir.exists() and labels_dir.exists():
                split_name = "train"
                logging.info("使用train数据集进行评估")
            else:
                images_dir = dataset_path / "images"
                labels_dir = dataset_path / "labels" 
                split_name = "root"
                if not images_dir.exists():
                    raise ValueError(f"未找到有效的图像目录")
                if not labels_dir.exists():
                    raise ValueError(f"标签目录不存在: {labels_dir}")
                logging.info("使用根目录下的images/labels进行评估")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        # 检查图像文件是否可访问，过滤损坏或无效的文件
        valid_image_files = []
        exclude_files = exclude_files or []
        exclude_labels_containing = exclude_labels_containing or []
        
        for image_file in image_files:
            # 检查是否在排除列表中
            if image_file.name in exclude_files:
                continue
                
            # 基本有效性检查：文件存在且大小大于0
            if not (image_file.exists() and image_file.stat().st_size > 0):
                continue
                
            # 检查是否有对应的标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                continue
                
            # 检查标签文件内容是否包含需要排除的内容
            if exclude_labels_containing:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(exclude_text in content for exclude_text in exclude_labels_containing):
                            continue
                except Exception:
                    continue
                    
            valid_image_files.append(image_file)
        
        image_files = valid_image_files
        
        if not image_files:
            logging.warning(f"在 {images_dir} 中未找到有效的图像文件")
        
        if max_images:
            image_files = image_files[:max_images]
        
        logging.info(f"开始评估{split_name}数据集，共 {len(image_files)} 张图像")
        
        predictions = []
        ground_truths = []
        
        # 加载类别名称
        names = {}
        data_yaml = dataset_path / "classes.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                names = data_config.get('names', {})
                if isinstance(names, list):
                    names = {i: name for i, name in enumerate(names)}
        
        # 性能统计
        times = {
            'preprocess': [],
            'inference': [],
            'postprocess': []
        }
        
        logging.info(f"评估 {len(image_files)} 张图像...")

        for i, image_file in enumerate(image_files):
            if i % 100 == 0:
                logging.info(f"处理进度: {i}/{len(image_files)}")
            
            # 读取图像
            start_time = time.time()
            image = cv2.imread(str(image_file))
            if image is None:
                logging.warning(f"无法读取图像: {image_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # 测量预处理时间
            preprocess_start = time.time()
            # 进行检测
            detections, original_shape = self.detector(image, conf_thres=conf_threshold)
            inference_end = time.time()
            
            # 应用输出转换（如果提供）
            if output_transform is not None:
                detections = output_transform(detections, original_shape)
            
            postprocess_end = time.time()
            
            # 记录时间
            times['preprocess'].append((preprocess_start - start_time) * 1000)
            times['inference'].append((inference_end - preprocess_start) * 1000)
            times['postprocess'].append((postprocess_end - inference_end) * 1000)
            
            # 处理检测结果（坐标已在模型后处理中正确缩放）
            if detections and len(detections[0]) > 0:
                pred = detections[0].copy()  # [N, 6] format: [x1, y1, x2, y2, conf, class]
                
                # 对于使用了新坐标处理逻辑的YOLO模型，检测结果已经在原图坐标系
                # 对于RT-DETR和RF-DETR等特殊模型，仍需要额外缩放
                if type(self.detector).__name__ in ['RtdetrORT', 'RfdetrORT']:
                    pred[:, [0, 2]] = pred[:, [0, 2]] * img_width / self.detector.input_shape[1]   # x坐标缩放
                    pred[:, [1, 3]] = pred[:, [1, 3]] * img_height / self.detector.input_shape[0]  # y坐标缩放
                # YoloOnnx使用新的坐标处理逻辑，坐标已正确缩放，无需额外处理
            else:
                pred = np.zeros((0, 6))
            
            predictions.append(pred)
            
            # 安全加载标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            gt = self._load_yolo_labels_safe(str(label_file), img_width, img_height)
            ground_truths.append(gt)
        
        # 计算指标
        results = evaluate_detection(predictions, ground_truths, names)
        
        # 添加性能统计
        if times['preprocess']:
            results['speed_preprocess'] = np.mean(times['preprocess'])
            results['speed_inference'] = np.mean(times['inference']) 
            results['speed_loss'] = 0.0
            results['speed_postprocess'] = np.mean(times['postprocess'])
        
        # 打印结果
        print_metrics(results, names)
        
        return results
    
    def _load_yolo_labels_safe(self, label_path: str, img_width: int, img_height: int) -> np.ndarray:
        """安全加载YOLO标签，跳过无效行"""
        if not Path(label_path).exists():
            return np.zeros((0, 5))
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # 转换为xyxy格式
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        labels.append([class_id, x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        # 跳过无效行
                        continue
        
        return np.array(labels) if labels else np.zeros((0, 5))