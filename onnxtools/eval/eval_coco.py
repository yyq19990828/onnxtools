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
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

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
        exclude_labels_containing: Optional[List[str]] = None,  # 允许用户指定需要排除的标签内容
        class_mapping: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None  # 类别映射
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
            exclude_labels_containing (Optional[List[str]]): 要排除的类别（检测目标级别过滤）
                支持类别名称或数字ID，例如：["other", "plate"] 或 ["12", "13"]
            class_mapping (Optional[Dict]): 类别映射，格式为 {目标类别: [源类别列表]}
                支持字符串或数字形式，例如：
                - {"vehicle": ["car", "truck", "bus"]}
                - {0: [0, 1, 2, 3, 4]}
                未在映射中的类别将被过滤掉，不参与评估

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
            # 进行检测 - BaseORT.__call__() 返回 Result 对象
            result = self.detector(image, conf_thres=conf_threshold)
            inference_end = time.time()

            # 从 Result 对象提取检测结果
            original_shape = result.orig_shape

            # 应用输出转换（如果提供）- 需要将 Result 转换为旧格式传递
            if output_transform is not None:
                # 构造旧格式的 detections 用于 output_transform
                if len(result) > 0:
                    old_format_det = np.column_stack([
                        result.boxes,
                        result.scores,
                        result.class_ids
                    ])
                    detections = [old_format_det]
                else:
                    detections = [np.zeros((0, 6))]
                detections = output_transform(detections, original_shape)

            postprocess_end = time.time()

            # 记录时间
            times['preprocess'].append((preprocess_start - start_time) * 1000)
            times['inference'].append((inference_end - preprocess_start) * 1000)
            times['postprocess'].append((postprocess_end - inference_end) * 1000)

            # 处理检测结果
            # 注意：Result 对象中的 boxes 坐标已经是原图坐标系
            # - YOLO: postprocess 中通过 scale_boxes() 缩放到原图
            # - RT-DETR/RF-DETR: postprocess 中通过 orig_shape 参数直接缩放到原图
            # 因此这里无需额外缩放
            if len(result) > 0:
                # 从 Result 对象构造 [N, 6] 格式: [x1, y1, x2, y2, conf, class]
                pred = np.column_stack([
                    result.boxes,
                    result.scores,
                    result.class_ids
                ]).copy()
            else:
                pred = np.zeros((0, 6))
            
            predictions.append(pred)
            
            # 安全加载标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            gt = self._load_yolo_labels_safe(str(label_file), img_width, img_height)
            ground_truths.append(gt)

        # 应用类别过滤（排除指定类别，检测目标级别）
        if exclude_labels_containing:
            exclude_class_ids = self._parse_exclude_classes(exclude_labels_containing, names)
            if exclude_class_ids:
                predictions, ground_truths = self._apply_class_exclusion(
                    predictions, ground_truths, exclude_class_ids
                )
                logging.info(f"类别排除已应用: 排除了 {len(exclude_class_ids)} 个类别")

        # 应用类别映射（如果提供）
        if class_mapping is not None:
            id_mapping, new_names = self._parse_class_mapping(class_mapping, names)
            predictions, ground_truths = self._apply_class_mapping(
                predictions, ground_truths, id_mapping
            )
            names = new_names
            logging.info(f"类别映射已应用: {len(id_mapping)} 个源类别 -> {len(new_names)} 个目标类别")

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

    def _parse_class_mapping(
        self,
        class_mapping: Dict[Union[int, str], List[Union[int, str]]],
        names: Dict[int, str]
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        解析类别映射，将 {目标: [源列表]} 转换为 {原ID: 新ID} 格式

        Args:
            class_mapping: {目标类别: [源类别列表]}，支持字符串或数字
            names: 原始类别名称字典 {id: name}

        Returns:
            (id_mapping, new_names)
            - id_mapping: {原始class_id: 新class_id}
            - new_names: {新class_id: 新类别名称}
        """
        # 构建 name -> id 的反向映射
        name_to_id = {v: k for k, v in names.items()}

        id_mapping = {}  # {原始ID: 新ID}
        new_names = {}   # {新ID: 新名称}

        for new_idx, (target, sources) in enumerate(class_mapping.items()):
            # 确定目标类别名称
            if isinstance(target, int):
                target_name = names.get(target, f"class_{target}")
            else:
                target_name = str(target)

            new_names[new_idx] = target_name

            # 处理源类别列表
            for src in sources:
                if isinstance(src, int):
                    src_id = src
                elif isinstance(src, str):
                    # 尝试将字符串解析为数字
                    if src.isdigit():
                        src_id = int(src)
                    else:
                        # 通过名称查找ID
                        src_id = name_to_id.get(src)
                        if src_id is None:
                            logging.warning(f"类别名称 '{src}' 未找到，跳过")
                            continue
                else:
                    continue

                id_mapping[src_id] = new_idx

        return id_mapping, new_names

    def _apply_class_mapping(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        id_mapping: Dict[int, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        对 pred 和 gt 应用类别ID映射

        未在映射中的类别将被过滤掉（不参与评估）

        Args:
            predictions: 预测结果列表，每个元素 [N, 6] (x1,y1,x2,y2,conf,class_id)
            ground_truths: 真值列表，每个元素 [M, 5] (class_id,x1,y1,x2,y2)
            id_mapping: {原始class_id: 新class_id}

        Returns:
            映射后的 (predictions, ground_truths)
        """
        mapped_predictions = []
        mapped_ground_truths = []

        for pred in predictions:
            if len(pred) == 0:
                mapped_predictions.append(np.zeros((0, 6)))
                continue

            # pred 格式: [x1, y1, x2, y2, conf, class_id]
            class_ids = pred[:, 5].astype(int)

            # 创建映射后的 class_id
            new_class_ids = np.array([id_mapping.get(cid, -1) for cid in class_ids])

            # 只保留在映射中的检测
            valid_mask = new_class_ids >= 0
            if np.any(valid_mask):
                new_pred = pred[valid_mask].copy()
                new_pred[:, 5] = new_class_ids[valid_mask]
                mapped_predictions.append(new_pred)
            else:
                mapped_predictions.append(np.zeros((0, 6)))

        for gt in ground_truths:
            if len(gt) == 0:
                mapped_ground_truths.append(np.zeros((0, 5)))
                continue

            # gt 格式: [class_id, x1, y1, x2, y2]
            class_ids = gt[:, 0].astype(int)

            # 创建映射后的 class_id
            new_class_ids = np.array([id_mapping.get(cid, -1) for cid in class_ids])

            # 只保留在映射中的真值
            valid_mask = new_class_ids >= 0
            if np.any(valid_mask):
                new_gt = gt[valid_mask].copy()
                new_gt[:, 0] = new_class_ids[valid_mask]
                mapped_ground_truths.append(new_gt)
            else:
                mapped_ground_truths.append(np.zeros((0, 5)))

        return mapped_predictions, mapped_ground_truths

    def _parse_exclude_classes(
        self,
        exclude_labels: List[str],
        names: Dict[int, str]
    ) -> set:
        """
        解析要排除的类别，返回类别ID集合

        Args:
            exclude_labels: 要排除的类别（支持类别名称或数字ID字符串）
            names: 原始类别名称字典 {id: name}

        Returns:
            set: 要排除的类别ID集合
        """
        # 构建 name -> id 的反向映射
        name_to_id = {v: k for k, v in names.items()}

        exclude_ids = set()
        for label in exclude_labels:
            label = label.strip()
            if not label:
                continue

            # 尝试解析为数字
            if label.isdigit():
                exclude_ids.add(int(label))
            else:
                # 通过名称查找ID
                if label in name_to_id:
                    exclude_ids.add(name_to_id[label])
                else:
                    logging.warning(f"排除类别 '{label}' 未找到，跳过")

        return exclude_ids

    def _apply_class_exclusion(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        exclude_ids: set
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        从 pred 和 gt 中排除指定类别的检测目标

        Args:
            predictions: 预测结果列表，每个元素 [N, 6] (x1,y1,x2,y2,conf,class_id)
            ground_truths: 真值列表，每个元素 [M, 5] (class_id,x1,y1,x2,y2)
            exclude_ids: 要排除的类别ID集合

        Returns:
            过滤后的 (predictions, ground_truths)
        """
        filtered_predictions = []
        filtered_ground_truths = []

        for pred in predictions:
            if len(pred) == 0:
                filtered_predictions.append(np.zeros((0, 6)))
                continue

            # pred 格式: [x1, y1, x2, y2, conf, class_id]
            class_ids = pred[:, 5].astype(int)

            # 保留不在排除列表中的检测
            valid_mask = ~np.isin(class_ids, list(exclude_ids))
            if np.any(valid_mask):
                filtered_predictions.append(pred[valid_mask].copy())
            else:
                filtered_predictions.append(np.zeros((0, 6)))

        for gt in ground_truths:
            if len(gt) == 0:
                filtered_ground_truths.append(np.zeros((0, 5)))
                continue

            # gt 格式: [class_id, x1, y1, x2, y2]
            class_ids = gt[:, 0].astype(int)

            # 保留不在排除列表中的真值
            valid_mask = ~np.isin(class_ids, list(exclude_ids))
            if np.any(valid_mask):
                filtered_ground_truths.append(gt[valid_mask].copy())
            else:
                filtered_ground_truths.append(np.zeros((0, 5)))

        return filtered_predictions, filtered_ground_truths