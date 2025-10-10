# Quick Start: 实现BaseOnnx子类

**Branch**: `005-baseonnx-postprocess-call` | **Date**: 2025-10-09 | **Spec**: [spec.md](./spec.md)

## Overview

本指南提供BaseOnnx子类的快速实现教程,包括:
1. 最小实现示例 (必须实现的抽象方法)
2. 自定义阶段方法 (可选重写)
3. 完整示例 (端到端推理流程)
4. 常见问题和最佳实践

**目标读者**: 需要集成新ONNX模型到推理引擎的开发者

## Minimum Implementation (最小实现)

### 步骤1: 创建子类并实现2个抽象方法

所有BaseOnnx子类**必须**实现以下2个抽象方法:
- `_postprocess()` - 模型输出后处理
- `_preprocess_static()` - 静态预处理方法

**示例**: 实现一个简单的分类模型推理类

```python
# File: infer_onnx/onnx_classifier.py

from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import cv2
from .onnx_base import BaseOnnx


class ClassifierOnnx(BaseOnnx):
    """图像分类模型推理 - 最小实现示例"""

    def __init__(
        self,
        onnx_path: str,
        class_names: List[str],
        conf_thres: float = 0.5,
        input_shape: Tuple[int, int] = (224, 224),
        **kwargs
    ):
        """
        Args:
            onnx_path: ONNX模型文件路径
            class_names: 类别名称列表
            conf_thres: 置信度阈值
            input_shape: 模型输入尺寸 (height, width)
        """
        super().__init__(onnx_path, conf_thres, input_shape=input_shape, **kwargs)
        self.class_names = class_names
        self.num_classes = len(class_names)

    # ============ 必须实现的抽象方法 ============

    def _postprocess(
        self,
        prediction: List[np.ndarray],
        conf_thres: float,
        **kwargs
    ) -> List[np.ndarray]:
        """分类模型后处理: Softmax + Top-K过滤"""
        results = []
        for pred in prediction:
            # 1. Softmax归一化
            exp_scores = np.exp(pred - np.max(pred))
            probabilities = exp_scores / np.sum(exp_scores)

            # 2. 置信度过滤
            top_k = kwargs.get('top_k', 5)
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            top_probs = probabilities[top_indices]

            # 3. 过滤低置信度结果
            mask = top_probs >= conf_thres
            filtered_indices = top_indices[mask]
            filtered_probs = top_probs[mask]

            # 4. 构造结果: [class_id, confidence]
            result = np.column_stack([filtered_indices, filtered_probs])
            results.append(result)

        return results if results else [np.empty((0, 2))]

    @staticmethod
    @abstractmethod
    def _preprocess_static(
        image: np.ndarray,
        input_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple]:
        """分类模型预处理: Resize + BGR2RGB + 归一化"""
        # 1. Resize到目标尺寸 (不保持宽高比)
        resized = cv2.resize(image, (input_shape[1], input_shape[0]))

        # 2. BGR -> RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. 归一化到 [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0

        # 4. 减均值除标准差 (ImageNet统计值)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # 5. NCHW格式
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        # 6. 缩放比例 (分类任务通常不需要,返回(1.0, 1.0))
        scale = (1.0, 1.0)

        return input_tensor, scale
```

### 步骤2: 使用子类进行推理

```python
# File: examples/classifier_example.py

import cv2
from infer_onnx.onnx_classifier import ClassifierOnnx

# 1. 初始化分类器
classifier = ClassifierOnnx(
    onnx_path='models/resnet50.onnx',
    class_names=['cat', 'dog', 'bird', ...],  # 1000个ImageNet类别
    conf_thres=0.3,
    input_shape=(224, 224)
)

# 2. 加载图像
image = cv2.imread('data/sample.jpg')

# 3. 推理 (调用__call__方法)
results, original_shape = classifier(image, conf_thres=0.5, top_k=3)

# 4. 解析结果
if results[0].size > 0:
    for class_id, confidence in results[0]:
        class_name = classifier.class_names[int(class_id)]
        print(f"{class_name}: {confidence:.3f}")
else:
    print("No classification above threshold")
```

**输出示例**:
```
cat: 0.856
dog: 0.112
bird: 0.032
```

### 步骤3: 编写单元测试

```python
# File: tests/unit/test_classifier_onnx.py

import pytest
import numpy as np
from infer_onnx.onnx_classifier import ClassifierOnnx


class TestClassifierOnnx:
    """ClassifierOnnx单元测试"""

    @pytest.fixture
    def classifier(self):
        """测试夹具"""
        return ClassifierOnnx(
            onnx_path='models/resnet50.onnx',
            class_names=['class_' + str(i) for i in range(1000)],
            conf_thres=0.3
        )

    def test_postprocess(self, classifier):
        """测试后处理方法"""
        # 模拟Softmax前的logits
        prediction = [np.random.randn(1000)]
        results = classifier._postprocess(prediction, conf_thres=0.3, top_k=5)

        assert isinstance(results, list)
        assert results[0].shape[1] == 2  # [class_id, confidence]
        assert results[0][:, 1].max() <= 1.0  # 概率最大为1
        assert results[0][:, 1].min() >= 0.3  # 置信度过滤

    def test_preprocess_static(self):
        """测试静态预处理方法"""
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        input_tensor, scale = ClassifierOnnx._preprocess_static(image, (224, 224))

        assert input_tensor.shape == (1, 3, 224, 224)
        assert isinstance(scale, tuple)

    def test_call_integration(self, classifier):
        """测试完整推理流程"""
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        results, original_shape = classifier(image, conf_thres=0.5, top_k=3)

        assert isinstance(results, list)
        assert original_shape == (1080, 1920)
        assert results[0].shape[1] == 2
```

---

## Custom Stage Methods (自定义阶段方法)

### 何时需要重写阶段方法?

| 阶段方法 | 重写场景 | 示例 |
|---------|---------|------|
| `_prepare_inference()` | 需要特殊的输入验证或预处理 | OCR双层车牌检测、特殊图像增强 |
| `_execute_inference()` | 需要多次推理或特殊推理逻辑 | TTA (Test Time Augmentation)、多尺度推理 |
| `_finalize_inference()` | 需要特殊的后处理或结果格式 | OCR文本解码、颜色分类返回类别名称 |

### 示例1: 重写`_prepare_inference()` - OCR双层车牌处理

```python
# File: infer_onnx/onnx_ocr.py

class OCRONNX(BaseOnnx):
    """OCR识别模型 - 重写prepare阶段"""

    def _prepare_inference(self, image, conf_thres, **kwargs):
        """OCR准备阶段: 检测双层车牌并调整预处理"""
        # 1. 调用父类默认实现
        super()._prepare_inference(image, conf_thres, **kwargs)

        # 2. OCR特殊逻辑: 双层车牌检测
        is_double_layer = kwargs.get('is_double_layer', False)
        if is_double_layer:
            # 双层车牌需要垂直拼接预处理
            self._context.input_tensor = self._preprocess_double_layer(image)
            self.logger.info("Double-layer plate detected, using special preprocessing")

    def _preprocess_double_layer(self, image):
        """双层车牌专用预处理"""
        # 切分上下层
        h = image.shape[0]
        top_half = image[:h//2, :]
        bottom_half = image[h//2:, :]

        # 分别预处理后拼接
        top_tensor, _ = self._preprocess_static(top_half, (24, 168))
        bottom_tensor, _ = self._preprocess_static(bottom_half, (24, 168))

        # 垂直拼接为 [1, 3, 48, 168]
        combined_tensor = np.concatenate([top_tensor, bottom_tensor], axis=2)
        return combined_tensor

    # ... (其他方法省略)
```

**使用示例**:
```python
ocr_model = OCRONNX(onnx_path='models/ocr.onnx', ...)

# 单层车牌 (默认)
result_single = ocr_model(plate_image, conf_thres=0.7)

# 双层车牌 (传递is_double_layer参数)
result_double = ocr_model(plate_image, conf_thres=0.7, is_double_layer=True)
```

### 示例2: 重写`_execute_inference()` - TTA多尺度推理

```python
# File: infer_onnx/onnx_yolo_tta.py

class YoloOnnxTTA(YoloOnnx):
    """YOLO模型 + TTA (Test Time Augmentation) - 重写execute阶段"""

    def _execute_inference(self, input_tensor):
        """TTA推理: 原始图 + 水平翻转 + 多尺度"""
        all_outputs = []

        # 1. 原始图推理
        outputs_original = super()._execute_inference(input_tensor)
        all_outputs.append(outputs_original)

        # 2. 水平翻转推理
        flipped_tensor = np.flip(input_tensor, axis=3)  # 沿宽度翻转
        outputs_flipped = self._runner.infer(feed_dict={self._runner.input_names[0]: flipped_tensor})
        outputs_flipped = [outputs_flipped[name] for name in self._runner.output_names]
        # 翻转回检测框坐标
        outputs_flipped = self._unflip_detections(outputs_flipped)
        all_outputs.append(outputs_flipped)

        # 3. 多尺度推理 (0.8x, 1.2x)
        for scale_factor in [0.8, 1.2]:
            scaled_shape = (int(self.input_shape[0] * scale_factor),
                           int(self.input_shape[1] * scale_factor))
            scaled_tensor, _ = self._preprocess_static(self._context.original_image, scaled_shape)
            outputs_scaled = self._runner.infer(feed_dict={self._runner.input_names[0]: scaled_tensor})
            outputs_scaled = [outputs_scaled[name] for name in self._runner.output_names]
            all_outputs.append(outputs_scaled)

        # 4. 融合所有结果 (WBF - Weighted Boxes Fusion)
        fused_outputs = self._fuse_tta_outputs(all_outputs)

        # 5. 保存融合结果
        if hasattr(self, '_context'):
            self._context.raw_outputs = fused_outputs
        else:
            self._raw_outputs = fused_outputs

    def _fuse_tta_outputs(self, all_outputs):
        """融合TTA结果 (简化版本: 平均融合)"""
        # 实际实现可以使用WBF算法
        return [np.mean([out[i] for out in all_outputs], axis=0) for i in range(len(all_outputs[0]))]
```

**使用示例**:
```python
# 标准YOLO推理
yolo_standard = YoloOnnx(onnx_path='models/yolo11n.onnx', conf_thres=0.5)
result_standard = yolo_standard(image)

# YOLO + TTA推理 (更高精度,更慢)
yolo_tta = YoloOnnxTTA(onnx_path='models/yolo11n.onnx', conf_thres=0.5)
result_tta = yolo_tta(image)  # 自动使用TTA推理
```

### 示例3: 重写`_finalize_inference()` - 颜色分类返回类别名

```python
# File: infer_onnx/onnx_color.py

class ColorLayerONNX(BaseOnnx):
    """车牌颜色和层级分类 - 重写finalize阶段"""

    def __init__(self, onnx_path, color_map, layer_map, **kwargs):
        super().__init__(onnx_path, **kwargs)
        self.color_map = color_map  # {0: 'blue', 1: 'yellow', ...}
        self.layer_map = layer_map  # {0: 'single', 1: 'double'}

    def _finalize_inference(self, outputs, scale, original_shape, conf_thres, **kwargs):
        """颜色分类完成阶段: 返回类别名称和置信度"""
        # 1. 调用后处理获取logits
        color_logits, layer_logits = self._postprocess(outputs, conf_thres, **kwargs)

        # 2. Softmax归一化
        color_probs = self._softmax(color_logits)
        layer_probs = self._softmax(layer_logits)

        # 3. 获取最高置信度的类别
        color_idx = np.argmax(color_probs)
        layer_idx = np.argmax(layer_probs)

        # 4. 映射到类别名称
        color_name = self.color_map[color_idx]
        layer_name = self.layer_map[layer_idx]

        # 5. 计算置信度
        color_conf = float(color_probs[color_idx])
        layer_conf = float(layer_probs[layer_idx])

        # 6. 返回自定义格式 (不是标准的检测框格式)
        return {
            'color': color_name,
            'layer': layer_name,
            'color_confidence': color_conf,
            'layer_confidence': layer_conf
        }

    @staticmethod
    def _softmax(logits):
        """Softmax归一化"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def _postprocess(self, prediction, conf_thres, **kwargs):
        """后处理: 提取颜色和层级logits"""
        # 假设模型输出2个张量: [color_logits, layer_logits]
        color_logits = prediction[0]
        layer_logits = prediction[1]
        return color_logits, layer_logits

    @staticmethod
    @abstractmethod
    def _preprocess_static(image, input_shape):
        """颜色分类预处理"""
        # ... (标准分类预处理)
        pass
```

**使用示例**:
```python
color_classifier = ColorLayerONNX(
    onnx_path='models/color_layer.onnx',
    color_map={0: 'blue', 1: 'yellow', 2: 'white', 3: 'black', 4: 'green'},
    layer_map={0: 'single', 1: 'double'},
    conf_thres=0.5
)

# 推理
result = color_classifier(plate_image, conf_thres=0.7)

# 输出自定义格式
print(f"颜色: {result['color']} (置信度: {result['color_confidence']:.3f})")
print(f"层级: {result['layer']} (置信度: {result['layer_confidence']:.3f})")
```

**输出示例**:
```
颜色: blue (置信度: 0.923)
层级: single (置信度: 0.987)
```

---

## Complete Example (完整示例)

### 场景: 实现RT-DETR目标检测模型

**需求**:
- 模型: RT-DETR (Real-Time DETR)
- 输出格式: 归一化坐标 (需要转换为绝对坐标)
- 特殊处理: 无需NMS (模型已内置)

**完整实现**:

```python
# File: infer_onnx/onnx_rtdetr.py

from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import cv2
from .onnx_base import BaseOnnx


class RTDETROnnx(BaseOnnx):
    """RT-DETR目标检测模型推理 - 完整示例"""

    def __init__(
        self,
        onnx_path: str,
        conf_thres: float = 0.5,
        input_shape: Tuple[int, int] = (640, 640),
        **kwargs
    ):
        super().__init__(onnx_path, conf_thres, input_shape=input_shape, **kwargs)
        self.logger.info(f"RT-DETR initialized with input shape {input_shape}")

    # ============ 必须实现的抽象方法 ============

    def _postprocess(
        self,
        prediction: List[np.ndarray],
        conf_thres: float,
        **kwargs
    ) -> List[np.ndarray]:
        """RT-DETR后处理: 直接过滤置信度 (无需NMS)"""
        results = []
        for pred in prediction:
            # RT-DETR输出格式: [num_boxes, 6] (x1, y1, x2, y2, conf, class)
            # 坐标已经是归一化的 [0, 1]
            if pred.ndim == 1:
                pred = pred.reshape(-1, 6)

            # 置信度过滤
            mask = pred[:, 4] > conf_thres
            filtered_pred = pred[mask]

            # 按置信度排序
            if filtered_pred.size > 0:
                sorted_indices = np.argsort(filtered_pred[:, 4])[::-1]
                filtered_pred = filtered_pred[sorted_indices]

            results.append(filtered_pred if filtered_pred.size > 0 else np.empty((0, 6)))

        return results

    @staticmethod
    @abstractmethod
    def _preprocess_static(
        image: np.ndarray,
        input_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple]:
        """RT-DETR预处理: Letterbox + BGR2RGB + 归一化"""
        # 1. Letterbox resize (保持宽高比)
        h, w = image.shape[:2]
        target_h, target_w = input_shape

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Padding
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # 灰色填充
        )

        # 2. BGR -> RGB
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # 3. 归一化到 [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0

        # 4. NCHW格式
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        # 5. 保存缩放信息 (用于坐标还原)
        scale_info = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'new_w': new_w,
            'new_h': new_h
        }

        return input_tensor, scale_info

    # ============ 可选重写的阶段方法 ============

    def _finalize_inference(self, outputs, scale, original_shape, conf_thres, **kwargs):
        """重写finalize: RT-DETR归一化坐标转换为绝对坐标"""
        # 1. 调用父类后处理
        detections = self._postprocess(outputs, conf_thres, **kwargs)

        # 2. RT-DETR特殊处理: 归一化坐标 -> 绝对坐标
        scaled_detections = []
        for det in detections:
            if det.size > 0:
                # 提取缩放信息
                scale_factor = scale['scale']
                pad_w = scale['pad_w']
                pad_h = scale['pad_h']

                # 归一化坐标 [0, 1] -> 填充图像坐标
                det[:, 0] *= self.input_shape[1]  # x1
                det[:, 1] *= self.input_shape[0]  # y1
                det[:, 2] *= self.input_shape[1]  # x2
                det[:, 3] *= self.input_shape[0]  # y2

                # 移除填充
                det[:, [0, 2]] -= pad_w
                det[:, [1, 3]] -= pad_h

                # 缩放到原图尺寸
                det[:, [0, 2]] /= scale_factor
                det[:, [1, 3]] /= scale_factor

                # 裁剪到原图边界
                det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, original_shape[1])
                det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, original_shape[0])

            scaled_detections.append(det)

        return scaled_detections if scaled_detections else [np.empty((0, 6))]
```

**使用示例**:

```python
# File: examples/rtdetr_example.py

import cv2
import numpy as np
from infer_onnx.onnx_rtdetr import RTDETROnnx

# 1. 初始化RT-DETR检测器
detector = RTDETROnnx(
    onnx_path='models/rtdetr-2024080100.onnx',
    conf_thres=0.25,
    input_shape=(640, 640)
)

# 2. 加载测试图像
image = cv2.imread('data/test.jpg')
print(f"Original image shape: {image.shape}")

# 3. 推理
detections, original_shape = detector(image, conf_thres=0.5)

# 4. 可视化结果
for det in detections:
    if det.size > 0:
        for box in det:
            x1, y1, x2, y2, conf, cls = box
            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 绘制标签
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 5. 显示结果
cv2.imshow('RT-DETR Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 保存结果
cv2.imwrite('output/rtdetr_result.jpg', image)
print(f"Detected {len(detections[0])} objects")
```

**测试示例**:

```python
# File: tests/unit/test_rtdetr_onnx.py

import pytest
import numpy as np
import cv2
from infer_onnx.onnx_rtdetr import RTDETROnnx


class TestRTDETROnnx:
    """RT-DETR单元测试"""

    @pytest.fixture
    def detector(self):
        return RTDETROnnx(
            onnx_path='models/rtdetr-2024080100.onnx',
            conf_thres=0.25
        )

    def test_postprocess_no_nms(self, detector):
        """验证RT-DETR不使用NMS"""
        # 模拟重叠的高置信度检测框
        prediction = [np.array([
            [100, 100, 200, 200, 0.9, 0],  # 高置信度
            [105, 105, 205, 205, 0.85, 0],  # 重叠框,高置信度
        ])]

        results = detector._postprocess(prediction, conf_thres=0.5)

        # RT-DETR不应该移除重叠框
        assert results[0].shape[0] == 2, "RT-DETR should not apply NMS"

    def test_preprocess_letterbox(self):
        """验证Letterbox预处理"""
        # 非正方形图像
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        input_tensor, scale_info = RTDETROnnx._preprocess_static(image, (640, 640))

        # 验证输出形状
        assert input_tensor.shape == (1, 3, 640, 640)

        # 验证缩放信息
        assert 'scale' in scale_info
        assert 'pad_w' in scale_info
        assert 'pad_h' in scale_info

        # 验证缩放比例正确 (应该是min(640/1920, 640/1080))
        expected_scale = min(640/1920, 640/1080)
        assert abs(scale_info['scale'] - expected_scale) < 1e-5

    def test_finalize_coordinate_transform(self, detector):
        """验证坐标转换正确性"""
        # 模拟归一化坐标 [0, 1]
        outputs = [np.array([
            [0.5, 0.5, 0.6, 0.6, 0.9, 0]  # 中心附近的框
        ])]

        scale_info = {
            'scale': 0.5,
            'pad_w': 50,
            'pad_h': 50,
            'new_w': 960,
            'new_h': 540
        }

        original_shape = (1080, 1920)

        detections = detector._finalize_inference(outputs, scale_info, original_shape, 0.5)

        # 验证坐标在原图范围内
        box = detections[0][0]
        assert 0 <= box[0] < original_shape[1]  # x1
        assert 0 <= box[1] < original_shape[0]  # y1
        assert box[0] < box[2] <= original_shape[1]  # x2
        assert box[1] < box[3] <= original_shape[0]  # y2

    def test_end_to_end_inference(self, detector):
        """端到端推理测试"""
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        detections, original_shape = detector(image, conf_thres=0.5)

        assert isinstance(detections, list)
        assert original_shape == (1080, 1920)
        assert detections[0].shape[1] == 6  # xyxy + conf + class
```

---

## Best Practices (最佳实践)

### 1. 类型提示

始终为方法添加完整的类型提示:

```python
from typing import List, Tuple, Optional
import numpy as np

def _postprocess(
    self,
    prediction: List[np.ndarray],  # 明确类型
    conf_thres: float,
    **kwargs
) -> List[np.ndarray]:  # 明确返回类型
    """文档字符串"""
    pass
```

### 2. 文档字符串

使用Google风格的docstring:

```python
def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple:
    """
    Static preprocessing method for RT-DETR.

    Args:
        image: Input image in BGR format, shape [H, W, C]
        input_shape: Target input size (height, width)

    Returns:
        Tuple containing:
            - input_tensor: Preprocessed tensor, shape [1, 3, H, W]
            - scale_info: Dictionary with scaling and padding information

    Raises:
        ValueError: If image dimensions are invalid

    Example:
        >>> image = cv2.imread('test.jpg')
        >>> tensor, scale = RTDETROnnx._preprocess_static(image, (640, 640))
        >>> print(tensor.shape)
        (1, 3, 640, 640)
    """
    pass
```

### 3. 异常处理

在方法开始处验证输入:

```python
def _postprocess(self, prediction, conf_thres, **kwargs):
    # 输入验证
    if not isinstance(prediction, list):
        raise TypeError(f"Expected list, got {type(prediction)}")

    if not (0 <= conf_thres <= 1):
        raise ValueError(f"conf_thres must be in [0, 1], got {conf_thres}")

    # 主要逻辑
    ...
```

### 4. 日志记录

在关键步骤添加日志:

```python
def _prepare_inference(self, image, conf_thres, **kwargs):
    self.logger.debug(f"Preparing inference for image shape {image.shape}")

    super()._prepare_inference(image, conf_thres, **kwargs)

    if kwargs.get('is_double_layer'):
        self.logger.info("Using double-layer preprocessing")
```

### 5. 性能优化

避免不必要的复制和转换:

```python
# ❌ 不推荐: 多次复制数组
def _postprocess(self, prediction, conf_thres, **kwargs):
    pred = prediction[0].copy()  # 复制1
    pred = pred[pred[:, 4] > conf_thres].copy()  # 复制2
    return [pred]

# ✅ 推荐: 就地操作或单次复制
def _postprocess(self, prediction, conf_thres, **kwargs):
    pred = prediction[0]
    mask = pred[:, 4] > conf_thres
    return [pred[mask]]  # 单次复制
```

### 6. 单元测试覆盖

确保每个方法都有对应的单元测试:

```python
class TestMyOnnx:
    def test_postprocess(self):
        """测试后处理"""
        pass

    def test_preprocess_static(self):
        """测试静态预处理"""
        pass

    def test_prepare_inference(self):
        """测试准备阶段 (如果重写了)"""
        pass

    def test_end_to_end(self):
        """测试完整推理流程"""
        pass
```

---

## Common Issues (常见问题)

### Q1: 实例化子类时报`TypeError: Can't instantiate abstract class`?

**原因**: 子类未实现所有抽象方法 (`_postprocess`, `_preprocess_static`)

**解决**:
```python
# 确保实现了2个抽象方法
class MyOnnx(BaseOnnx):
    def _postprocess(self, prediction, conf_thres, **kwargs):
        # 实现
        pass

    @staticmethod
    @abstractmethod
    def _preprocess_static(image, input_shape):
        # 实现
        pass
```

### Q2: `_preprocess_static`装饰器顺序错误导致`TypeError`?

**原因**: 装饰器顺序错误,应该是`@staticmethod`在外层

**错误示例**:
```python
@abstractmethod      # ❌ 错误: abstractmethod在外层
@staticmethod
def _preprocess_static(image, input_shape):
    pass
```

**正确示例**:
```python
@staticmethod        # ✅ 正确: staticmethod在外层
@abstractmethod
def _preprocess_static(image, input_shape):
    pass
```

### Q3: 推理结果坐标超出图像边界?

**原因**: `_finalize_inference()`中未正确转换坐标或未裁剪边界

**解决**:
```python
def _finalize_inference(self, outputs, scale, original_shape, conf_thres, **kwargs):
    detections = self._postprocess(outputs, conf_thres, **kwargs)

    for det in detections:
        if det.size > 0:
            # 坐标转换
            det[:, :4] = self._rescale_boxes(det[:, :4], scale, original_shape)

            # 裁剪到原图边界
            det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, original_shape[1])  # x坐标
            det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, original_shape[0])  # y坐标

    return detections
```

### Q4: 推理速度慢?

**原因**: 可能的原因包括:
- 模型未初始化 (每次推理都重新加载)
- 预处理效率低 (多次复制/转换)
- 未使用GPU

**解决**:
1. 确保模型懒加载:
```python
def _prepare_inference(self, image, conf_thres, **kwargs):
    self._ensure_initialized()  # 仅首次推理加载模型
```

2. 优化预处理:
```python
# 使用cv2.resize替代PIL
resized = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
```

3. 检查GPU使用:
```python
detector = MyOnnx(onnx_path='model.onnx', device='cuda')  # 确保使用GPU
```

### Q5: 如何调试推理流程?

**解决**: 使用日志和断点:

```python
def __call__(self, image, conf_thres, **kwargs):
    import time

    self.logger.debug(f"Input image shape: {image.shape}")

    # 阶段1
    start = time.perf_counter()
    self._prepare_inference(image, conf_thres, **kwargs)
    self.logger.debug(f"Prepare time: {(time.perf_counter() - start) * 1000:.2f}ms")

    # 阶段2
    start = time.perf_counter()
    self._execute_inference(self._context.input_tensor)
    self.logger.debug(f"Execute time: {(time.perf_counter() - start) * 1000:.2f}ms")

    # 阶段3
    start = time.perf_counter()
    detections = self._finalize_inference(...)
    self.logger.debug(f"Finalize time: {(time.perf_counter() - start) * 1000:.2f}ms")

    self.logger.debug(f"Detected {len(detections[0])} objects")

    return detections, self._context.original_shape
```

---

## Checklist (实现检查清单)

在提交代码前,请确认以下检查项:

### 必需实现

- [ ] 实现了`_postprocess()`抽象方法
- [ ] 实现了`_preprocess_static()`抽象方法
- [ ] `_preprocess_static()`使用正确的装饰器顺序 (`@staticmethod` -> `@abstractmethod`)
- [ ] 所有方法添加了类型提示
- [ ] 所有方法添加了docstring

### 可选重写

- [ ] 如需特殊准备逻辑,重写了`_prepare_inference()`
- [ ] 如需多次推理,重写了`_execute_inference()`
- [ ] 如需特殊后处理,重写了`_finalize_inference()`

### 测试和文档

- [ ] 编写了单元测试 (至少测试`_postprocess`和`_preprocess_static`)
- [ ] 编写了端到端集成测试
- [ ] 测试通过率100%
- [ ] 更新了`infer_onnx/CLAUDE.md`文档

### 性能和质量

- [ ] 推理延迟 < 50ms (640x640输入, GPU)
- [ ] 无明显内存泄漏
- [ ] 代码通过pylint检查 (评分 > 8.0)
- [ ] 代码通过mypy类型检查

---

## Next Steps (下一步)

完成BaseOnnx子类实现后:

1. **运行合约测试**: `pytest tests/contract/test_baseonnx_contract.py -v`
2. **运行性能测试**: `pytest tests/performance/ -v --benchmark-only`
3. **生成覆盖率报告**: `pytest --cov=infer_onnx --cov-report=html`
4. **更新文档**: 在`infer_onnx/CLAUDE.md`中添加新模型说明
5. **提交代码**: 创建PR并请求代码审查

---

*Quickstart generated: 2025-10-09*
*For more details, see [data-model.md](./data-model.md) and [contracts/baseonnx_api.md](./contracts/baseonnx_api.md)*
