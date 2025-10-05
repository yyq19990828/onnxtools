# Quickstart: Supervision Annotators扩展使用指南

**Feature**: 添加更多Supervision Annotators类型
**Date**: 2025-09-30
**Estimated Time**: 15-20分钟

## 概述

本快速入门指南演示如何使用新增的13种supervision annotators实现丰富的可视化效果，包括5种预设场景和自定义组合方式。

## 前置条件

```bash
# Python环境
python >= 3.10

# 必需依赖
pip install supervision>=0.26.0 opencv-python numpy pillow

# 可选：性能基准测试
pip install pytest-benchmark
```

## 1. 基础使用 - 单个Annotator

### 示例1: 圆角边框
```python
from utils.annotator_factory import AnnotatorFactory, AnnotatorType
import supervision as sv
import cv2

# 加载图像和检测结果
image = cv2.imread("test.jpg")
detections = sv.Detections(...)  # 从检测模型获取

# 创建圆角边框annotator
annotator = AnnotatorFactory.create(
    AnnotatorType.ROUND_BOX,
    {'thickness': 3, 'roundness': 0.3}
)

# 应用标注
result = annotator.annotate(image, detections)
cv2.imshow("Result", result)
cv2.waitKey(0)
```

### 示例2: 置信度条形图
```python
# 创建置信度条形图annotator
annotator = AnnotatorFactory.create(
    AnnotatorType.PERCENTAGE_BAR,
    {
        'height': 16,
        'width': 80,
        'position': 'top_left'
    }
)

result = annotator.annotate(image, detections)
```

### 示例3: 隐私保护 - 模糊车牌
```python
# 创建模糊annotator
annotator = AnnotatorFactory.create(
    AnnotatorType.BLUR,
    {'kernel_size': 15}
)

# 仅对车牌类别应用模糊
plate_detections = detections[detections.class_id == 1]  # class_id=1为车牌
result = annotator.annotate(image, plate_detections)
```

## 2. 组合使用 - AnnotatorPipeline

### 示例4: 圆角边框 + 置信度条 + 标签
```python
from utils.annotator_factory import AnnotatorPipeline

# 创建pipeline并添加多个annotator (链式调用)
pipeline = (AnnotatorPipeline()
    .add(AnnotatorType.ROUND_BOX, {'thickness': 3, 'roundness': 0.3})
    .add(AnnotatorType.PERCENTAGE_BAR, {'height': 16, 'width': 80})
    .add(AnnotatorType.RICH_LABEL, {'font_size': 18})
)

# 检查冲突警告
warnings = pipeline.check_conflicts()
if warnings:
    print("Warnings:", warnings)

# 应用所有annotator
result = pipeline.annotate(image, detections)
```

### 示例5: 高对比展示 - 填充 + 背景变暗
```python
pipeline = (AnnotatorPipeline()
    .add(AnnotatorType.COLOR, {'opacity': 0.3})
    .add(AnnotatorType.BACKGROUND_OVERLAY, {'opacity': 0.5, 'color': 'black'})
)

result = pipeline.annotate(image, detections)
```

## 3. 预设场景 - 快速配置

### 示例6: 使用标准检测模式
```python
from utils.annotator_factory import VisualizationPreset, Presets

# 加载预设场景
preset = VisualizationPreset.from_yaml(Presets.STANDARD)
print(f"Using preset: {preset.name}")

# 创建pipeline
pipeline = preset.create_pipeline()

# 应用可视化
result = pipeline.annotate(image, detections)
```

### 示例7: 隐私保护模式 (车牌模糊)
```python
# 加载隐私保护预设
preset = VisualizationPreset.from_yaml(Presets.PRIVACY)
pipeline = preset.create_pipeline()

# 应用 (自动对车牌类别应用模糊)
result = pipeline.annotate(image, detections)
```

### 示例8: 调试分析模式 (详细信息)
```python
# 加载调试预设
preset = VisualizationPreset.from_yaml(Presets.DEBUG)
pipeline = preset.create_pipeline()

# 应用 (圆角框 + 置信度条 + 详细标签)
result = pipeline.annotate(image, detections)
```

## 4. 自定义预设场景

### 示例9: 创建自定义YAML预设
```yaml
# custom_presets.yaml
presets:
  my_custom:
    name: "我的自定义模式"
    description: "点标记 + 三角形 + 光晕"
    annotators:
      - type: dot
        radius: 6
        position: center
      - type: triangle
        base: 25
        height: 25
        position: top_center
      - type: halo
        opacity: 0.4
        kernel_size: 50
```

```python
# 加载自定义预设
preset = VisualizationPreset.from_yaml("my_custom", "custom_presets.yaml")
pipeline = preset.create_pipeline()
result = pipeline.annotate(image, detections)
```

## 5. 性能基准测试

### 示例10: 测试annotator性能
```python
import time
import numpy as np

# 准备测试数据
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
test_detections = sv.Detections(
    xyxy=np.random.rand(20, 4) * 640,
    confidence=np.random.rand(20),
    class_id=np.random.randint(0, 2, 20)
)

# 测试单个annotator
annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {'thickness': 2})

start = time.perf_counter()
for _ in range(100):
    result = annotator.annotate(test_image.copy(), test_detections)
elapsed = (time.perf_counter() - start) * 1000 / 100

print(f"RoundBoxAnnotator: {elapsed:.2f}ms per frame (20 objects)")
```

### 示例11: 使用pytest-benchmark
```python
# tests/performance/test_annotator_benchmark.py
import pytest
from utils.annotator_factory import AnnotatorFactory, AnnotatorType

@pytest.mark.benchmark(group="annotators")
def test_blur_annotator_performance(benchmark, test_image, test_detections):
    """Benchmark BlurAnnotator rendering time."""
    annotator = AnnotatorFactory.create(AnnotatorType.BLUR, {'kernel_size': 15})
    result = benchmark(annotator.annotate, test_image, test_detections)
    assert result.shape == test_image.shape
```

运行基准测试：
```bash
pytest tests/performance/test_annotator_benchmark.py --benchmark-only
```

## 6. 常见问题

### Q: 如何同时使用多个几何标记？
A: 使用AnnotatorPipeline组合，但会收到冲突警告（可忽略）

### Q: 预设场景可以修改吗？
A: 可以，直接编辑`configs/visualization_presets.yaml`或创建自定义YAML

### Q: 如何提高渲染性能？
A: 1) 减少annotator数量; 2) 使用轻量级annotator(如DotAnnotator); 3) 降低图像分辨率

### Q: 冲突警告是什么意思？
A: 表示某些annotator组合可能视觉效果重叠，但仍允许执行

## 7. 完整示例 - 视频处理

```python
import cv2
from utils.annotator_factory import VisualizationPreset, Presets
from utils.pipeline import initialize_models, process_frame

# 初始化模型
models = initialize_models(args)
detector, _, _, _, class_names, colors = models

# 加载预设场景
preset = VisualizationPreset.from_yaml(Presets.DEBUG)
pipeline = preset.create_pipeline()

# 处理视频
cap = cv2.VideoCapture("input.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测
    detections = detector.predict(frame)

    # 应用可视化
    result = pipeline.annotate(frame, detections)

    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

**下一步**: 运行合约测试 `pytest tests/contract/test_annotator_*`