# Quickstart: BaseORT结果包装类

**Date**: 2025-11-05
**Feature**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)

## 简介

Result类是BaseORT推理结果的面向对象包装，提供便捷的属性访问、索引操作、可视化和数据转换功能。本快速入门指南展示了Result类的核心用法和常见场景。

---

## 基础用法

### 1. 获取Result对象

**从BaseORT推理获取**（推荐方式）:

```python
from onnxtools import create_detector

# 创建检测器
detector = create_detector(
    model_type='rtdetr',
    onnx_path='models/rtdetr.onnx',
    conf_thres=0.5
)

# 执行推理，返回Result对象
result = detector(image)  # image: np.ndarray

# 现在可以直接访问Result对象的属性
print(f"检测到 {len(result)} 个目标")
```

**手动创建Result对象**（测试或调试）:

```python
from onnxtools import Result
import numpy as np

# 模拟检测结果
result = Result(
    boxes=np.array([[10, 20, 100, 150], [200, 300, 400, 500]]),
    scores=np.array([0.95, 0.87]),
    class_ids=np.array([0, 1]),
    orig_img=image,  # 原始图像numpy数组
    orig_shape=(640, 640),  # 原始图像形状
    names={0: 'vehicle', 1: 'plate'}  # 类别名称
)
```

---

### 2. 访问检测结果

**基础属性访问**:

```python
# 获取检测数量
num_detections = len(result)
print(f"检测目标数量: {num_detections}")

# 访问边界框（xyxy格式）
boxes = result.boxes
print(f"边界框形状: {boxes.shape}")  # (N, 4)

# 访问置信度分数
scores = result.scores
print(f"置信度: {scores}")  # [0.95, 0.87, ...]

# 访问类别ID
class_ids = result.class_ids
print(f"类别ID: {class_ids}")  # [0, 1, ...]

# 访问类别名称映射
names = result.names
print(f"类别名称: {names}")  # {0: 'vehicle', 1: 'plate'}

# 访问原始图像信息
orig_shape = result.orig_shape
print(f"原始图像尺寸: {orig_shape}")  # (640, 640)
```

**只读保护**:

```python
# ❌ 尝试赋值会抛出AttributeError
try:
    result.boxes = new_boxes
except AttributeError as e:
    print(f"错误: {e}")  # can't set attribute 'boxes'

# ✅ 但可以修改内部数组元素（浅层不可变设计）
result.boxes[0] = [15, 25, 105, 155]  # 允许
```

---

### 3. 索引和切片

**获取单个检测**:

```python
# 通过索引访问第一个检测
first_detection = result[0]
print(f"第一个检测的边界框: {first_detection.boxes}")
print(f"第一个检测的置信度: {first_detection.scores}")

# 访问最后一个检测
last_detection = result[-1]
```

**切片访问**:

```python
# 获取前3个检测
top3 = result[:3]
print(f"前3个检测数量: {len(top3)}")

# 获取置信度最高的检测（假设已按置信度排序）
high_conf = result[0:5]
```

**遍历所有检测**:

```python
for i, detection in enumerate(result):
    box = detection.boxes[0]
    score = detection.scores[0]
    class_id = detection.class_ids[0]
    class_name = detection.names.get(class_id, 'unknown')

    print(f"检测 {i}: {class_name} (置信度: {score:.2f})")
    print(f"  边界框: x1={box[0]:.1f}, y1={box[1]:.1f}, "
          f"x2={box[2]:.1f}, y2={box[3]:.1f}")
```

---

## 常见场景

### 场景1: 过滤检测结果

**按置信度过滤**:

```python
# 保留置信度>=0.7的检测
high_conf_result = result.filter(conf_threshold=0.7)
print(f"高置信度检测: {len(high_conf_result)} 个")
```

**按类别过滤**:

```python
# 仅保留车辆（class_id=0）
vehicles_only = result.filter(classes=[0])
print(f"车辆检测: {len(vehicles_only)} 个")

# 保留车辆和车牌（class_id=0和1）
vehicles_and_plates = result.filter(classes=[0, 1])
```

**组合过滤**:

```python
# 保留置信度>=0.8且类别为车牌（class_id=1）的检测
high_conf_plates = result.filter(
    conf_threshold=0.8,
    classes=[1]
)
print(f"高置信度车牌: {len(high_conf_plates)} 个")
```

---

### 场景2: 可视化检测结果

**绘制标注图像**:

```python
# 使用默认可视化风格
annotated_img = result.plot()

# 使用自定义annotator预设
annotated_img_debug = result.plot(annotator_preset='debug')
annotated_img_privacy = result.plot(annotator_preset='privacy')

# 保存标注图像
import cv2
cv2.imwrite('output.jpg', annotated_img)
```

**显示结果**:

```python
# 在窗口中显示（会阻塞）
result.show(window_name="检测结果")

# 等价于:
# annotated_img = result.plot()
# cv2.imshow("检测结果", annotated_img)
# cv2.waitKey(0)
```

**保存到文件**:

```python
# 直接保存标注后的图像
result.save('output/result.jpg')

# 等价于:
# annotated_img = result.plot()
# cv2.imwrite('output/result.jpg', annotated_img)
```

---

### 场景3: 获取统计信息

**生成摘要**:

```python
# 获取检测统计信息
summary = result.summary()

print(f"检测总数: {summary['total_detections']}")
print(f"平均置信度: {summary['avg_confidence']:.2f}")
print(f"最小置信度: {summary['min_confidence']:.2f}")
print(f"最大置信度: {summary['max_confidence']:.2f}")

# 各类别数量
class_counts = summary['class_counts']
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} 个")
```

**输出示例**:
```
检测总数: 5
平均置信度: 0.89
最小置信度: 0.75
最大置信度: 0.98
  vehicle: 3 个
  plate: 2 个
```

---

### 场景4: 与Supervision集成

**转换为Supervision Detections**:

```python
# 转换为supervision.Detections对象
detections = result.to_supervision()

# 使用Supervision的高级可视化工具
import supervision as sv

# 使用Supervision的annotator
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 绘制标注
annotated_frame = box_annotator.annotate(
    scene=result.orig_img.copy(),
    detections=detections
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections
)
```

---

### 场景5: 向后兼容（Deprecated）

**转换为字典格式**（不推荐，将在第2个迭代移除）:

```python
import warnings

# 旧代码可能需要字典格式
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    result_dict = result.to_dict()

# result_dict 包含:
# {
#     'boxes': np.ndarray,
#     'scores': np.ndarray,
#     'class_ids': np.ndarray
# }

# ⚠️ 推荐迁移到属性访问:
# boxes = result.boxes  # 而非 result.to_dict()['boxes']
```

---

## 处理边界情况

### 空检测结果

```python
# 当没有检测到目标时
empty_result = Result(
    boxes=None,
    scores=None,
    class_ids=None,
    orig_img=image,
    orig_shape=(640, 640)
)

# 空检测的行为
print(len(empty_result))  # 0
print(empty_result.boxes.shape)  # (0, 4)
print(empty_result.scores.shape)  # (0,)

# 尝试索引会抛出IndexError
try:
    first = empty_result[0]
except IndexError:
    print("空结果，无法索引")

# 过滤空结果仍返回空Result对象
filtered = empty_result.filter(conf_threshold=0.5)
print(len(filtered))  # 0
```

### 缺少原始图像

```python
# 创建不包含原始图像的Result（仅数据）
data_only_result = Result(
    boxes=np.array([[10, 20, 30, 40]]),
    scores=np.array([0.9]),
    class_ids=np.array([0]),
    orig_img=None,  # 无原始图像
    orig_shape=(640, 640)
)

# 可以访问数据属性
print(data_only_result.boxes)  # OK

# 但不能可视化
try:
    data_only_result.plot()
except ValueError as e:
    print(f"错误: {e}")  # orig_img is None, cannot plot without original image
```

---

## 性能提示

### 1. 内存优化

```python
# 索引和切片使用numpy视图（不拷贝数据）
subset = result[0:10]  # 视图，内存高效

# 如果需要独立副本（罕见场景）
import copy
independent_result = copy.deepcopy(result)
```

### 2. 避免不必要的转换

```python
# ❌ 不推荐：频繁转换为字典
for i in range(len(result)):
    data = result.to_dict()  # 每次都创建新字典，低效
    process(data['boxes'][i])

# ✅ 推荐：直接访问属性
boxes = result.boxes  # 一次访问
for i in range(len(result)):
    process(boxes[i])  # 高效
```

### 3. 批量操作

```python
# ✅ 推荐：批量过滤后再处理
filtered_result = result.filter(conf_threshold=0.7, classes=[1])
for detection in filtered_result:
    process(detection)

# ❌ 不推荐：逐个检查
for detection in result:
    if detection.scores[0] >= 0.7 and detection.class_ids[0] == 1:
        process(detection)
```

---

## 完整示例

### 示例1: 端到端检测流程

```python
from onnxtools import create_detector
import cv2

# 1. 创建检测器
detector = create_detector(
    model_type='rtdetr',
    onnx_path='models/rtdetr.onnx',
    conf_thres=0.5,
    iou_thres=0.5
)

# 2. 加载图像
image = cv2.imread('input.jpg')

# 3. 执行推理
result = detector(image)

# 4. 过滤结果（可选）
high_conf_result = result.filter(conf_threshold=0.7)

# 5. 打印统计信息
summary = high_conf_result.summary()
print(f"高置信度检测: {summary['total_detections']} 个")
for class_name, count in summary['class_counts'].items():
    print(f"  {class_name}: {count} 个")

# 6. 可视化并保存
high_conf_result.save('output/high_conf_result.jpg')

# 7. 遍历检测结果
for i, detection in enumerate(high_conf_result):
    box = detection.boxes[0]
    score = detection.scores[0]
    class_id = detection.class_ids[0]
    class_name = detection.names[class_id]

    print(f"检测 {i+1}: {class_name} ({score:.2%})")
    print(f"  位置: ({box[0]:.0f}, {box[1]:.0f}) - ({box[2]:.0f}, {box[3]:.0f})")
```

### 示例2: 多模型管道（车辆+车牌+OCR）

```python
from onnxtools import create_detector, OcrORT
import cv2

# 1. 创建检测器
vehicle_detector = create_detector('rtdetr', 'models/vehicle_det.onnx')
plate_detector = create_detector('rtdetr', 'models/plate_det.onnx')
ocr_model = OcrORT('models/ocr.onnx', character="省会0123456789ABCD...")

# 2. 检测车辆
image = cv2.imread('input.jpg')
vehicle_result = vehicle_detector(image)

# 3. 对每个车辆检测车牌
for vehicle in vehicle_result:
    # 裁剪车辆区域
    box = vehicle.boxes[0].astype(int)
    vehicle_img = image[box[1]:box[3], box[0]:box[2]]

    # 检测车牌
    plate_result = plate_detector(vehicle_img)

    # 4. OCR识别车牌
    for plate in plate_result:
        plate_box = plate.boxes[0].astype(int)
        plate_img = vehicle_img[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]

        # 识别车牌文本
        ocr_output = ocr_model(plate_img)
        if ocr_output:
            plate_text, confidence, char_scores = ocr_output
            print(f"车牌号: {plate_text} (置信度: {confidence:.2%})")
```

---

## 常见问题

### Q1: Result对象可以修改吗？

A: Result对象采用**浅层不可变**设计：
- ❌ 不能重新赋值属性（`result.boxes = new_boxes`）
- ✅ 可以修改内部数组元素（`result.boxes[0] = [10, 20, 30, 40]`）
- 如需"修改"检测结果，创建新的Result对象（如通过`filter()`）

### Q2: 如何合并多个Result对象？

A: 当前版本不支持Result对象合并（Post-MVP功能）。临时方案：

```python
# 手动合并
combined = Result(
    boxes=np.vstack([result1.boxes, result2.boxes]),
    scores=np.concatenate([result1.scores, result2.scores]),
    class_ids=np.concatenate([result1.class_ids, result2.class_ids]),
    orig_img=result1.orig_img,  # 选择一个原图
    orig_shape=result1.orig_shape,
    names=result1.names
)
```

### Q3: 性能开销有多大？

A: 根据性能目标（SC-006, SC-002）：
- 创建Result对象: <5ms（20个检测目标）
- 可视化: <1秒（20个目标，640x640图像）
- 内存占用: <120%原始字典格式

实际基准测试将在实现后验证。

### Q4: to_dict()何时会被移除？

A: 根据澄清会话决策：
- **第1个迭代**：添加DeprecationWarning
- **第2个迭代**：完全移除to_dict()方法

请尽快迁移到属性访问方式。

---

## 下一步

- 查看[数据模型文档](./data-model.md)了解Result类的内部结构
- 查看[API合约](./contracts/result_api.yaml)了解完整的接口规范
- 查看[实施计划](./plan.md)了解开发路线图

---

**文档版本**: 1.0.0
**最后更新**: 2025-11-05
