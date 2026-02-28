# 模型支持列表

本文档基于推理类源码,详细描述项目支持的所有模型及其输入/输出规格、动态维度支持、前后处理流程。

> **最后更新**: 2026-02-27
> **兼容性**: Python 3.10+, ONNX Runtime 1.22.0+

---

## 检测模型

检测模型继承 `BaseORT` 基类，返回 `Result` 对象。

### 1. YOLO系列 (YoloORT)

**实现类**: `infer_onnx/onnx_yolo.py::YoloORT`
**支持版本**: YOLOv5, YOLOv8, YOLO11
**特点**: 成熟稳定,需要NMS后处理

#### 初始化参数

```python
YoloORT(
    onnx_path: str,
    input_shape: Tuple[int, int] = (640, 640),      # 输入尺寸 (H, W)
    conf_thres: float = 0.5,                        # 置信度阈值
    iou_thres: float = 0.5,                         # NMS IoU阈值
    multi_label: bool = True,                       # 多标签检测
    use_ultralytics_preprocess: bool = True,        # Ultralytics兼容预处理
    has_objectness: bool = False,                   # objectness分支
    providers: Optional[List[str]] = None           # ONNX Runtime providers
)
```

#### 输入输出规格

```python
# 输入 (基于BaseORT机制)
输入名称: 从模型自动读取 (通常 "images")
输入形状: [1, 3, H, W]  # NCHW格式
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
数据类型: float32
数值范围: [0.0, 1.0]  # 归一化RGB
颜色空间: RGB (从BGR转换)

# 输出 (基于 _postprocess 自适应处理)
输出形状: [B, N, 4+C] 或 [B, 4+C, N]  # 自动检测并转换
坐标格式: [x_center, y_center, width, height]
类别输出: [conf1, ..., confC]  # 每类独立置信度
最终输出: Result(boxes=[x1,y1,x2,y2], scores, class_ids)  # 经NMS后
```

#### 前处理流程

```python
# 源自 onnx_yolo.py::_preprocess_static()
# Letterbox: 保持宽高比 + padding
letterbox = UltralyticsLetterBox(new_shape=(640, 640))
input_tensor, scale, original_shape, ratio_pad = letterbox(image)
```

#### 后处理流程

```python
# 源自 onnx_yolo.py::_postprocess()
# 1. 格式自适应: [B,C,N] -> [B,N,C]
# 2. 坐标归一化检测并转换为像素坐标
# 3. NMS后处理 (multi_label, has_objectness)
# 4. 坐标还原: letterbox需考虑ratio_pad, 否则简单缩放
# 5. 返回 Result 对象
```

---

### 2. RT-DETR (RtdetrORT)

**实现类**: `infer_onnx/onnx_rtdetr.py::RtdetrORT`
**原始框架**: Ultralytics RT-DETR
**特点**: 端到端检测,无需NMS,300个query

#### 初始化参数

```python
RtdetrORT(
    onnx_path: str,
    input_shape: Tuple[int, int] = (640, 640),  # 输入尺寸
    conf_thres: float = 0.001,                  # 置信度阈值 (推荐低阈值)
    iou_thres: float = 0.5,                     # 未使用 (保持接口一致)
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [batch, 3, 640, 640]
数值范围: [0.0, 1.0]
颜色空间: RGB

# 输出
输出形状: [batch, 300, num_features]
         # 300 = query数量
         # num_features = 4 bbox + C classes
坐标格式: [x_center, y_center, width, height]  # 归一化 [0,1]
最终输出: Result(boxes, scores, class_ids)  # 排序+过滤后
```

#### 前处理流程

```python
# 源自 onnx_rtdetr.py::_preprocess_static()
# 直接Resize (不保持宽高比)
resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = rgb_image.astype(np.float32) / 255.0
tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
```

#### 后处理流程

```python
# 源自 onnx_rtdetr.py::_postprocess()
# 1. 分离bbox和scores
# 2. 智能归一化 (_smart_normalize_scores)
# 3. bbox缩放: 归一化坐标 * 640 -> 像素坐标
# 4. 坐标转换: xywh -> xyxy
# 5. 排序 + 置信度过滤
# 6. 返回 Result 对象
```

---

### 3. RF-DETR (RfdetrORT)

**实现类**: `infer_onnx/onnx_rfdetr.py::RfdetrORT`
**原始框架**: RF-DETR (ResNet + FPN + DETR)
**特点**: 双输出,ImageNet标准化,TopK选择

#### 初始化参数

```python
RfdetrORT(
    onnx_path: str,
    input_shape: Tuple[int, int] = (576, 576),  # 注意: 默认576×576
    conf_thres: float = 0.001,
    iou_thres: float = 0.5,  # 未使用
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [batch, 3, 576, 576]
数值范围: ImageNet标准化 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
颜色空间: RGB

# 输出 (双输出)
输出1 (pred_boxes): [batch, num_queries, 4]  # bbox, 归一化
输出2 (pred_logits): [batch, num_queries, C]  # 类别logits
最终输出: Result(boxes, scores, class_ids)  # TopK选择后
```

#### 前处理流程

```python
# 源自 onnx_rfdetr.py::_preprocess_static()
resized = cv2.resize(image, (576, 576))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = rgb.astype(np.float32) / 255.0

# ImageNet标准化 (关键差异)
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
normalized = (normalized - imagenet_mean) / imagenet_std
```

#### 后处理流程

```python
# 源自 onnx_rfdetr.py::_postprocess()
# 1. 应用sigmoid激活
# 2. TopK选择 (展平所有query×classes, 选top 300)
# 3. 坐标转换: cxcywh -> xyxy
# 4. 缩放到输入图像尺寸
# 5. 置信度过滤, 返回 Result 对象
```

---

## 分类模型

分类模型继承 `BaseClsORT` 基类，返回 `ClsResult` 对象，支持元组解包。

### 4. 颜色层级分类 (ColorLayerORT)

**实现类**: `infer_onnx/onnx_cls.py::ColorLayerORT`
**基类**: `BaseClsORT`
**功能**: 车牌颜色和层级分类 (双分支)
**特点**: 双输出,同时预测颜色和层级,返回 `ClsResult`

#### 初始化参数

```python
ColorLayerORT(
    onnx_path: str,
    color_map: Optional[Dict[int, str]] = None,    # 默认从config加载
    layer_map: Optional[Dict[int, str]] = None,    # 默认从config加载
    input_shape: Tuple[int, int] = (48, 168),
    conf_thres: float = 0.5,
    providers: Optional[List[str]] = None,
    plate_config_path: Optional[str] = None        # 外部配置文件
)
```

#### 输入输出规格

```python
# 输入
输入形状: [1, 3, 48, 168]
数值范围: [-1.0, 1.0]  # (x/255 - 0.5) / 0.5
颜色空间: RGB (从BGR转换)

# 输出 (双输出)
输出1 (color_logits): [batch, 5]  # 5个颜色类别
输出2 (layer_logits): [batch, 2]  # 2个层级类别
类别映射:
  color: {0:'black', 1:'blue', 2:'green', 3:'white', 4:'yellow'}
  layer: {0:'single', 1:'double'}
最终输出: ClsResult(labels=[color, layer], confidences=[c1, c2])
```

#### 使用示例

```python
from onnxtools import ColorLayerORT

classifier = ColorLayerORT('models/color_layer.onnx')
result = classifier(plate_image)  # ClsResult
print(f"Color: {result.labels[0]}, Layer: {result.labels[1]}")

# 元组解包 (向后兼容)
color, layer, conf = classifier(plate_image)
```

---

### 5. 车辆属性分类 (VehicleAttributeORT)

**实现类**: `infer_onnx/onnx_cls.py::VehicleAttributeORT`
**基类**: `BaseClsORT`
**功能**: 车辆类型和颜色分类 (多标签)
**特点**: 单输出拆分为两个分支,sigmoid已在模型内部应用

#### 初始化参数

```python
VehicleAttributeORT(
    onnx_path: str,
    type_map: Optional[Dict[int, str]] = None,     # 默认13类车型
    color_map: Optional[Dict[int, str]] = None,     # 默认11种颜色
    input_shape: Tuple[int, int] = (224, 224),
    conf_thres: float = 0.5,
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [1, 3, 224, 224]
数值范围: [0.0, 1.0]  # 仅 /255, 无mean/std
颜色空间: RGB (从BGR转换)

# 输出 (单输出,内部拆分)
输出: [batch, 24]  # 已sigmoid, 前13为车型, 后11为颜色
类别映射:
  type: {0:'car', 1:'truck', 2:'bus', ..., 12:'school_bus'}
  color: {0:'black', 1:'white', 2:'gray', ..., 10:'other'}
最终输出: ClsResult(labels=[type, color], confidences=[c1, c2])
```

#### 使用示例

```python
from onnxtools import VehicleAttributeORT

classifier = VehicleAttributeORT('models/vehicle_attribute.onnx')
result = classifier(vehicle_image)
print(f"Type: {result.labels[0]}, Color: {result.labels[1]}")

# 元组解包
vehicle_type, color, conf = classifier(vehicle_image)
```

---

### 6. 头盔佩戴分类 (HelmetORT)

**实现类**: `infer_onnx/onnx_cls.py::HelmetORT`
**基类**: `BaseClsORT`
**模型架构**: ConvNeXtV2-Pico
**功能**: 头盔佩戴二分类 (单分支)
**精度**: ~94% accuracy
**特点**: 固定batch=4 ONNX模型,LetterBox预处理,ImageNet归一化

#### 初始化参数

```python
HelmetORT(
    onnx_path: str,
    helmet_map: Optional[Dict[int, str]] = None,   # 默认 {0:'normal', 1:'helmet_missing'}
    input_shape: Tuple[int, int] = (128, 128),
    conf_thres: float = 0.5,
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [4, 3, 128, 128]  # 固定batch=4 (内部透明处理,调用方无感知)
数值范围: ImageNet标准化 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
颜色空间: RGB (从BGR转换)
预处理: LetterBox (保持宽高比, padding值=127)

# 输出
输出: [4, 2]  # raw logits, 内部应用softmax
类别映射:
  {0: 'normal', 1: 'helmet_missing'}
最终输出: ClsResult(labels=[status], confidences=[conf])
```

#### 固定batch=4处理

该ONNX模型的batch维度固定为4。`HelmetORT` 通过重写 `_execute_inference` 透明处理：
- 将单张输入 `[1,3,128,128]` 复制填充到 `[4,3,128,128]`
- 推理后只取第一个结果 `outputs[:1]`
- 上层调用完全无感知,接口与其他分类器一致

> 建议后续重新导出动态batch的ONNX模型,届时只需删除 `_execute_inference` 重写即可。

#### 前处理流程

```python
# 源自 onnx_cls.py::HelmetORT.preprocess()
# 1. BGR -> RGB
# 2. LetterBox resize到128×128 (保持宽高比, padding=127)
# 3. ImageNet归一化: (x/255 - mean) / std
# 4. HWC -> CHW, 添加batch维度
```

#### 使用示例

```python
from onnxtools import HelmetORT

helmet = HelmetORT('models/convnextv2_pico_2cls_i128-1226-20_batch4_simplified.onnx')
result = helmet(head_image)  # ClsResult
print(f"Status: {result.labels[0]}, Conf: {result.avg_confidence:.3f}")

# 元组解包 (单分支: label, conf)
label, conf = helmet(head_image)
# label: 'normal' 或 'helmet_missing'
```

---

## OCR模型

OCR模型为独立推理类,不继承BaseORT/BaseClsORT,返回 `Optional[Tuple]`。

### 7. OCR模型 (OcrORT)

**实现类**: `infer_onnx/onnx_ocr.py::OcrORT`
**支持模型**: rec_ppocr_v3, rec_ppocr_v5
**功能**: 车牌号码识别
**特点**: 支持单层/双层车牌,倾斜校正,分割拼接

#### 初始化参数

```python
OcrORT(
    onnx_path: str,
    character: List[str],                       # OCR字符字典 (必需)
    input_shape: Tuple[int, int] = (48, 168),  # 输入尺寸
    conf_thres: float = 0.5,                    # 平均置信度阈值
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [1, 3, 48, 168]
数值范围: [-1.0, 1.0]
颜色空间: BGR (不转换)

# 输出
输出形状: [batch, seq_len, num_classes]
         # seq_len: 序列长度 (动态, 通常21)
         # num_classes: 字符类别数 (例如85)
输出格式: CTC格式, 每个时间步的字符概率分布
最终输出: Optional[(text, avg_confidence, char_confidences)]
```

#### 前处理流程

```python
# 源自 onnx_ocr.py
# 1. 倾斜检测和校正 (Hough线变换 + 仿射变换)
# 2. 双层车牌处理 (if is_double_layer=True):
#    - CLAHE对比度增强
#    - 水平投影寻找分割线
#    - 分割上下层, 拼接成单层
# 3. 保持宽高比resize (max_w=168)
# 4. 归一化到 [-1, 1]
# 5. 右侧padding到168
```

#### 后处理流程

```python
# CTC解码:
# 1. argmax提取字符索引
# 2. 移除连续重复字符
# 3. 过滤blank token
# 4. 映射索引到字符
# 5. 计算平均置信度
```

---

## 模型对比表

### 检测模型对比

| 模型 | 推理类 | 基类 | 输入尺寸 | 前处理 | 后处理 | 返回类型 | 推荐场景 |
|------|--------|------|---------|--------|--------|---------|---------|
| YOLO | `YoloORT` | BaseORT | 640×640 | Letterbox | NMS | `Result` | 实时检测 |
| RT-DETR | `RtdetrORT` | BaseORT | 640×640 | Resize | 排序过滤 | `Result` | 平衡场景 |
| RF-DETR | `RfdetrORT` | BaseORT | 576×576 | ImageNet | TopK | `Result` | 高精度 |

### 分类模型对比

| 模型 | 推理类 | 基类 | 输入尺寸 | 分支数 | 归一化 | 返回类型 | 推荐场景 |
|------|--------|------|---------|--------|--------|---------|---------|
| 颜色层级 | `ColorLayerORT` | BaseClsORT | 48×168 | 2 (颜色+层级) | [-1,1] | `ClsResult` | 车牌属性 |
| 车辆属性 | `VehicleAttributeORT` | BaseClsORT | 224×224 | 2 (车型+颜色) | [0,1] | `ClsResult` | 车辆分类 |
| 头盔检测 | `HelmetORT` | BaseClsORT | 128×128 | 1 (佩戴状态) | ImageNet | `ClsResult` | 安全检测 |

### OCR模型

| 模型 | 推理类 | 输入尺寸 | 特殊处理 | 返回类型 | 推荐场景 |
|------|--------|---------|---------|---------|---------|
| 车牌OCR | `OcrORT` | 48×168 | 倾斜校正+双层分割 | `Optional[Tuple]` | 车牌识别 |

### 关键差异总结

| 特性 | YOLO | RT-DETR | RF-DETR | ColorLayer | VehicleAttr | Helmet | OCR |
|------|------|---------|---------|-----------|-------------|--------|-----|
| **预处理** | Letterbox | Resize | Resize | Resize | Resize | LetterBox | 倾斜+拼接 |
| **归一化** | [0,1] | [0,1] | ImageNet | [-1,1] | [0,1] | ImageNet | [-1,1] |
| **颜色空间** | RGB | RGB | RGB | RGB | RGB | RGB | BGR |
| **后处理** | NMS | 排序 | TopK | Softmax | Sigmoid | Softmax | CTC |
| **NMS需求** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **固定Batch** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (4) | ❌ |

---

## 常见问题

### Q1: 为什么不同模型的conf_thres默认值不同？

**A**: 基于模型输出特性:
- **YOLO (0.5)**: 输出是类别置信度,范围[0,1]
- **RT-DETR/RF-DETR (0.001)**: 输出300个queries,需要低阈值保留候选,无NMS依靠排序过滤

### Q2: 输入尺寸是如何确定的？

**A**: `BaseORT` 和 `BaseClsORT` 使用智能加载机制:

1. 从ONNX模型元数据读取输入形状
2. 如果H, W是固定整数 → 使用模型值
3. 如果是动态符号 → 使用用户指定的 `input_shape`
4. 查看模型尺寸:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("model.onnx")
   print(session.get_inputs()[0].shape)
   ```

### Q3: HelmetORT的batch=4是怎么回事？

**A**: 当前ONNX模型在导出时固定了batch=4。`HelmetORT` 重写了 `_execute_inference` 方法:
- 单张输入复制4份填充batch维度
- 推理后只取第一个结果
- 调用方完全无感知,接口与其他分类器一致

建议后续重新导出动态batch模型,届时只需删除该重写方法即可。

### Q4: 检测器和分类器的区别？

**A**: 项目提供三类推理架构:

| 类型 | 基类 | 返回类型 | 用途 |
|------|------|---------|------|
| 检测器 | `BaseORT` | `Result` | 目标检测,输出框+置信度+类别 |
| 分类器 | `BaseClsORT` | `ClsResult` | 图像分类,支持单/双/多分支 |
| OCR | 独立 | `Optional[Tuple]` | 序列识别,可变长度字符输出 |

### Q5: ClsResult怎么使用？

**A**: `ClsResult` 支持属性访问和元组解包:

```python
result = classifier(image)

# 属性访问
result.labels         # ['blue', 'single']
result.confidences    # [0.95, 0.88]
result.avg_confidence # 0.915

# 元组解包 (向后兼容)
# 单分支: label, conf = result
# 双分支: label1, label2, conf = result
```

---

## 附录

### 模型文件命名规范

```
检测模型:
- yolo11n.onnx                                          # YOLO11 Nano
- rtdetr-2024080100.onnx                                 # RT-DETR, 日期戳
- rfdetr-20250811.onnx                                   # RF-DETR, 日期戳

分类模型:
- color_layer.onnx                                       # 颜色层级分类
- vehicle_attribute.onnx                                 # 车辆属性分类
- convnextv2_pico_2cls_i128-1226-20_batch4_simplified.onnx  # 头盔分类

OCR模型:
- ocr.onnx                                              # OCR模型
```

### 相关文档

- [项目总览](../README.md)
- [根目录CLAUDE.md](../CLAUDE.md)
- [推理引擎文档](../onnxtools/infer_onnx/CLAUDE.md)
- [测试文档](../tests/CLAUDE.md)

---

**文档维护**: 本文档基于推理类源码生成,与代码同步更新。

**最后更新**: 2026-02-27
**作者**: yyq19990828
**版本**: v3.0.0
