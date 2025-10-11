# 模型支持列表

本文档基于推理类源码,详细描述项目支持的所有模型及其输入/输出规格、动态维度支持、前后处理流程。

> **最后更新**: 2025-10-11
> **兼容性**: Python 3.10+, ONNX Runtime 1.22.0+

---

## 检测模型

### 1. YOLO系列 (YoloOnnx)

**实现类**: `infer_onnx/onnx_yolo.py::YoloOnnx`
**支持版本**: YOLOv5, YOLOv8, YOLO11
**特点**: 成熟稳定,需要NMS后处理

#### 初始化参数

```python
# 源自 onnx_yolo.py::__init__()
YoloOnnx(
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
# 输入 (基于BaseOnnx懒加载机制)
输入名称: 从模型自动读取 (通常 "images")
输入形状: [1, 3, H, W]  # NCHW格式
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载 (源自onnx_base.py:115-122)
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
数据类型: float32
数值范围: [0.0, 1.0]  # 归一化RGB
颜色空间: RGB (从BGR转换)

# 输出 (基于 _postprocess 自适应处理)
输出形状: [B, N, 4+C] 或 [B, 4+C, N]  # 自动检测并转换
         # 例如: [1, 8400, 84] 或 [1, 84, 8400]
坐标格式: [x_center, y_center, width, height]  # 可能归一化或像素
类别输出: [conf1, ..., confC]  # 每类独立置信度
最终输出: [x1, y1, x2, y2, conf, class_id]  # 经NMS后
```

#### 前处理流程

```python
# 源自 onnx_yolo.py::_preprocess_static()
if use_ultralytics_preprocess:
    # Letterbox: 保持宽高比 + padding
    letterbox = UltralyticsLetterBox(new_shape=(640, 640))
    input_tensor, scale, original_shape, ratio_pad = letterbox(image)
else:
    # 简单resize: 直接拉伸
    input_tensor, scale, original_shape = preprocess_image(image, (640, 640))
```

#### 后处理流程

```python
# 源自 onnx_yolo.py::_postprocess()
# 1. 格式自适应: [B,C,N] -> [B,N,C]
# 2. 坐标归一化检测并转换为像素坐标
# 3. NMS后处理 (multi_label, has_objectness)
# 4. 坐标还原: letterbox需考虑ratio_pad, 否则简单缩放
```

---

### 2. RT-DETR (RTDETROnnx)

**实现类**: `infer_onnx/onnx_rtdetr.py::RTDETROnnx`
**原始框架**: Ultralytics RT-DETR
**特点**: 端到端检测,无需NMS,300个query

#### 初始化参数

```python
# 源自 onnx_rtdetr.py::__init__()
RTDETROnnx(
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
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载 (源自onnx_base.py:115-122)
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
    * 当前默认: 640×640
数值范围: [0.0, 1.0]
颜色空间: RGB

# 输出 (源自 onnx_rtdetr.py 注释)
输出形状: [batch, 300, num_features]
         # 300 = query数量
         # num_features = 4 bbox + C classes (例如19=4+15)
坐标格式: [x_center, y_center, width, height]  # 归一化 [0,1]
类别输出: [logit1, ..., logitC]  # 智能检测:logits/sigmoid/softmax
最终输出: [x1, y1, x2, y2, conf, class_id]  # 排序+过滤后
```

#### 前处理流程

```python
# 源自 onnx_rtdetr.py::_preprocess_static()
# 直接Resize (不保持宽高比)
resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = rgb_image.astype(np.float32) / 255.0
tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
scale = (640/original_width, 640/original_height)
```

#### 后处理流程

```python
# 源自 onnx_rtdetr.py::_postprocess()
# 1. 分离bbox和scores
# 2. 智能归一化 (_smart_normalize_scores): 自动检测并处理
# 3. bbox缩放: 归一化坐标 * 640 -> 像素坐标
# 4. 坐标转换: xywh -> xyxy
# 5. 提取最大类别和置信度
# 6. 排序 + 置信度过滤
```

---

### 3. RF-DETR (RFDETROnnx)

**实现类**: `infer_onnx/onnx_rfdetr.py::RFDETROnnx`
**原始框架**: RF-DETR (ResNet + FPN + DETR)
**特点**: 双输出,ImageNet标准化,TopK选择

#### 初始化参数

```python
# 源自 onnx_rfdetr.py::__init__()
RFDETROnnx(
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
输入形状: [batch, 3, 576, 576]  # 注意尺寸差异
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载 (源自onnx_base.py:115-122)
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
    * 当前默认: 576×576
数值范围: ImageNet标准化
          mean=[0.485, 0.456, 0.406]
          std=[0.229, 0.224, 0.225]
颜色空间: RGB

# 输出 (双输出)
输出1 (pred_boxes): [batch, num_queries, 4]  # bbox, 归一化
输出2 (pred_logits): [batch, num_queries, C]  # 类别logits
最终输出: [x1, y1, x2, y2, conf, class_id]  # TopK选择后
```

#### 前处理流程

```python
# 源自 onnx_rfdetr.py::_preprocess_static()
resized = cv2.resize(image, (576, 576), interpolation=cv2.INTER_LINEAR)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = rgb.astype(np.float32) / 255.0

# ImageNet标准化 (关键差异)
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
normalized = (normalized - imagenet_mean) / imagenet_std

tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
```

#### 后处理流程

```python
# 源自 onnx_rfdetr.py::_postprocess()
# 1. 应用sigmoid激活
# 2. TopK选择 (展平所有query×classes, 选top 300)
# 3. 坐标转换: cxcywh -> xyxy (clamp w,h >= 0)
# 4. 缩放到输入图像尺寸 (* 576)
# 5. 置信度过滤
```

---

## OCR与分类模型

### 4. OCR模型 (OCRONNX)

**实现类**: `infer_onnx/onnx_ocr.py::OCRONNX`
**支持模型**: rec_ppocr_v3, rec_ppocr_v5
**功能**: 车牌号码识别
**特点**: 支持单层/双层车牌,倾斜校正,分割拼接

#### 初始化参数

```python
# 源自 onnx_ocr.py::__init__()
OCRONNX(
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
输入形状: [batch, 3, 48, 168]
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载 (源自onnx_base.py:115-122)
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
    * 当前默认: 48×168
数值范围: [-1.0, 1.0]
颜色空间: BGR (不转换)

# 输出
输出形状: [batch, seq_len, num_classes]
         # seq_len: 序列长度 (动态, 通常21)
         # num_classes: 字符类别数 (例如85)
输出格式: CTC格式, 每个时间步的字符概率分布
最终输出: (text, avg_confidence, char_confidences)
```

#### 前处理流程

```python
# 源自 onnx_ocr.py::_preprocess()
# 1. 倾斜检测和校正 (Hough线变换 + 仿射变换)
# 2. 双层车牌处理 (if is_double_layer=True):
#    - CLAHE对比度增强
#    - 水平投影寻找分割线
#    - 分割上下层
#    - 拼接成单层 (上层缩小50%宽度)
# 3. 保持宽高比resize (max_w=168)
# 4. 归一化到 [-1, 1]
# 5. 右侧padding到168
```

#### 后处理流程

```python
# 源自 onnx_ocr.py::_postprocess()
# CTC解码:
# 1. argmax提取字符索引
# 2. 移除连续重复字符
# 3. 过滤忽略token (blank token 0)
# 4. 映射索引到字符
# 5. 后处理规则 (例如: '苏'->'京')
# 6. 计算平均置信度
```

---

### 5. 颜色层级分类 (ColorLayerONNX)

**实现类**: `infer_onnx/onnx_ocr.py::ColorLayerONNX`
**功能**: 车牌颜色和层级分类
**特点**: 双输出,同时预测颜色和层级

#### 初始化参数

```python
# 源自 onnx_ocr.py::__init__()
ColorLayerONNX(
    onnx_path: str,
    color_map: Dict[int, str],                  # 颜色索引映射 (必需)
    layer_map: Dict[int, str],                  # 层级索引映射 (必需)
    input_shape: Tuple[int, int] = (48, 168),
    conf_thres: float = 0.5,
    providers: Optional[List[str]] = None
)
```

#### 输入输出规格

```python
# 输入
输入形状: [batch, 3, 48, 168]
动态维度支持:
  - batch: 固定为1 (代码使用batch=1推理)
  - H, W: 智能加载 (源自onnx_base.py:115-122)
    * 模型固定尺寸 -> 从模型自动读取
    * 模型动态尺寸 -> 使用input_shape参数
    * 当前默认: 48×168
数值范围: [-1.0, 1.0]
颜色空间: BGR

# 输出 (双输出)
输出1 (color_logits): [batch, 5]  # 5个颜色类别
输出2 (layer_logits): [batch, 2]  # 2个层级类别
类别映射 (示例):
  color: {0:'blue', 1:'yellow', 2:'white', 3:'black', 4:'green'}
  layer: {0:'single', 1:'double'}
最终输出: (color_name, layer_name, average_confidence)
```

#### 前处理流程

```python
# 源自 onnx_ocr.py::_preprocess_static()
img = cv2.resize(img, (168, 48))
img = (img / 255.0 - 0.5) / 0.5  # 归一化到 [-1, 1]
img = img.transpose([2, 0, 1])   # HWC -> CHW
input_tensor = img[np.newaxis, :, :, :]
```

#### 后处理流程

```python
# 源自 onnx_ocr.py::__call__()
# 1. 获取color_logits和layer_logits
# 2. 应用softmax
# 3. argmax获取预测类别
# 4. 映射到名称
# 5. 置信度过滤
```

---

## 模型对比表

### 检测模型对比

| 模型 | 推理类 | 输入尺寸 | Batch动态 | 前处理 | 后处理 | 推荐场景 |
|------|--------|---------|----------|--------|--------|---------|
| YOLO | YoloOnnx | 640×640 | ✅ | Letterbox | NMS | 实时检测 |
| RT-DETR | RTDETROnnx | 640×640 | ✅ | Resize | 排序过滤 | 平衡场景 |
| RF-DETR | RFDETROnnx | 576×576 | ✅ | ImageNet | TopK | 高精度 |

### OCR与分类模型对比

| 模型 | 推理类 | 输入尺寸 | Batch动态 | 特殊处理 | 推荐场景 |
|------|--------|---------|----------|---------|---------|
| OCR | OCRONNX | 48×168 | ✅ | 倾斜校正+双层分割 | 车牌识别 |
| 颜色层级 | ColorLayerONNX | 48×168 | ✅ | Softmax | 属性分类 |

### 关键差异总结

| 特性 | YOLO | RT-DETR | RF-DETR | OCR | ColorLayer |
|------|------|---------|---------|-----|-----------|
| **预处理** | Letterbox | Resize | ImageNet | 倾斜+拼接 | Resize |
| **归一化** | [0,1] | [0,1] | ImageNet | [-1,1] | [-1,1] |
| **颜色空间** | RGB | RGB | RGB | BGR | BGR |
| **后处理** | NMS | 排序 | TopK | CTC | Softmax |
| **置信度阈值** | 0.5 | 0.001 | 0.001 | 0.5 | 0.5 |
| **NMS需求** | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 常见问题

### Q1: 为什么不同模型的conf_thres默认值不同？

**A**: 基于模型输出特性:
- **YOLO (0.5)**: 输出是类别置信度,范围[0,1]
- **RT-DETR/RF-DETR (0.001)**: 输出300个queries,需要低阈值保留候选,无NMS依靠排序过滤

### Q2: 输入尺寸是如何确定的？

**A**: BaseOnnx使用智能加载机制 (源自 `onnx_base.py:115-122`):

1. **从ONNX模型读取输入形状**:
   ```python
   input_shape_from_model = input_metadata[self.input_name].shape
   # 例如: [1, 3, 640, 640] 或 ['batch', 3, 'height', 'width']
   ```

2. **判断H, W是否固定**:
   - 如果 `input_shape_from_model[2]` 和 `[3]` 是**整数且>0** → **固定尺寸**,从模型读取
   - 如果是**字符串/符号** (如`'height'`, `'p2o.DynamicDimension.0'`) → **动态尺寸**,使用用户指定的`input_shape`

3. **batch维度**: 模型可能支持动态batch,但代码始终使用`batch=1`

4. **查看模型尺寸**:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("model.onnx")
   shape = session.get_inputs()[0].shape
   print(shape)  # [1, 3, 640, 640] 或 ['batch', 3, 640, 640]
   ```

### Q3: 如何验证模型输入输出格式？

**A**: 使用`onnxruntime`检查:
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

for inp in session.get_inputs():
    print(f"Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
for out in session.get_outputs():
    print(f"Output: {out.name}, Shape: {out.shape}, Type: {out.type}")
```

### Q5: 如何实现真正的批处理推理？

**A**:
- **模型层面**: RT-DETR、RF-DETR、OCR、ColorLayer支持动态batch
- **推理类层面**: 当前实现都使用`batch=1`
- **如需批处理**: 需要修改推理类的`__call__()`方法

### Q6: 坐标还原到原图为什么不准确？

**A**: 检查:
1. **预处理方式**: Letterbox (保持比例+padding) vs 直接Resize (拉伸)
2. **坐标格式**: 归一化[0,1] vs 像素坐标
3. **ratio_pad**: YOLO使用Letterbox时必须考虑padding

### Q7: batch=1 是什么意思？模型不是支持动态batch吗？

**A**: 你的理解完全正确！虽然某些模型支持动态batch维度，但**代码实现始终是一张一张图片喂进去**。

1. **输入始终是单张图片** (源自 `onnx_base.py:255,269-272`):
   ```python
   # __call__() 方法只接受单张图片
   def __call__(self, image: np.ndarray, ...):
       # image是单张图片，shape [H, W, 3]
       input_tensor, ... = self._prepare_inference(image)
       # input_tensor.shape = [1, 3, H, W]，batch=1
   ```

2. **特殊情况处理** (如果模型固定batch>1):
   ```python
   if expected_batch_size > 1 and input_tensor.shape[0] == 1:
       # 重复同一张图片来满足模型要求
       input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
       # 这不是真正的批处理，只是适配模型
   ```

3. **返回结果也只取第一个**:
   ```python
   if (expected_batch_size > 1 and len(detections) > 1):
       detections = [detections[0]]  # 只返回第一个batch结果
   ```

4. **为什么不支持真正的批处理？**
   - **简化接口**: 单张图片推理更直观
   - **实时应用**: 大多数应用场景是实时视频流，本身就是逐帧处理
   - **内存效率**: 批处理需要积累图片，增加延迟和内存占用
   - **灵活性**: 每张图片独立处理，可以有不同的预处理参数

5. **如果需要批处理**:
   需要修改 `__call__()` 方法，接受图片列表并批量预处理：
   ```python
   def __call__(self, images: List[np.ndarray], ...):
       # 批量预处理
       batch_tensor = np.stack([self._preprocess(img)[0] for img in images])
       # batch_tensor.shape = [N, 3, H, W]
       ...
   ```

---

## 附录

### 模型文件命名规范

```
检测模型:
- yolo11n.onnx           # YOLO11 Nano
- rtdetr-2024080100.onnx # RT-DETR, 日期戳
- rfdetr-20250811.onnx   # RF-DETR, 日期戳

OCR/分类模型:
- ocr.onnx              # OCR模型
- color_layer.onnx      # 颜色层级分类
```

### 相关文档

- [项目总览](../README.md)
- [模块文档](../CLAUDE.md)
- [推理引擎](../infer_onnx/CLAUDE.md)
- [测试文档](../tests/CLAUDE.md)

---

**文档维护**: 本文档基于推理类源码生成,与代码同步更新。

**最后更新**: 2025-10-11
**作者**: yyq19990828
**版本**: v2.1.0
