[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **infer_onnx**

# 推理引擎模块 (onnxtools.infer_onnx)

## 模块职责

核心ONNX推理引擎，提供多种目标检测模型架构的统一接口，包括车辆检测、车牌检测、OCR识别和颜色/层级/属性分类功能。支持YOLO、RT-DETR、RF-DETR等主流检测架构，提供BaseORT和BaseClsORT双基类架构和统一的推理接口。

## 入口和启动

- **主要工厂函数**: `onnxtools.create_detector()` (根模块)
- **检测基类**: `onnx_base.py::BaseORT`
- **分类基类**: `onnx_cls.py::BaseClsORT`
- **模块导入**: `__init__.py` 提供统一的API接口

### 快速开始
```python
from onnxtools import create_detector

# 创建检测器实例
detector = create_detector(
    model_type='rtdetr',  # 'yolo', 'rtdetr', 'rfdetr'
    onnx_path='models/rtdetr-2024080100.onnx',
    conf_thres=0.5,
    iou_thres=0.5
)

# 执行推理 - 返回Result对象
result = detector(image)  # 使用 __call__ 接口

# 访问检测结果
for i in range(len(result)):
    box = result.boxes[i]
    score = result.scores[i]
    class_id = result.class_ids[i]
    print(f"Detection {i}: {box} with confidence {score}")

# 可视化和保存
result.plot(annotator_preset='debug')  # 标注图像
result.show()  # 显示
result.save('output.jpg')  # 保存

# 过滤和统计
high_conf = result.filter(conf_threshold=0.8)
stats = result.summary()
print(f"Total detections: {stats['total_detections']}")
```

## 外部接口

### 0. Result类 - 检测结果包装器

Result类是BaseORT子类返回的统一检测结果对象,提供面向对象的数据访问、可视化、过滤和统计功能。

#### 创建Result对象
```python
from onnxtools.infer_onnx import Result
import numpy as np

# 从检测结果创建
result = Result(
    boxes=np.array([[10, 20, 100, 150]], dtype=np.float32),  # [N, 4] xyxy格式
    scores=np.array([0.95], dtype=np.float32),               # [N] 置信度
    class_ids=np.array([0], dtype=np.int32),                 # [N] 类别ID
    orig_shape=(640, 640),                                    # (H, W) 原图尺寸
    names={0: 'vehicle', 1: 'plate'},                        # 类别名称映射
    path='image.jpg',                                         # 图像路径(可选)
    orig_img=image_array                                      # 原图(可选,用于可视化)
)
```

#### 属性访问
```python
# 只读属性
result.boxes       # np.ndarray [N, 4] - 边界框 (xyxy格式)
result.scores      # np.ndarray [N] - 置信度分数
result.class_ids   # np.ndarray [N] - 类别ID
result.orig_shape  # Tuple[int, int] - 原图尺寸 (H, W)
result.names       # Dict[int, str] - 类别名称映射
result.path        # Optional[str] - 图像路径
result.orig_img    # Optional[np.ndarray] - 原始图像

# 基础操作
len(result)        # 检测数量
str(result)        # 字符串表示
```

#### 索引和切片 (User Story 1)
```python
# 整数索引 - 返回单个检测的Result对象
first = result[0]
last = result[-1]
assert isinstance(first, Result)
assert len(first) == 1

# 切片 - 返回子集Result对象
subset = result[1:3]
assert isinstance(subset, Result)
assert len(subset) == 2

# 迭代
for detection in result:
    print(f"Box: {detection.boxes[0]}, Score: {detection.scores[0]}")
```

#### 可视化功能 (User Story 2)
```python
# plot() - 生成标注图像
annotated = result.plot(annotator_preset='standard')  # 返回np.ndarray
# 可用预设: 'standard', 'debug', 'lightweight', 'privacy'

# show() - 显示标注图像
result.show(window_name='Detections', annotator_preset='debug')

# save() - 保存标注图像
result.save('output.jpg', annotator_preset='standard')

# to_supervision() - 转换为supervision.Detections
sv_detections = result.to_supervision()  # 用于高级可视化
```

#### 过滤和统计 (User Story 3)
```python
# filter() - 按条件过滤检测
high_conf = result.filter(conf_threshold=0.8)        # 置信度过滤
vehicles = result.filter(classes=[0])                 # 类别过滤
high_vehicles = result.filter(conf_threshold=0.7, classes=[0])  # 组合过滤

# summary() - 获取统计信息
stats = result.summary()
# 返回: {
#   'total_detections': 10,
#   'class_counts': {'vehicle': 8, 'plate': 2},
#   'avg_confidence': 0.85,
#   'min_confidence': 0.65,
#   'max_confidence': 0.98
# }
```

#### 完整工作流示例
```python
from onnxtools import create_detector
import cv2

# 1. 创建检测器
detector = create_detector('rtdetr', 'models/rtdetr.onnx', conf_thres=0.5)

# 2. 执行推理
image = cv2.imread('test.jpg')
result = detector(image)

# 3. 查看统计
print(result)  # "Result(10 detections, 2 classes)"
stats = result.summary()
print(f"Found {stats['total_detections']} objects")
print(f"Class distribution: {stats['class_counts']}")

# 4. 过滤高置信度检测
high_conf = result.filter(conf_threshold=0.8)
print(f"High confidence detections: {len(high_conf)}")

# 5. 可视化和保存
high_conf.save('high_confidence.jpg', annotator_preset='debug')

# 6. 处理单个检测
for i, detection in enumerate(high_conf):
    box = detection.boxes[0]
    score = detection.scores[0]
    class_name = detection.names[detection.class_ids[0]]
    print(f"{i}: {class_name} @ {score:.2f} - Box: {box}")
```

### 1. 检测器工厂函数
```python
# 位于根模块 onnxtools/__init__.py
def create_detector(model_type: str, onnx_path: str, **kwargs) -> BaseORT:
    """
    根据模型类型创建相应的检测器

    Args:
        model_type: 'yolo', 'rtdetr', 'rfdetr'
        onnx_path: ONNX模型文件路径
        **kwargs: conf_thres, iou_thres, providers等

    Returns:
        BaseORT子类实例
    """
```

### 2. 分类器类 (NEW - 继承BaseClsORT)

**架构升级**: 从2025-11-25开始,所有分类模型(车牌颜色/层级、车辆属性)统一继承`BaseClsORT`基类,返回`ClsResult`对象。

```python
from onnxtools import ColorLayerORT, VehicleAttributeORT, ClsResult

# 车牌颜色和层级分类 - 双分支分类器
color_classifier = ColorLayerORT(
    onnx_path='models/color_layer.onnx',
    # 可选: 使用默认映射或外部配置
    color_map={0: 'black', 1: 'blue', 2: 'green', 3: 'white', 4: 'yellow'},
    layer_map={0: 'single', 1: 'double'},
    input_shape=(48, 168),
    conf_thres=0.5
)

# 返回ClsResult对象,支持属性访问和元组解包
result = color_classifier(plate_image)
print(result.labels[0])        # 颜色标签: 'blue'
print(result.labels[1])        # 层级标签: 'single'
print(result.avg_confidence)   # 平均置信度: 0.92

# 向后兼容: 元组解包
color, layer, conf = color_classifier(plate_image)

# 车辆属性分类 - 多标签分类器 (车型 + 颜色)
vehicle_classifier = VehicleAttributeORT(
    onnx_path='models/vehicle_attribute.onnx',
    input_shape=(224, 224),
    conf_thres=0.5
)

# 返回ClsResult: 车型 + 车辆颜色
result = vehicle_classifier(vehicle_image)
print(f"Vehicle: {result.labels[0]}")  # 'car'
print(f"Color: {result.labels[1]}")    # 'white'
print(f"Type Confidence: {result.confidences[0]}")
print(f"Color Confidence: {result.confidences[1]}")

# 元组解包
vehicle_type, color, avg_conf = vehicle_classifier(vehicle_image)
```

#### ClsResult API
```python
from onnxtools import ClsResult

# ClsResult属性
result.labels           # List[str] - 分类标签列表
result.confidences      # List[float] - 每个分支的置信度
result.avg_confidence   # float - 平均置信度
result.logits           # Optional[List[np.ndarray]] - 原始logits

# ClsResult方法
len(result)            # 分支数量
result[0]              # (label, confidence) 元组
for label, conf in result:  # 迭代所有分支
    print(f"{label}: {conf:.3f}")

# 元组解包支持
# 单分支: label, conf = result
# 双分支: label1, label2, avg_conf = result
# 多分支: labels, confs, avg_conf = result
```

### 3. OCR识别器（独立推理类）

**设计说明**: `OcrORT` 保持独立推理类设计,不继承BaseORT/BaseClsORT,因为OCR是序列识别任务,返回可变长度字符序列。

```python
from onnxtools import OcrORT

# 车牌OCR识别 - 独立类,返回Optional元组
ocr_model = OcrORT(
    onnx_path='models/ocr.onnx',
    character=['京', '沪', 'A', 'B', '0', '1', ...],  # 字符字典
    input_shape=(48, 168),
    conf_thres=0.7
)

# 返回Optional[(text: str, confidence: float, char_confs: List[float])]
result = ocr_model(plate_image, is_double_layer=True)
if result:
    text, confidence, char_confs = result
    print(f"Plate: {text}, Conf: {confidence:.3f}")
    print(f"Char confidences: {char_confs}")
```

### 4. 数据集评估器
```python
from onnxtools import DetDatasetEvaluator, OCRDatasetEvaluator

# COCO数据集评估
evaluator = DetDatasetEvaluator(detector)
metrics = evaluator.evaluate_dataset(dataset_path)

# OCR数据集评估
ocr_evaluator = OCRDatasetEvaluator(ocr_model)
results = ocr_evaluator.evaluate_dataset(label_file, dataset_base)
```

## 关键依赖和配置

### 运行时依赖
- **onnxruntime-gpu** (1.22.0): ONNX模型推理引擎
- **numpy** (>=2.2.6): 数值计算和张量操作
- **opencv-contrib-python** (>=4.12.0): 图像预处理和后处理
- **pyyaml** (>=6.0.2): 配置文件解析
- **python-levenshtein** (>=0.25.0): OCR编辑距离计算

### 配置文件
- `configs/det_config.yaml`: 检测类别名称和可视化颜色
- `configs/plate.yaml`: OCR字典和颜色/层级映射

### 模型文件要求
| 模型类型 | 输入形状 | 输出格式 | 说明 |
|---------|---------|---------|------|
| YOLO | [1,3,640,640] | [1,N,85] | N个检测，85维(x,y,w,h,conf,classes) |
| RT-DETR | [1,3,640,640] | [1,N,6] | N个检测，6维(x1,y1,x2,y2,score,cls) |
| RF-DETR | [1,3,640,640] | [1,N,6] | 同RT-DETR格式 |
| OCR | [1,3,48,168] | [1,T,C] | T个时间步，C个字符类别 |
| ColorLayer | [1,3,48,168] | [[1,5],[1,2]] | 双输出: 颜色(5类)+层级(2类) |
| VehicleAttribute | [1,3,224,224] | [1,24] | 单输出: 车型(13)+颜色(11) |

## 数据模型

### Result类 - 统一检测结果对象
```python
from onnxtools.infer_onnx import Result

# Result对象属性
result = Result(
    boxes=np.ndarray,           # [N, 4] xyxy格式边界框 (必需)
    scores=np.ndarray,          # [N] 置信度分数 (可选,默认全1)
    class_ids=np.ndarray,       # [N] 类别ID (可选,默认全0)
    orig_shape=(H, W),          # 原图尺寸 (必需)
    names={int: str},           # 类别名称映射 (可选)
    path=str,                   # 图像路径 (可选)
    orig_img=np.ndarray         # 原始图像 (可选,可视化需要)
)

# Result对象方法
result.__len__()                            # 检测数量
result.__getitem__(index)                   # 索引/切片访问
result.__iter__()                           # 迭代支持
result.plot(annotator_preset='standard')    # 生成标注图像
result.show(window_name='Result')           # 显示图像
result.save(output_path)                    # 保存图像
result.filter(conf_threshold, classes)      # 过滤检测
result.summary()                            # 统计信息
result.to_supervision()                     # 转换为sv.Detections

# 所有BaseORT子类现在返回Result对象
detector = create_detector('yolo', 'model.onnx')
result = detector(image)  # 返回Result实例
assert isinstance(result, Result)
```

### ClsResult类 - 统一分类结果对象 (NEW)
```python
from onnxtools.infer_onnx import ClsResult

# ClsResult对象属性
result = ClsResult(
    labels=['blue', 'single'],          # List[str] - 分类标签
    confidences=[0.95, 0.88],           # List[float] - 置信度
    avg_confidence=0.915,               # float - 平均置信度
    logits=[logits1, logits2]           # Optional - 原始输出
)

# 属性访问
result.labels[0]        # 'blue'
result.confidences[0]   # 0.95
result.avg_confidence   # 0.915
len(result)             # 2 (分支数)

# 元组解包(向后兼容)
color, layer, conf = result  # 双分支
label, conf = result         # 单分支
```

### OCR结果结构
```python
ocr_result = (
    text: str,                  # 识别的文本内容
    confidence: float,          # 平均置信度
    char_scores: List[float]    # 每个字符的置信度
)
# 或 None（识别失败时）
```

## 测试和质量

### 单元测试覆盖
- [x] `test_ocr_onnx_refactored.py` - OcrORT重构后的27个单元测试
- [x] `test_ocr_metrics.py` - OCR指标计算23个单元测试
- [x] `test_load_label_file.py` - 标签文件加载12个单元测试
- [ ] BaseClsORT基类功能测试 (待补充)
- [ ] ClsResult对象测试 (待补充)

### 集成测试覆盖
- [x] `test_pipeline_integration.py` - 完整推理管道测试
- [x] `test_ocr_integration.py` - OCR识别流程测试
- [x] `test_ocr_evaluation_integration.py` - OCR评估集成测试 (8个用例)
- [ ] 分类模型集成测试 (待补充)

### 合约测试覆盖
- [x] `test_ocr_onnx_refactored_contract.py` - OcrORT API合约
- [x] `test_ocr_evaluator_contract.py` - OCR评估器合约 (11个用例)
- [ ] BaseClsORT/ClsResult合约测试 (待补充)

### 性能基准
- 目标: 推理延迟 < 50ms (640x640图像)
- 目标: GPU内存使用 < 2GB (batch_size=1)
- 实际: OCR评估 <1秒处理5张图像

## 常见问题 (FAQ)

### Q: 如何添加新的模型架构？
A: 1) 继承 `BaseORT` 基类; 2) 实现 `_preprocess_static()`, `_postprocess()` 抽象方法; 3) 在根模块 `create_detector()` 中注册新模型类型

### Q: 模型推理速度慢怎么优化？
A: 1) 使用TensorRT引擎替代ONNX (`tools/build_engine.py`); 2) 调整输入分辨率; 3) 使用FP16精度; 4) 确保GPU推理 (`providers=['CUDAExecutionProvider']`)

### Q: OCR识别准确率低怎么改善？
A: 1) 检查车牌图像预处理质量（`_process_plate_image_static()`）; 2) 调整OCR模型置信度阈值; 3) 使用更大的OCR模型; 4) 增加训练数据覆盖

### Q: 为什么分类模型现在返回ClsResult而不是元组？
A: 为了提供统一的API体验和更好的扩展性。ClsResult支持:
- 属性访问: `result.labels[0]`
- 元组解包(向后兼容): `color, layer, conf = result`
- 迭代和索引: `for label, conf in result:`
- 支持任意数量的分类分支

### Q: ColorLayerORT移到哪里了？
A: 从`onnx_ocr.py`迁移到`onnx_cls.py`,现在继承`BaseClsORT`。API保持兼容:
```python
from onnxtools import ColorLayerORT  # 仍然有效
color, layer, conf = classifier(image)  # 元组解包仍然支持
```

### Q: 如何进行OCR数据集评估？
A: 使用命令行工具：
```bash
python tools/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5
```
支持表格和JSON两种输出格式，以及详细的字符级错误分析。

## 相关文件列表

### 核心推理文件
- `onnx_base.py` - BaseORT抽象基类，定义检测器统一接口
- `onnx_yolo.py` - YoloORT，YOLO系列模型推理
- `onnx_rtdetr.py` - RtdetrORT，RT-DETR模型推理
- `onnx_rfdetr.py` - RfdetrORT，RF-DETR模型推理
- `infer_utils.py` - 推理辅助工具函数

### 分类和OCR模块 (NEW架构)
- `onnx_cls.py` - **BaseClsORT基类, ClsResult, ColorLayerORT, VehicleAttributeORT**
- `onnx_ocr.py` - OcrORT，OCR序列识别
- `result.py` - Result类，检测结果包装器

### 其他模块
- `engine_dataloader.py` - TensorRT引擎数据加载器

### 配置和接口
- `__init__.py` - 模块导入和API定义，导出所有公共类

## 架构设计

### 类继承关系 (2025-11-25更新)
```
检测器架构:
BaseORT (抽象基类 - 目标检测)
├── YoloORT (YOLO系列)
├── RtdetrORT (RT-DETR)
└── RfdetrORT (RF-DETR)
    → 返回 Result 对象

分类器架构 (NEW):
BaseClsORT (抽象基类 - 分类任务)
├── ColorLayerORT (车牌颜色/层级 - 双分支)
└── VehicleAttributeORT (车辆类型/颜色 - 多标签)
    → 返回 ClsResult 对象

独立推理类 (序列识别):
OcrORT (OCR识别 - 序列任务)
    → 返回 Optional[Tuple]
```

**架构决策说明**:

**为什么分类模型现在继承BaseClsORT?**

1. **统一抽象模式**:
   - 检测器(BaseORT) → Result对象
   - 分类器(BaseClsORT) → ClsResult对象
   - OCR(独立) → Optional[Tuple]

2. **代码复用和维护性**:
   - Template Method Pattern在BaseClsORT中实现
   - 所有分类模型共享预处理/推理/后处理流程
   - 减少重复代码,提高可维护性

3. **扩展性**:
   - ClsResult支持任意数量的分类分支
   - 向后兼容元组解包: `color, layer, conf = result`
   - 支持属性访问和迭代: `result.labels[0]`, `for label, conf in result`

4. **一致性**:
   - BaseORT和BaseClsORT使用相同的设计模式
   - 新分类模型只需实现`preprocess()`和`postprocess()`

**为什么OcrORT保持独立?**

1. **任务本质不同**: OCR是序列识别,不是分类
2. **返回类型特殊**: 可变长度字符序列 + 字符级置信度
3. **预处理复杂**: 需要双层车牌处理、倾斜校正等特殊逻辑

### 核心抽象方法
**BaseORT子类**必须实现：
- `_preprocess_static(img, **kwargs)` - 静态预处理方法
- `_postprocess(outputs, **kwargs)` - 后处理输出

**BaseClsORT子类**必须实现：
- `preprocess(img, input_shape, **kwargs)` - 静态预处理方法
- `postprocess(outputs, conf_thres, **kwargs)` - 后处理输出

### 统一调用接口
```python
# 检测器类 - 返回Result对象
detector = create_detector('yolo', 'model.onnx')
result = detector(image)  # Result实例
boxes = result.boxes
scores = result.scores

# 分类器类 - 返回ClsResult对象 (NEW)
classifier = ColorLayerORT('color.onnx')
result = classifier(plate_image)  # ClsResult实例
color = result.labels[0]
layer = result.labels[1]
# 或元组解包(向后兼容)
color, layer, conf = classifier(plate_image)

# OCR类 - 返回Optional[Tuple]
ocr = OcrORT('ocr.onnx', character)
ocr_result = ocr(plate_image)
if ocr_result:
    text, conf, char_confs = ocr_result  # 元组解包
```

## 变更日志 (Changelog)

**2025-11-25** - 分类架构重大升级
- ✅ **新增**: `onnx_cls.py` - BaseClsORT抽象基类和ClsResult结果类
- ✅ **新增**: `VehicleAttributeORT` - 车辆属性分类器(车型+颜色多标签)
- ✅ **迁移**: `ColorLayerORT` 从 `onnx_ocr.py` 迁移到 `onnx_cls.py`
- ✅ **重构**: `ColorLayerORT` 现在继承 `BaseClsORT`,返回 `ClsResult`
- ✅ **保持**: `OcrORT` 保留为独立推理类,API无变化
- ✅ **更新**: `__init__.py` 导出新的分类相关类
- ✅ **更新**: 文档完整记录新架构和迁移说明

**2025-11-05 (阶段1.3)** - OCR/分类类架构独立化和文档更新
- ✅ ColorLayerORT和OcrORT不再继承BaseORT
- ✅ 添加"为什么OCR和分类类不继承BaseORT?"架构决策说明
- ✅ 更新类继承关系图,区分检测器类和独立推理类
- ✅ 完善OCR/分类器API文档,强调元组返回类型
- ✅ 确认导出无变化(仍在__all__中)

**2025-11-05** - 初始化完整模块文档，建立清晰的面包屑导航
- 更新面包屑路径: [根目录] > [onnxtools] > [infer_onnx]
- 完善API文档和使用示例
- 添加数据模型详细定义
- 补充测试覆盖统计和FAQ

**2025-10-11** - Bug修复和配置优化
- 修复OCR评估器JSON数组格式支持
- TensorRT改为可选依赖 `[trt]`
- 新增12个单元测试用例

**2025-10-10** - OCR指标评估功能完成
- OCRDatasetEvaluator类提供完整评估
- 三大指标：完全准确率、归一化编辑距离、编辑距离相似度
- 字符级分析（SampleEvaluation）
- 表格对齐终端输出 + JSON导出

**2025-10-09** - ColorLayerORT和OcrORT重构完成
- 继承BaseORT，统一`__call__()`接口
- 所有预处理/后处理函数迁移为静态方法
- 删除utils模块OCR相关依赖
- 27个单元测试通过，115/122集成测试通过

**2025-09-30** - BaseORT抽象方法强制实现
- `_postprocess()`和`_preprocess_static()`添加@abstractmethod
- `__call__`方法重构，代码减少83.3%
- 所有5个子类验证通过

**2025-09-15** - 初始化推理引擎模块文档

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/onnxtools/infer_onnx/`*
*最后更新: 2025-11-25 14:42:10*
