[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **infer_onnx**

# 推理引擎模块 (onnxtools.infer_onnx)

## 模块职责

核心ONNX推理引擎，提供多种目标检测模型架构的统一接口，包括车辆检测、车牌检测、OCR识别和颜色/层级分类功能。支持YOLO、RT-DETR、RF-DETR等主流检测架构，提供BaseORT基类和统一的推理接口。

## 入口和启动

- **主要工厂函数**: `onnxtools.create_detector()` (根模块)
- **基础抽象类**: `onnx_base.py::BaseORT`
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

### 0. Result类 - 检测结果包装器 (NEW)

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

#### 错误处理
```python
# 可视化方法需要orig_img
result_no_img = Result(boxes=boxes, orig_shape=(640, 640))
try:
    result_no_img.plot()
except ValueError as e:
    print(f"Error: {e}")  # "Cannot plot detections: orig_img is None"

# filter()参数验证
try:
    result.filter(conf_threshold=1.5)  # 超出范围
except ValueError as e:
    print(f"Error: {e}")  # "conf_threshold must be in [0.0, 1.0] range"
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

### 2. OCR和颜色分类器
```python
from onnxtools import ColorLayerORT, OcrORT
import yaml

# 加载配置
with open('configs/plate.yaml') as f:
    config = yaml.safe_load(f)

# 车牌颜色和层级分类
color_classifier = ColorLayerORT(
    onnx_path='models/color_layer.onnx',
    color_map=config['color_map'],
    layer_map=config['layer_map'],
    input_shape=(48, 168),
    conf_thres=0.5
)
color, layer, conf = color_classifier(plate_image)

# 车牌OCR识别
ocr_model = OcrORT(
    onnx_path='models/ocr.onnx',
    character=config['plate_dict']['character'],
    input_shape=(48, 168),
    conf_thres=0.7
)
result = ocr_model(plate_image, is_double_layer=True)
if result:
    text, confidence, char_confs = result
```

### 3. 数据集评估器
```python
from onnxtools import DatasetEvaluator, OCRDatasetEvaluator

# COCO数据集评估
evaluator = DatasetEvaluator(dataset_path, annotations_path)
metrics = evaluator.evaluate(detector)

# OCR数据集评估
ocr_evaluator = OCRDatasetEvaluator(ocr_model, character_dict)
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
| OCR | [1,3,48,320] | [1,T,C] | T个时间步，C个字符类别 |
| Color/Layer | [1,3,224,224] | [1,K] | K个类别的logits |

## 数据模型

### Result类 - 统一检测结果对象 (NEW)
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
result.to_dict()                            # 转换为字典(已废弃)

# 所有BaseORT子类现在返回Result对象而不是字典
detector = create_detector('yolo', 'model.onnx')
result = detector(image)  # 返回Result实例
assert isinstance(result, Result)
```

### 检测结果字典结构 (旧格式,已废弃)
```python
# 注意: BaseORT现在返回Result对象,此格式仅用于向后兼容
detection_result = {
    'boxes': np.ndarray,        # [N, 4] xyxy格式边界框
    'scores': np.ndarray,       # [N] 置信度分数
    'class_ids': np.ndarray,    # [N] 类别ID
    'mask': np.ndarray          # [N] NMS后的有效掩码（可选）
}
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

### 颜色分类结果
```python
classification_result = (
    color: str,                 # 'blue', 'yellow', 'white', 'black', 'green'
    layer: str,                 # 'single', 'double'
    confidence: float           # 分类置信度
)
```

### OCR评估结果（SampleEvaluation）
```python
from onnxtools import SampleEvaluation

sample = SampleEvaluation(
    image_path='val_001.jpg',
    label='京A12345',
    prediction='京A12345',
    exact_match=True,
    normalized_edit_distance=0.0,
    edit_distance_similarity=1.0,
    confidence=0.95,
    processing_time_ms=25.3
)
```

## 测试和质量

### 单元测试覆盖
- [x] `test_ocr_onnx_refactored.py` - OCRONNX重构后的27个单元测试
- [x] `test_ocr_metrics.py` - OCR指标计算23个单元测试
- [x] `test_load_label_file.py` - 标签文件加载12个单元测试
- [ ] BaseORT基类功能测试
- [ ] 多模型架构兼容性测试

### 集成测试覆盖
- [x] `test_pipeline_integration.py` - 完整推理管道测试
- [x] `test_ocr_integration.py` - OCR识别流程测试
- [x] `test_ocr_evaluation_integration.py` - OCR评估集成测试 (8个用例)
- [x] 115/122 集成测试通过

### 合约测试覆盖
- [x] `test_ocr_onnx_refactored_contract.py` - OCRONNX API合约
- [x] `test_ocr_evaluator_contract.py` - OCR评估器合约 (11个用例)
- [x] 基础评估流程、编辑距离、置信度过滤、JSON导出验证

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

### Q: 支持哪些ONNX模型版本？
A: 当前支持ONNX opset版本11-17，推荐使用opset 17以获得最佳兼容性。使用 `onnx.version_converter` 可以转换旧版本模型。

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
- `onnx_base.py` - BaseORT抽象基类，定义统一接口
- `onnx_yolo.py` - YoloORT，YOLO系列模型推理
- `onnx_rtdetr.py` - RtdetrORT，RT-DETR模型推理
- `onnx_rfdetr.py` - RfdetrORT，RF-DETR模型推理
- `infer_utils.py` - 推理辅助工具函数

### 专用功能模块
- `onnx_ocr.py` - OcrORT和ColorLayerORT，OCR识别和颜色分类
- `eval_coco.py` - DatasetEvaluator，COCO数据集评估
- `eval_ocr.py` - OCRDatasetEvaluator，OCR数据集评估
- `engine_dataloader.py` - TensorRT引擎数据加载器

### 配置和接口
- `__init__.py` - 模块导入和API定义，导出所有公共类

## 架构设计

### 类继承关系
```
BaseORT (抽象基类)
├── YoloORT (YOLO系列)
├── RtdetrORT (RT-DETR)
├── RfdetrORT (RF-DETR)
├── ColorLayerORT (颜色/层级分类)
└── OcrORT (OCR识别)
```

### 核心抽象方法
所有子类必须实现：
- `_preprocess_static(img, **kwargs)` - 静态预处理方法
- `_postprocess(outputs, **kwargs)` - 后处理输出

### 统一调用接口
```python
# 所有推理类使用 __call__ 方法
result = model(image, **kwargs)

# 内部执行流程：
# 1. _prepare_inference() - 准备推理（预处理）
# 2. _execute_inference() - 执行推理（会话运行）
# 3. _finalize_inference() - 完成推理（后处理）
```

## 变更日志 (Changelog)

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
*最后更新: 2025-11-05 15:02:47*
