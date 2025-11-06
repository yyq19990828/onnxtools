[根目录](../CLAUDE.md) > **onnxtools**

# ONNX工具包核心模块 (onnxtools)

## 模块职责

onnxtools是项目的Python包核心，提供统一的ONNX模型推理接口、多架构检测器支持、OCR识别、数据集评估和可视化工具。作为根模块，它整合了推理引擎(infer_onnx)和工具函数(utils)，对外暴露简洁的API。

## 入口和启动

- **根模块入口**: `onnxtools/__init__.py`
- **工厂函数**: `create_detector()` - 统一的检测器创建接口
- **主程序入口**: `main.py` (项目根目录)

### 快速开始
```python
# 基础检测
from onnxtools import create_detector, setup_logger

setup_logger('INFO')
detector = create_detector('rtdetr', 'models/rtdetr.onnx', conf_thres=0.5)
results = detector(image)

# OCR识别
from onnxtools import OcrORT, ColorLayerORT
import yaml

with open('configs/plate.yaml') as f:
    config = yaml.safe_load(f)

ocr_model = OcrORT('models/ocr.onnx', character=config['plate_dict']['character'])
text, conf, char_scores = ocr_model(plate_image)

# 数据集评估
from onnxtools import DatasetEvaluator, OCRDatasetEvaluator

evaluator = DatasetEvaluator(dataset_path, annotations_path)
metrics = evaluator.evaluate(detector)
```

## 外部接口

### 1. Result类 - 检测结果对象 (NEW)
```python
from onnxtools import Result  # 或 from onnxtools.infer_onnx import Result

# 所有检测器现在返回Result对象
detector = create_detector('rtdetr', 'models/rtdetr.onnx')
result = detector(image)  # 返回Result实例

# Result对象提供丰富的API
print(result)  # "Result(10 detections, 2 classes)"
print(f"Found {len(result)} objects")

# 访问检测数据
result.boxes       # np.ndarray [N, 4]
result.scores      # np.ndarray [N]
result.class_ids   # np.ndarray [N]

# 索引和切片
first = result[0]         # 单个检测
subset = result[1:3]      # 多个检测
for det in result:        # 迭代
    print(det.boxes[0])

# 可视化
result.plot(annotator_preset='debug')
result.show()
result.save('output.jpg')

# 过滤和统计
high_conf = result.filter(conf_threshold=0.8)
vehicles = result.filter(classes=[0])
stats = result.summary()

# 参考 onnxtools/infer_onnx/CLAUDE.md 查看完整API文档
```

### 2. 核心推理接口

**架构说明**: onnxtools提供两类推理类:
- **检测器类** (继承BaseORT): 目标检测任务,返回`Result`对象
- **分类器/OCR类** (独立): 分类/序列识别任务,返回元组

```python
from onnxtools import (
    # 检测器类 (继承BaseORT) - 返回Result
    BaseORT,           # 抽象基类
    YoloORT,           # YOLO系列检测器
    RtdetrORT,         # RT-DETR检测器
    RfdetrORT,         # RF-DETR检测器

    # 独立分类器/OCR类 - 返回元组
    ColorLayerORT,     # 颜色/层级分类器
    OcrORT,            # OCR识别器

    # 其他
    Result,            # 检测结果类 (NEW)
    create_detector    # 工厂函数
)

# 检测器使用（推荐）- 返回Result对象
detector = create_detector('yolo', 'models/yolo11n.onnx')
result = detector(image)  # Result实例
boxes = result.boxes
scores = result.scores

# 分类器/OCR使用 - 返回元组
classifier = ColorLayerORT('color.onnx', color_map, layer_map)
color, layer, conf = classifier(plate_image)  # 元组解包

# 或直接实例化检测器
detector = RtdetrORT('models/rtdetr.onnx', conf_thres=0.5, iou_thres=0.7)
result = detector(image)  # Result实例
```

### 2. 评估工具
```python
from onnxtools import (
    DatasetEvaluator,      # COCO数据集评估
    OCRDatasetEvaluator,   # OCR数据集评估
    SampleEvaluation       # OCR样本评估数据类
)
```

### 3. 工具函数
```python
from onnxtools import setup_logger

# 其他工具函数通过子模块访问
from onnxtools.utils import (
    preprocess_image,
    draw_detections_supervision,
    convert_to_supervision_detections
)
```

## 模块结构

```
onnxtools/
├── __init__.py                 # 根模块，导出公共API
├── pipeline.py                 # 完整推理管道
├── infer_onnx/                 # 推理引擎子模块
│   ├── __init__.py
│   ├── onnx_base.py            # BaseORT抽象基类
│   ├── onnx_yolo.py            # YOLO推理
│   ├── onnx_rtdetr.py          # RT-DETR推理
│   ├── onnx_rfdetr.py          # RF-DETR推理
│   ├── onnx_ocr.py             # OCR和颜色分类
│   ├── eval_coco.py            # COCO评估
│   ├── eval_ocr.py             # OCR评估
│   ├── infer_utils.py          # 推理工具
│   └── engine_dataloader.py   # TensorRT数据加载
└── utils/                      # 工具函数子模块
    ├── __init__.py
    ├── image_processing.py     # 图像预处理
    ├── drawing.py              # 可视化绘制
    ├── supervision_converter.py  # Supervision转换
    ├── supervision_config.py   # Supervision配置
    ├── supervision_labels.py   # 标签创建
    ├── annotator_factory.py    # Annotator工厂
    ├── visualization_preset.py # 可视化预设
    ├── ocr_metrics.py          # OCR指标计算
    ├── detection_metrics.py    # 检测指标
    ├── nms.py                  # NMS算法
    ├── font_utils.py           # 字体工具
    ├── output_transforms.py    # 输出转换
    └── logging_config.py       # 日志配置
```

## 关键依赖和配置

### 核心依赖
```toml
[project.dependencies]
onnxruntime-gpu = "1.22.0"
opencv-contrib-python = ">=4.12.0.88"
numpy = ">=2.2.6"
pyyaml = ">=6.0.2"
supervision = "0.26.1"
python-levenshtein = ">=0.25.0"
colorlog = ">=6.9.0"
```

### 可选依赖
```toml
[project.optional-dependencies]
trt = [
    "tensorrt==8.6.1.post1",
    "tensorrt-bindings==8.6.1",
    "tensorrt-libs==8.6.1"
]
```

### 配置文件
- `configs/det_config.yaml`: 检测类别和颜色
- `configs/plate.yaml`: OCR字典和映射
- `configs/visualization_presets.yaml`: Supervision预设

## 数据模型

### 统一推理输出
```python
# 检测器输出（BaseORT子类）
detection_result = {
    'boxes': np.ndarray,        # [N, 4] xyxy格式
    'scores': np.ndarray,       # [N] 置信度
    'class_ids': np.ndarray,    # [N] 类别ID
    'mask': np.ndarray          # [N] 有效性掩码（可选）
}

# OCR输出
ocr_result = (
    text: str,                  # 识别文本
    confidence: float,          # 平均置信度
    char_scores: List[float]    # 字符置信度列表
) or None

# 颜色分类输出
color_result = (
    color: str,                 # 颜色类别
    layer: str,                 # 层级类别
    confidence: float           # 置信度
)
```

## 测试和质量

### 测试体系
- **单元测试**: 62+ 测试用例
  - `test_ocr_onnx_refactored.py` (27个)
  - `test_ocr_metrics.py` (23个)
  - `test_load_label_file.py` (12个)

- **集成测试**: 30+ 测试套件
  - 管道集成、OCR集成、可视化集成
  - Annotator集成（13种类型）
  - 预设场景测试

- **合约测试**: 15+ 测试套件
  - API合约验证
  - 数据转换合约
  - 性能基准合约

- **性能测试**: 基准测试
  - Annotator性能：75μs ~ 1.5ms
  - 推理延迟目标：< 50ms (640x640)

### 测试覆盖率
- 单元测试通过率: 100% (62/62)
- 集成测试通过率: 96.6% (170/176)
- 关键模块覆盖: BaseORT, OcrORT, ColorLayerORT, OCRDatasetEvaluator

## 常见问题 (FAQ)

### Q: 如何选择合适的检测器？
A:
- **实时性优先**: YOLO系列（yolo11n.onnx）
- **精度优先**: RF-DETR（rfdetr-20250811.onnx）
- **平衡需求**: RT-DETR（rtdetr-2024080100.onnx）

### Q: 如何进行完整的推理流程？
A: 参考 `main.py` 示例：
```bash
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show \
    --annotator-preset debug
```

### Q: 如何扩展新的模型架构？
A:
1. 在 `infer_onnx/` 创建新的推理类，继承 `BaseORT`
2. 实现 `_preprocess_static()` 和 `_postprocess()` 方法
3. 在 `onnxtools/__init__.py` 的 `create_detector()` 中注册

### Q: 如何使用TensorRT加速？
A:
```bash
# 安装TensorRT可选依赖
uv pip install pip setuptools wheel
uv pip install -e ".[trt]"

# 构建TensorRT引擎
python tools/build_engine.py \
    --onnx-path models/rtdetr.onnx \
    --engine-path models/rtdetr.engine \
    --fp16
```

### Q: 如何自定义可视化风格？
A: 使用Annotator预设或工厂：
```python
from onnxtools.utils import load_visualization_preset, AnnotatorFactory

# 方式1: 使用预设
annotators = load_visualization_preset('debug')

# 方式2: 自定义组合
factory = AnnotatorFactory()
annotators = [
    factory.create('round_box', roundness=0.4),
    factory.create('rich_label')
]
```

## 相关文件列表

### 根模块文件
- `__init__.py` - 根模块入口，导出公共API
- `pipeline.py` - 完整推理管道实现

### 子模块文档
- [`infer_onnx/CLAUDE.md`](infer_onnx/CLAUDE.md) - 推理引擎模块文档
- [`utils/CLAUDE.md`](utils/CLAUDE.md) - 工具函数模块文档

### 配置文件
- `configs/det_config.yaml` - 检测配置
- `configs/plate.yaml` - OCR配置
- `configs/visualization_presets.yaml` - 可视化预设

## 架构设计

### 模块分层
```
┌─────────────────────────────────────────┐
│        onnxtools (根模块)                │
│  create_detector(), setup_logger()      │
├─────────────────┬───────────────────────┤
│  infer_onnx     │      utils            │
│  (推理引擎)      │    (工具函数)          │
├─────────────────┼───────────────────────┤
│ • BaseORT       │ • pipeline            │
│ • YoloORT       │ • supervision_*       │
│ • RtdetrORT     │ • annotator_*         │
│ • RfdetrORT     │ • ocr_metrics         │
│ • OcrORT        │ • image_processing    │
│ • Evaluators    │ • drawing             │
└─────────────────┴───────────────────────┘
```

### 工作流程
```mermaid
graph LR
    A[输入图像] --> B[create_detector]
    B --> C[BaseORT子类]
    C --> D[__call__ 推理]
    D --> E[后处理]
    E --> F[Supervision转换]
    F --> G[Annotator渲染]
    G --> H[输出结果]
```

## 变更日志 (Changelog)

**2025-11-05 (阶段1.3)** - OCR/分类类架构独立化
- ✅ 更新核心推理接口说明,区分检测器类和分类器/OCR类
- ✅ 强调两类推理类的返回类型差异(Result vs 元组)
- ✅ 添加架构说明注释

**2025-11-05** - 初始化完整模块文档，建立清晰的模块结构
- 创建onnxtools根模块文档
- 更新面包屑导航系统
- 完善API文档和使用示例
- 补充模块结构图和工作流程

**2025-10-11** - Bug修复和配置优化
- TensorRT改为可选依赖组 `[trt]`
- 修复OCR评估器JSON数组支持

**2025-10-10** - OCR评估功能完成
- OCRDatasetEvaluator完整实现
- 三大指标：完全匹配、编辑距离、相似度

**2025-09-30** - Supervision集成和Annotators扩展
- 13种annotator类型支持
- 5种可视化预设场景
- 性能基准测试完成

**2025-10-09** - 核心重构完成
- BaseORT抽象方法强制实现
- OcrORT和ColorLayerORT重构
- 统一__call__推理接口

**2025-09-15** - 初始化项目结构

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/onnxtools/`*
*最后更新: 2025-11-05 15:02:47*
