[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **utils**

# 工具模块 (onnxtools.utils)

## 模块职责

提供图像处理、可视化绘制、Supervision集成、OCR指标计算等通用工具函数,为整个项目提供核心的数据处理和可视化支持。

## 入口和启动

- **模块导入**: `__init__.py` 导出常用工具函数
- **日志配置**: `logger.py::setup_logger()`
- **可视化**: `drawing.py::draw_detections()`, `drawing.py::convert_to_supervision_detections()`

### 快速开始
```python
from onnxtools import setup_logger
from onnxtools.utils import (
    preprocess_image,
    draw_detections,
    convert_to_supervision_detections,
    create_ocr_labels,
    get_chinese_font_path,
    non_max_suppression
)

# 设置日志
setup_logger('INFO')

# 图像预处理
processed_img = preprocess_image(image, target_size=(640, 640))

# Supervision可视化
sv_detections = convert_to_supervision_detections(
    detections=result,
    original_shape=(h, w),
    class_names=class_names
)
result_img = draw_detections(image, sv_detections)
```

## 外部接口

### 1. 图像预处理
```python
from onnxtools.utils import preprocess_image

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    keep_ratio: bool = True,
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """图像预处理,调整尺寸并保持宽高比

    Args:
        image: 输入图像 BGR格式
        target_size: 目标尺寸 (width, height)
        keep_ratio: 是否保持宽高比
        pad_color: 填充颜色

    Returns:
        np.ndarray: 预处理后的图像
    """
    pass
```

### 2. 结果可视化(Supervision集成)
```python
from onnxtools.utils import draw_detections, convert_to_supervision_detections
import supervision as sv

def convert_to_supervision_detections(
    detections: Union[Result, Dict],
    original_shape: Tuple[int, int],
    class_names: Optional[Dict[int, str]] = None
) -> sv.Detections:
    """转换检测结果为Supervision格式

    Args:
        detections: Result对象或检测结果字典
        original_shape: 原图尺寸 (H, W)
        class_names: 类别名称映射

    Returns:
        sv.Detections: Supervision检测对象
    """
    pass

def draw_detections(
    image: np.ndarray,
    sv_detections: sv.Detections,
    annotator_pipeline: Optional[AnnotatorPipeline] = None
) -> np.ndarray:
    """使用Supervision绘制检测结果

    Args:
        image: 输入图像
        sv_detections: Supervision检测对象
        annotator_pipeline: Annotator管道(可选)

    Returns:
        np.ndarray: 标注后的图像
    """
    pass
```

### 3. Annotator工厂和管道
```python
from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorPipeline, AnnotatorType
from onnxtools.utils.supervision_preset import load_visualization_preset

# 使用预设场景
annotators = load_visualization_preset('debug')  # standard, lightweight, privacy, high_contrast
pipeline = AnnotatorPipeline(annotators)

# 或自定义创建
factory = AnnotatorFactory()
pipeline = AnnotatorPipeline([
    factory.create(AnnotatorType.ROUND_BOX, roundness=0.4, thickness=3),
    factory.create(AnnotatorType.PERCENTAGE_BAR),
    factory.create(AnnotatorType.RICH_LABEL)
])

# 应用标注
annotated_image = pipeline.annotate(image, sv_detections)
```

### 4. OCR指标计算
```python
from onnxtools.utils.ocr_metrics import (
    calculate_exact_match_accuracy,
    calculate_normalized_edit_distance,
    calculate_edit_distance_similarity
)

# 完全匹配准确率
exact_match = calculate_exact_match_accuracy(predictions, labels)

# 归一化编辑距离 (0=完全匹配, 1=完全不匹配)
ned = calculate_normalized_edit_distance(pred, label)

# 编辑距离相似度 (0=完全不匹配, 1=完全匹配)
similarity = calculate_edit_distance_similarity(pred, label)
```

### 5. OCR标签创建
```python
from onnxtools.utils import create_ocr_labels

def create_ocr_labels(
    detections: sv.Detections,
    ocr_results: List[Optional[Tuple[str, float, List[float]]]],
    color_results: List[Tuple[str, str, float]]
) -> List[str]:
    """创建包含OCR和颜色信息的标签

    Args:
        detections: Supervision检测对象
        ocr_results: OCR识别结果列表
        color_results: 颜色分类结果列表

    Returns:
        List[str]: 格式化的标签列表
    """
    pass
```

### 6. 非极大值抑制(NMS)
```python
from onnxtools.utils import non_max_suppression

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """执行NMS算法

    Args:
        boxes: 边界框数组 [N, 4] xyxy格式
        scores: 置信度数组 [N]
        iou_threshold: IoU阈值

    Returns:
        np.ndarray: 保留的索引数组
    """
    pass
```

### 7. 字体工具
```python
from onnxtools.utils import get_chinese_font_path, get_fallback_font_path

# 获取中文字体路径
font_path = get_chinese_font_path()

# 获取备用字体路径
fallback_font = get_fallback_font_path()
```

### 8. 日志配置
```python
from onnxtools import setup_logger

# 设置日志级别
setup_logger('INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 模块结构

```
onnxtools/utils/
├── __init__.py                  # 导出公共API
├── drawing.py                   # 可视化绘制和Supervision转换
├── image_processing.py          # 图像预处理
├── supervision_labels.py        # OCR标签创建
├── supervision_annotator.py     # Annotator工厂和管道(13种类型)
├── supervision_preset.py        # 可视化预设(5种场景)
├── ocr_metrics.py               # OCR评估指标
├── detection_metrics.py         # 检测指标计算
├── nms.py                       # 非极大值抑制算法
├── font_utils.py                # 字体工具
├── logger.py                    # 日志配置
└── CLAUDE.md                    # 本文档
```

## 关键依赖和配置

### 运行时依赖
- **opencv-contrib-python** (>=4.12.0): 图像处理和绘制
- **numpy** (>=2.2.6): 数值计算
- **colorlog** (>=6.9.0): 彩色日志输出
- **pillow** (>=11.3.0): 图像格式转换和字体渲染
- **supervision** (0.26.1): 高级可视化annotators
- **python-levenshtein** (>=0.25.0): 快速编辑距离计算

### 配置文件
- `configs/visualization_presets.yaml`: Supervision可视化预设配置
- `configs/det_config.yaml`: 检测类别和颜色配置

### 字体文件
- 自动搜索路径: `/usr/share/fonts/`, `/System/Library/Fonts/`, `C:\Windows\Fonts\`
- 支持字体: SimHei, WenQuanYi, Noto Sans CJK, Arial Unicode MS

## 数据模型

### Supervision转换格式
```python
sv_detections = sv.Detections(
    xyxy=np.ndarray,          # [N, 4] 边界框坐标
    confidence=np.ndarray,    # [N] 置信度
    class_id=np.ndarray,      # [N] 类别ID
    data={
        'class_name': List[str],  # 类别名称列表
    }
)
```

### Annotator配置
```python
annotator_config = {
    'type': AnnotatorType,    # 枚举类型
    'params': dict,           # 特定annotator的参数
    'enabled': bool           # 是否启用
}
```

### OCR指标结果
```python
ocr_metrics = {
    'exact_match_accuracy': float,        # [0, 1] 完全匹配准确率
    'normalized_edit_distance': float,    # [0, 1] 归一化编辑距离
    'edit_distance_similarity': float,    # [0, 1] 编辑距离相似度
    'total_samples': int,                 # 总样本数
    'correct_samples': int                # 完全匹配样本数
}
```

## 测试和质量

### 单元测试覆盖
- [x] `test_ocr_metrics.py` - OCR指标计算23个单元测试
  - 边界情况:空字符串、长度差异、插入删除替换
  - 中文字符处理和真实OCR场景
- [x] `test_load_label_file.py` - 标签文件加载12个单元测试
- [ ] 图像预处理函数测试
- [ ] 可视化绘制功能测试

### 集成测试覆盖
- [x] `test_supervision_only.py` - Supervision库集成测试
- [x] `test_basic_drawing.py` - 基础绘制测试
- [x] Annotator集成测试(round_box, box_corner, geometric, fill, privacy等)
- [x] `test_preset_scenarios.py` - 预设场景测试

### 合约测试覆盖
- [x] `test_convert_detections_contract.py` - 数据转换合约
- [x] `test_draw_detections_contract.py` - 可视化API合约
- [x] `test_annotator_factory_contract.py` - Annotator工厂合约
- [x] `test_annotator_pipeline_contract.py` - Annotator管道合约

### 性能测试
- [x] `test_annotator_benchmark.py` - 13种annotator性能基准
  - 最快: 75μs (dot)
  - 最慢: 1.5ms (blur/pixelate)
  - 目标: < 30ms for 20 objects

## 常见问题 (FAQ)

### Q: 图像预处理为什么要保持宽高比？
A: 保持宽高比避免目标变形,提高检测精度。使用padding填充到目标尺寸,后处理时根据scale_factor还原坐标。

### Q: 如何自定义可视化风格？
A:
1. 修改 `configs/visualization_presets.yaml` 添加新预设
2. 使用AnnotatorFactory自定义参数
3. 继承Supervision的Annotator基类创建自定义annotator

### Q: OCR指标中的归一化编辑距离和相似度有什么区别？
A:
- **归一化编辑距离(NED)**: 范围[0,1],0表示完全匹配,1表示完全不同,值越小越好
- **编辑距离相似度(EDS)**: 范围[0,1],0表示完全不同,1表示完全匹配,值越大越好
- **关系**: EDS = 1 - NED

### Q: 如何选择合适的Annotator预设？
A:
- `standard`: 通用场景,默认边框+标签
- `lightweight`: 低资源场景,简化可视化
- `privacy`: 隐私保护,模糊化处理
- `debug`: 开发调试,详细信息显示(OCR文本、置信度条)
- `high_contrast`: 高对比度场景,增强视觉效果

### Q: drawing.py中包含哪些函数？
A:
- `draw_detections()`: 使用Supervision绘制检测结果
- `convert_to_supervision_detections()`: 转换Result/字典为sv.Detections格式
- 注意: 没有单独的 `supervision_converter.py` 文件,转换函数集成在 `drawing.py` 中

## 相关文件列表

### 核心处理文件
- `drawing.py` - 可视化绘制和Supervision数据转换
- `image_processing.py` - 通用图像预处理工具
- `detection_metrics.py` - 检测性能评估指标
- `nms.py` - 非极大值抑制算法实现

### Supervision集成
- `supervision_labels.py` - OCR标签创建
- `supervision_annotator.py` - Annotator工厂和管道(13种类型)
- `supervision_preset.py` - 可视化预设加载器(5种场景)

### OCR和度量
- `ocr_metrics.py` - OCR评估指标计算函数
- `font_utils.py` - 中文字体路径查找

### 系统工具
- `logger.py` - 日志系统配置
- `__init__.py` - 模块导入和API定义

## 架构设计

### Annotator类型支持(13种)
```
边框类:
  - Box (标准边框)
  - RoundBox (圆角边框)
  - BoxCorner (转角标记)

几何标记:
  - Circle (圆形)
  - Triangle (三角形)
  - Ellipse (椭圆)
  - Dot (点标记)

填充类:
  - Color (区域填充)
  - BackgroundOverlay (背景叠加)

特效类:
  - Halo (光晕效果)
  - PercentageBar (置信度条)

隐私保护:
  - Blur (模糊处理)
  - Pixelate (像素化)
```

### 预设场景配置
```yaml
standard:      # 标准模式
  - box_corner
  - label

lightweight:   # 轻量级
  - dot
  - label

privacy:       # 隐私保护
  - box
  - blur
  - label

debug:         # 调试模式
  - round_box
  - percentage_bar
  - rich_label

high_contrast: # 高对比度
  - color
  - background_overlay
  - rich_label
```

## 变更日志 (Changelog)

**2025-11-13** - 文档全面更新和结构修正
- ✅ 修正模块结构:明确 `drawing.py` 包含转换函数,无单独的 `supervision_converter.py`
- ✅ 更新文件列表,反映实际文件组织(`logger.py` 而非 `logging_config.py`)
- ✅ 补充 `AnnotatorType` 枚举类型说明
- ✅ 完善API文档,增加函数签名和返回值说明
- ✅ 更新时间戳至 2025-11-13

**2025-11-05** - 初始化完整模块文档,建立清晰的面包屑导航
- 更新面包屑路径: [根目录] > [onnxtools] > [utils]
- 添加Supervision集成详细文档
- 补充Annotator工厂和预设使用指南
- 完善OCR指标计算说明

**2025-10-10** - 新增OCR指标计算模块
- `ocr_metrics.py` - 三大指标计算函数
- 23个单元测试覆盖所有边界情况
- 支持中文字符和真实OCR场景

**2025-09-30** - Supervision Annotators扩展集成
- 新增13种annotator类型
- 实现AnnotatorFactory和AnnotatorPipeline
- 5种预设场景配置
- 性能基准测试报告

**2025-10-09** - OCR模块重构
- ❌ 删除 `ocr_image_processing.py` - 迁移到infer_onnx
- ❌ 删除 `ocr_post_processing.py` - 迁移到infer_onnx
- ✅ 更新 `__init__.py` - 移除OCR函数导出

**2025-09-15** - 初始化工具模块文档

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/onnxtools/utils/`*
*最后更新: 2025-11-13 20:55:00*
