[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **utils**

# 工具模块 (onnxtools.utils)

## 模块职责

提供图像处理、可视化绘制、Supervision集成、OCR指标计算和推理管道等通用工具函数，为整个项目提供核心的数据处理和可视化支持。

## 入口和启动

- **主处理管道**: `pipeline.py::initialize_models()`, `pipeline.py::process_frame()`
- **模块导入**: `__init__.py` 导出常用工具函数
- **日志配置**: `logging_config.py::setup_logger()`

### 快速开始
```python
from onnxtools.pipeline import initialize_models, process_frame
from onnxtools import setup_logger

# 设置日志
setup_logger('INFO')

# 初始化所有模型
models = initialize_models(args)
detector, color_classifier, ocr_model, character, class_names, colors, annotator_pipeline = models

# 处理单帧图像
result_img, output_data = process_frame(
    frame, detector, color_classifier, ocr_model,
    character, class_names, colors, args, annotator_pipeline
)
```

## 外部接口

### 1. 图像预处理
```python
from onnxtools.utils import preprocess_image

# 标准图像预处理
processed_img, scale_factor = preprocess_image(image, target_size=(640, 640))
```

### 2. 结果可视化（Supervision集成）
```python
from onnxtools.utils import draw_detections_supervision, convert_to_supervision_detections
import supervision as sv

# 转换为Supervision格式
sv_detections = convert_to_supervision_detections(
    detections=detections,
    original_shape=(h, w),
    class_names=class_names
)

# 使用Supervision绘制
result_img = draw_detections_supervision(
    image, sv_detections, annotator_pipeline
)
```

### 3. Annotator工厂和管道
```python
from onnxtools.utils import AnnotatorFactory, AnnotatorPipeline
from onnxtools.utils import load_visualization_preset

# 使用预设场景
annotators = load_visualization_preset('debug')  # standard, lightweight, privacy, debug, high_contrast
pipeline = AnnotatorPipeline(annotators)

# 或自定义创建
factory = AnnotatorFactory()
pipeline = AnnotatorPipeline([
    factory.create('round_box', roundness=0.4, thickness=3),
    factory.create('percentage_bar'),
    factory.create('rich_label')
])
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

### 5. 日志配置
```python
from onnxtools import setup_logger

# 设置日志级别
setup_logger('INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
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

### 预处理输出格式
```python
preprocess_result = {
    'image': np.ndarray,      # [C, H, W] 预处理后图像
    'scale_factor': float,    # 缩放比例
    'padding': tuple,         # (pad_w, pad_h) 填充尺寸
    'original_shape': tuple   # (H, W) 原始图像尺寸
}
```

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
    'type': str,              # 'round_box', 'box', 'label', etc.
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
  - 边界情况：空字符串、长度差异、插入删除替换
  - 中文字符处理和真实OCR场景
- [x] `test_load_label_file.py` - 标签文件加载12个单元测试
- [ ] 图像预处理函数测试
- [ ] 可视化绘制功能测试

### 集成测试覆盖
- [x] `test_supervision_only.py` - Supervision库集成测试
- [x] `test_basic_drawing.py` - 基础绘制测试
- [x] Annotator集成测试（round_box, box_corner, geometric, fill, privacy等）
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
A: 保持宽高比避免目标变形，提高检测精度。使用padding填充到目标尺寸，后处理时根据scale_factor还原坐标。

### Q: 如何自定义可视化风格？
A: 1) 修改 `configs/visualization_presets.yaml` 添加新预设; 2) 使用AnnotatorFactory自定义参数; 3) 继承Supervision的Annotator基类创建自定义annotator

### Q: OCR指标中的归一化编辑距离和相似度有什么区别？
A:
- 归一化编辑距离(NED): 范围[0,1]，0表示完全匹配，1表示完全不同，值越小越好
- 编辑距离相似度(EDS): 范围[0,1]，0表示完全不同，1表示完全匹配，值越大越好
- 关系: EDS = 1 - NED

### Q: 如何选择合适的Annotator预设？
A:
- `standard`: 通用场景，默认边框+标签
- `lightweight`: 低资源场景，简化可视化
- `privacy`: 隐私保护，模糊化处理
- `debug`: 开发调试，详细信息显示
- `high_contrast`: 高对比度场景，增强视觉效果

### Q: 旧版OCR处理函数去哪了？
A: OCR预处理和后处理函数已迁移到 `onnxtools.infer_onnx.OcrORT` 类的静态方法：
- `process_plate_image()` → `OcrORT._process_plate_image_static()`
- `decode()` → `OcrORT._decode_static()`
详见重构规范 `specs/004-refactor-colorlayeronnx-ocronnx/`

## 相关文件列表

### 核心处理文件
- `pipeline.py` - 主处理管道和模型初始化
- `image_processing.py` - 通用图像预处理工具
- `output_transforms.py` - 输出格式转换工具
- `detection_metrics.py` - 检测性能评估指标
- `nms.py` - 非极大值抑制算法实现

### Supervision集成
- `supervision_converter.py` - 数据格式转换为Supervision
- `supervision_labels.py` - OCR标签创建
- `supervision_annotator.py` - Annotator工厂和管道
- `supervision_preset.py` - 可视化预设加载器

### OCR和度量
- `ocr_metrics.py` - OCR评估指标计算函数
- `font_utils.py` - 中文字体路径查找

### 可视化和工具
- `drawing.py` - 检测结果可视化绘制（传统方式+Supervision方式）
- `logging_config.py` - 日志系统配置

### 配置和接口
- `__init__.py` - 模块导入和API定义

## 架构设计

### Annotator类型支持（13种）
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
  - box
  - rich_label

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

**2025-11-05** - 初始化完整模块文档，建立清晰的面包屑导航
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
- ✅ 更新 `pipeline.py` - 使用新的OCRORT接口
- ✅ 更新 `__init__.py` - 移除OCR函数导出

**2025-09-15** - 初始化工具模块文档

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/onnxtools/utils/`*
*最后更新: 2025-11-05 15:02:47*
