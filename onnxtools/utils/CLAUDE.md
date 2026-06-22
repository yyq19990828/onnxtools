[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **utils**

# 工具模块 (onnxtools.utils)

## 模块职责

通用工具集：图像预处理、Supervision 可视化（annotator 工厂/预设）、OCR/检测指标、NMS、中文字体查找、日志配置。为推理与评估流程提供数据处理和绘制支持。

## 核心能力与入口

| 能力 | 入口 | 说明 |
|------|------|------|
| 图像预处理 | `image_processing.UltralyticsLetterBox` | letterbox 保持宽高比缩放，返回 `(tensor, scale, orig_shape, ratio_pad)` |
| 结果转换 | `drawing.convert_to_supervision_detections(detections, original_shape, class_names=None)` | `Result`/dict → `sv.Detections` |
| 可视化绘制 | `drawing.draw_detections(image, sv_detections, annotator_pipeline=None)` | 基于 Supervision 绘制，返回标注图 |
| Annotator 工厂/管道 | `supervision_annotator.{AnnotatorFactory, AnnotatorPipeline, AnnotatorType}` | 工厂按类型创建，管道按序 `annotate()` |
| 可视化预设 | `supervision_preset.load_visualization_preset(name)` | 5 种场景：`standard/lightweight/privacy/debug/high_contrast` |
| OCR 标签 | `supervision_labels.create_ocr_labels(detections, ocr_results, color_results)` | 组合 OCR 文本与颜色生成标签列表 |
| OCR 指标 | `ocr_metrics.{calculate_exact_match_accuracy, calculate_normalized_edit_distance, calculate_edit_distance_similarity}` | 完全匹配 / 归一化编辑距离 / 相似度 |
| 检测指标 | `detection_metrics` | 检测性能评估 |
| NMS | `nms.non_max_suppression(boxes, scores, iou_threshold=0.5)` | xyxy 输入，返回保留索引 |
| 字体 | `font_utils.{get_chinese_font_path, get_fallback_font_path}` | 自动搜索系统中文字体 |
| 日志 | `logger.setup_logger(level)` | `DEBUG/INFO/WARNING/ERROR/CRITICAL` |

常用导出可直接 `from onnxtools.utils import ...`（`setup_logger` 从 `onnxtools` 导入）。

### 最小示例
```python
from onnxtools import setup_logger
from onnxtools.utils import (
    UltralyticsLetterBox, convert_to_supervision_detections, draw_detections,
)

setup_logger('INFO')
tensor, scale, orig_shape, ratio_pad = UltralyticsLetterBox(new_shape=(640, 640))(image)
sv_det = convert_to_supervision_detections(result, (h, w), class_names)
out = draw_detections(image, sv_det)
```

```python
from onnxtools.utils.supervision_preset import load_visualization_preset
from onnxtools.utils.supervision_annotator import (
    AnnotatorFactory, AnnotatorPipeline, AnnotatorType,
)

# 用预设
pipeline = AnnotatorPipeline(load_visualization_preset('debug'))
# 或自定义
factory = AnnotatorFactory()
pipeline = AnnotatorPipeline([
    factory.create(AnnotatorType.ROUND_BOX, roundness=0.4, thickness=3),
    factory.create(AnnotatorType.PERCENTAGE_BAR),
    factory.create(AnnotatorType.RICH_LABEL),
])
annotated = pipeline.annotate(image, sv_det)
```

## 模块结构

```
onnxtools/utils/
├── __init__.py                  # 导出公共API
├── drawing.py                   # 可视化绘制 + Supervision 转换（无单独 supervision_converter.py）
├── image_processing.py          # 图像预处理（letterbox）
├── supervision_labels.py        # OCR 标签创建
├── supervision_annotator.py     # Annotator 工厂和管道（13 种类型）
├── supervision_preset.py        # 可视化预设（5 种场景）
├── ocr_metrics.py               # OCR 评估指标
├── detection_metrics.py         # 检测指标
├── nms.py                       # NMS 算法
├── font_utils.py                # 中文字体查找
├── logger.py                    # 日志配置
└── CLAUDE.md                    # 本文档
```

## 数据约定

- **sv.Detections**：`xyxy [N,4]` / `confidence [N]` / `class_id [N]` / `data={'class_name': List[str]}`。
- **Annotator 类型（13 种，按用途）**：边框类 `Box/RoundBox/BoxCorner`；几何类 `Circle/Triangle/Ellipse/Dot`；填充类 `Color/BackgroundOverlay`；特效类 `Halo/PercentageBar`；隐私类 `Blur/Pixelate`。性能区间约 75μs（dot）~1.5ms（blur/pixelate），20 目标目标 <30ms。
- **OCR 指标关系**：`相似度(EDS) = 1 - 归一化编辑距离(NED)`；NED 越小越好，EDS 越大越好，均为 `[0,1]`。

## 重要约定

- **已全面迁移到 Supervision**：所有绘制走 annotator 管道。❌ 不要写新的 PIL-based 绘制代码。
- 转换函数集成在 `drawing.py`，没有单独的 `supervision_converter.py`。
- 新增可视化风格：优先改 `configs/visualization_presets.yaml`，或用 `AnnotatorFactory` 组合；需要新类型才继承 Supervision Annotator 基类。
- 修改公共函数签名/行为时，同步更新本文档与对应测试。

## 关键依赖

opencv-contrib-python(>=4.12) · numpy(>=2.2.6) · supervision(0.26.1) · pillow(>=11.3) · python-levenshtein(>=0.25) · colorlog(>=6.9)。
配置：`configs/visualization_presets.yaml`、`configs/det_config.yaml`。
