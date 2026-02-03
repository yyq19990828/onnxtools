# Annotator使用指南

本文档说明如何在`main.py`中使用新的annotator功能进行可视化。

## 快速开始

### 1. 使用预设场景

最简单的方式是使用预定义的可视化预设：

```bash
# 标准检测模式（默认边框+标签）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-preset standard

# 轻量级模式（点标记+简单标签）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-preset lightweight

# 隐私保护模式（边框+车牌模糊）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-preset privacy

# 调试模式（圆角框+置信度条+详细标签）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-preset debug

# 高对比度模式（区域填充+背景变暗）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-preset high_contrast
```

### 2. 自定义Annotator组合

您也可以自定义选择多个annotator：

```bash
# 圆角边框 + 置信度条
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-types round_box percentage_bar

# 点标记 + 光晕效果
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-types dot halo

# 边框 + 模糊（隐私保护）
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-types box blur
```

### 3. 调整参数

可以通过额外的参数调整annotator效果：

```bash
# 自定义圆角边框参数
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-types round_box \
    --box-thickness 4 \
    --roundness 0.5

# 自定义模糊效果
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --annotator-types blur \
    --blur-kernel-size 25
```

## 可用的Annotator类型

### 边框类 (Border Annotators)
- `box` - 标准方形边框
- `round_box` - 圆角边框
- `box_corner` - 仅绘制四个角点

### 几何标记类 (Geometric Markers)
- `circle` - 圆形标注
- `triangle` - 三角形标注
- `ellipse` - 椭圆标注
- `dot` - 点标记

### 填充类 (Fill Annotators)
- `color` - 区域透明填充
- `background_overlay` - 背景叠加变暗

### 特效类 (Effect Annotators)
- `halo` - 光晕效果
- `percentage_bar` - 置信度条形图
- `rich_label` - 富文本标签

### 隐私保护类 (Privacy Annotators)
- `blur` - 模糊处理
- `pixelate` - 像素化处理（注意：当前版本有bug）

## 命令行参数详解

### --annotator-preset
使用预定义的可视化预设，会覆盖`--annotator-types`设置。

**可选值**:
- `standard` - 标准检测模式
- `lightweight` - 轻量级模式
- `privacy` - 隐私保护模式
- `debug` - 调试分析模式
- `high_contrast` - 高对比度模式

### --annotator-types
自定义annotator类型列表，可以指定多个。

**示例**:
```bash
--annotator-types round_box rich_label
--annotator-types dot percentage_bar halo
```

### --box-thickness
边框和角点的线条粗细（像素）。

**默认值**: 2
**适用于**: `box`, `round_box`, `box_corner`

### --roundness
圆角边框的圆角程度（0.0-1.0）。

**默认值**: 0.3
**适用于**: `round_box`

### --blur-kernel-size
模糊效果的核大小（奇数）。

**默认值**: 15
**适用于**: `blur`

## 性能建议

根据性能测试结果，不同annotator的渲染时间差异较大：

### 高性能场景（> 30 FPS）
推荐使用轻量级annotator：
- `dot` (108 μs)
- `halo` (75 μs)
- `triangle` (144 μs)
- `percentage_bar` (157 μs)

### 平衡场景（> 15 FPS）
推荐使用标准annotator：
- `box` (226 μs)
- `box_corner` (278 μs)
- `round_box` (411 μs)

### 质量优先场景
可以使用复杂annotator：
- `color` (479 μs)
- `blur` (504 μs)
- `background_overlay` (1,506 μs)

## 常见问题

### Q: 预设和自定义annotator可以同时使用吗？
A: 不可以。`--annotator-preset`会覆盖`--annotator-types`设置。

### Q: 如何使用旧版绘制系统？
A: 不指定`--annotator-preset`和`--annotator-types`即可自动使用旧版`draw_detections()`。

### Q: 某些annotator组合会有冲突吗？
A: 会。系统会自动检测冲突并输出警告，但仍允许执行。例如：
- `color` + `blur` - 填充会遮挡模糊效果
- `box` + `round_box` - 同时绘制两种边框

### Q: 如何查看所有可用的参数？
A: 运行 `python main.py --help` 查看完整参数列表。

### Q: 性能不够怎么办？
A:
1. 使用轻量级annotator（dot, halo, triangle）
2. 减少annotator数量
3. 降低图像分辨率
4. 使用`--frame-skip`参数跳帧处理

## 完整示例

### 示例1: 实时视频处理（轻量级）
```bash
python main.py \
    --model-path models/yolo11n.onnx \
    --model-type yolo \
    --input 0 \
    --output-mode show \
    --annotator-preset lightweight \
    --frame-skip 2
```

### 示例2: 离线视频处理（高质量）
```bash
python main.py \
    --model-path models/rfdetr.onnx \
    --model-type rfdetr \
    --input videos/traffic.mp4 \
    --output-mode save \
    --annotator-preset debug \
    --save-json
```

### 示例3: 图像批处理（自定义）
```bash
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/images/ \
    --output-mode save \
    --annotator-types round_box percentage_bar rich_label \
    --box-thickness 3 \
    --roundness 0.4
```

## 技术细节

### 实现原理
- 新的annotator系统基于supervision库（v0.26.0+）
- 使用工厂模式动态创建annotator实例
- 使用管道模式组合多个annotator
- 自动检测detections格式并转换为supervision格式
- 向后兼容：未指定annotator时自动回退到旧版绘制

### 代码位置
- Annotator工厂: `utils/annotator_factory.py`
- 预设加载器: `utils/visualization_preset.py`
- Pipeline集成: `utils/pipeline.py::create_annotator_pipeline()`
- 主程序集成: `utils/pipeline.py::process_frame()`

---

**更新日期**: 2025-09-30
**版本**: 1.0.0
**相关文档**:
- [性能报告](../specs/003-add-more-annotators/performance_report.md)
- [快速入门](../specs/003-add-more-annotators/quickstart.md)
