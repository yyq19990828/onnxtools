[根目录](../CLAUDE.md) > **utils**

# 工具模块 (utils)

## 模块职责

提供图像处理、可视化绘制、OCR后处理和推理管道等通用工具函数，为整个项目提供核心的数据处理和可视化支持。

## 入口和启动

- **主处理管道**: `pipeline.py::initialize_models()`, `pipeline.py::process_frame()`
- **模块导入**: `__init__.py` 导出常用工具函数
- **日志配置**: `logging_config.py::setup_logger()`

## 外部接口

### 主处理管道
```python
from utils.pipeline import initialize_models, process_frame

# 初始化所有模型
models = initialize_models(args)
detector, color_classifier, ocr_model, character, class_names, colors = models

# 处理单帧图像
result_img, output_data = process_frame(
    frame, detector, color_classifier, ocr_model,
    character, class_names, colors, args
)
```

### 图像预处理
```python
from utils import preprocess_image

# 标准图像预处理
processed_img, scale_factor = preprocess_image(image, target_size=(640, 640))
```

### 结果可视化
```python
from utils import draw_detections

# 绘制检测结果
result_img = draw_detections(
    image, boxes, scores, class_ids, class_names, colors
)
```

### OCR相关处理 (已迁移到infer_onnx模块)
```python
# ⚠️ 注意: OCR预处理和后处理函数已迁移到infer_onnx.OCRONNX类
# 旧版本:
# from utils import process_plate_image, decode

# 新版本: 使用OCRONNX类的统一接口
from infer_onnx import OCRONNX

ocr_model = OCRONNX('models/ocr.onnx', character=character_dict)
result = ocr_model(plate_image, is_double_layer=True)
if result:
    text, confidence, char_confs = result

# 如果需要独立调用预处理函数:
from infer_onnx.ocr_onnx import OCRONNX
processed = OCRONNX._process_plate_image_static(plate_img, is_double_layer=True)
```

## 关键依赖和配置

### 运行时依赖
- **opencv-python**: 图像处理和绘制
- **numpy**: 数值计算
- **colorlog**: 彩色日志输出
- **pillow**: 图像格式转换

### 配置参数
- 图像预处理: 目标尺寸、填充模式、归一化参数
- 可视化配置: 字体、颜色、线条粗细
- OCR参数: 字符高度、宽度比例、置信度阈值

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

### 检测绘制参数
```python
draw_config = {
    'font_scale': 0.6,        # 字体大小
    'thickness': 2,           # 线条粗细
    'text_thickness': 1,      # 文字粗细
    'alpha': 0.3             # 透明度
}
```

### OCR处理配置
```python
ocr_config = {
    'target_height': 48,      # 目标高度
    'target_width': 320,      # 目标宽度
    'mean': [0.5, 0.5, 0.5],  # 均值归一化
    'std': [0.5, 0.5, 0.5]    # 标准差归一化
}
```

## 测试和质量

### 单元测试覆盖
- [ ] 图像预处理函数测试
- [ ] OCR图像处理测试
- [ ] 后处理解码算法测试
- [ ] 可视化绘制功能测试

### 性能要求
- [ ] 图像预处理性能 (< 5ms for 1920x1080)
- [ ] OCR预处理性能 (< 2ms for 200x100 plate)
- [ ] 绘制渲染性能 (< 10ms for 20 objects)

### 质量指标
- [ ] 图像预处理质量保持
- [ ] OCR预处理增强效果
- [ ] 可视化结果清晰度

## 常见问题 (FAQ)

### Q: 图像预处理为什么要保持宽高比？
A: 保持宽高比避免目标变形，提高检测精度。使用padding填充到目标尺寸。

### Q: OCR预处理有哪些关键步骤？
A: 1) 灰度转换; 2) 尺寸归一化; 3) 对比度增强; 4) 噪声去除; 5) 标准化

### Q: 如何自定义可视化颜色？
A: 修改 `configs/det_config.yaml` 中的 `visual_colors` 配置项

### Q: NMS参数如何调优？
A: 根据检测场景调整IoU阈值，密集场景降低阈值，稀疏场景提高阈值

## 相关文件列表

### 核心处理文件
- `pipeline.py` - 主处理管道和模型初始化
- `image_processing.py` - 通用图像预处理工具
- ~~`ocr_image_processing.py`~~ - ❌ 已删除,功能迁移到`infer_onnx.OCRONNX`
- ~~`ocr_post_processing.py`~~ - ❌ 已删除,功能迁移到`infer_onnx.OCRONNX`

### 可视化和工具
- `drawing.py` - 检测结果可视化绘制
- `nms.py` - 非极大值抑制算法实现
- `output_transforms.py` - 输出格式转换工具
- `detection_metrics.py` - 检测性能评估指标

### 配置和日志
- `logging_config.py` - 日志系统配置
- `__init__.py` - 模块导入和API定义

## 变更日志 (Changelog)

**2025-10-09** - 完成OCR模块重构 (004-refactor-colorlayeronnx-ocronnx)
- ❌ 删除`ocr_image_processing.py` - 所有预处理函数迁移到`infer_onnx.OCRONNX`静态方法
- ❌ 删除`ocr_post_processing.py` - 所有后处理函数迁移到`infer_onnx.OCRONNX`静态方法
- ✅ 更新`pipeline.py` - 改用OCRONNX和ColorLayerONNX的统一`__call__()`接口
- ✅ 更新`__init__.py` - 移除OCR相关函数导出
- ⚠️ 迁移指南: 使用`OCRONNX._process_plate_image_static()`替代`process_plate_image()`

**2025-09-15 20:01:23 CST** - 初始化工具模块文档，建立图像处理和可视化工具规范

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/`*