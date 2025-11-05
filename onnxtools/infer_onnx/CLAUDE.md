[根目录](../CLAUDE.md) > **infer_onnx**

# 推理引擎模块 (infer_onnx)

## 模块职责

核心ONNX推理引擎，提供多种目标检测模型架构的统一接口，包括车辆检测、车牌检测、OCR识别和颜色/层级分类功能。

## 入口和启动

- **主要工厂函数**: `infer_models.py::create_detector()`
- **基础类**: `base_onnx.py::BaseOnnx`
- **模块导入**: `__init__.py` 提供统一的API接口

## 外部接口

### 核心检测器创建
```python
from infer_onnx import create_detector

# 创建检测器实例
detector = create_detector(
    model_type='rtdetr',  # 'yolo', 'rtdetr', 'rfdetr'
    onnx_path='models/rtdetr-2024080100.onnx',
    conf_thres=0.5,
    iou_thres=0.5
)
```

### OCR和颜色分类 (重构后 - 继承BaseOnnx)
```python
from infer_onnx import ColorLayerONNX, OCRONNX
import yaml

# 加载配置
with open('configs/plate.yaml') as f:
    config = yaml.safe_load(f)

# 车牌颜色和层级分类 (统一的__call__接口)
color_classifier = ColorLayerONNX(
    onnx_path='models/color_layer.onnx',
    color_map=config['color_map'],
    layer_map=config['layer_map'],
    input_shape=(48, 168),
    conf_thres=0.5
)
color, layer, conf = color_classifier(plate_image)

# 车牌OCR识别 (支持双层车牌自动处理)
ocr_model = OCRONNX(
    onnx_path='models/ocr.onnx',
    character=config['plate_dict']['character'],
    input_shape=(48, 168),
    conf_thres=0.7
)
result = ocr_model(plate_image, is_double_layer=True)
if result:
    text, confidence, char_confs = result
    print(f"识别结果: {text}, 置信度: {confidence:.3f}")
```

### 模型评估
```python
from infer_onnx import DatasetEvaluator

# COCO格式数据集评估
evaluator = DatasetEvaluator(dataset_path, annotations_path)
```

## 关键依赖和配置

### 运行时依赖
- **onnxruntime-gpu**: ONNX模型推理引擎
- **numpy**: 数值计算和张量操作
- **opencv-python**: 图像预处理和后处理
- **yaml**: 配置文件解析

### 配置文件
- `../configs/det_config.yaml`: 检测类别名称和可视化颜色
- `../configs/plate.yaml`: OCR字典和颜色/层级映射

### 模型文件要求
- 检测模型: 支持YOLO/RT-DETR/RF-DETR格式的ONNX文件
- OCR模型: 输入形状为[1, 3, 48, 320]的CRNN模型
- 颜色分类: 输入形状为[1, 3, 224, 224]的分类模型

## 数据模型

### 检测结果结构
```python
# 检测输出格式
detection_result = {
    'boxes': np.ndarray,     # [N, 4] xyxy格式边界框
    'scores': np.ndarray,    # [N] 置信度分数
    'class_ids': np.ndarray, # [N] 类别ID
    'mask': np.ndarray       # [N] NMS后的有效掩码
}
```

### OCR结果结构
```python
# OCR识别输出
ocr_result = {
    'text': str,           # 识别的文本内容
    'confidence': float,   # 识别置信度
    'char_scores': list    # 每个字符的置信度
}
```

### 颜色分类结果
```python
# 颜色和层级分类输出
classification_result = {
    'color': str,          # 车牌颜色: 'blue', 'yellow', 'white', 'black', 'green'
    'layer': str,          # 车牌层数: 'single', 'double'
    'color_conf': float,   # 颜色分类置信度
    'layer_conf': float    # 层级分类置信度
}
```

## 测试和质量

### 单元测试范围
- [ ] 基础推理引擎 (`BaseOnnx`) 功能测试
- [ ] 多模型架构兼容性测试 (YOLO/RT-DETR/RF-DETR)
- [ ] OCR和颜色分类准确性测试
- [ ] 模型加载和错误处理测试

### 性能基准
- [ ] 推理延迟测试 (< 50ms for 640x640 image)
- [ ] GPU内存使用监控 (< 2GB for batch_size=1)
- [ ] 模型精度评估 (mAP@0.5 > 0.85)

### 集成测试
- [ ] 端到端检测管道测试
- [ ] TensorRT引擎兼容性测试
- [ ] 批处理推理测试

## 常见问题 (FAQ)

### Q: 如何添加新的模型架构？
A: 1) 继承 `BaseOnnx` 基类; 2) 实现 `predict()` 和 `postprocess()` 方法; 3) 在 `infer_models.py` 中注册新模型类型

### Q: 模型推理速度慢怎么优化？
A: 1) 使用TensorRT引擎替代ONNX; 2) 调整输入分辨率; 3) 使用FP16精度; 4) 启用GPU推理

### Q: OCR识别准确率低怎么改善？
A: 1) 检查车牌图像预处理质量; 2) 调整OCR模型置信度阈值; 3) 使用更大的OCR模型; 4) 增加训练数据

### Q: 支持哪些ONNX模型版本？
A: 当前支持ONNX opset版本11-17，推荐使用opset 17以获得最佳兼容性

## 相关文件列表

### 核心推理文件
- `base_onnx.py` - 基础ONNX推理抽象类
- `yolo_onnx.py` - YOLO系列模型推理实现
- `rtdetr_onnx.py` - RT-DETR模型推理实现
- `rfdetr_onnx.py` - RF-DETR模型推理实现
- `infer_models.py` - 模型工厂和统一接口
- `infer_utils.py` - 推理工具函数

### 专用功能模块
- `ocr_onnx.py` - OCR识别和颜色分类推理
- `eval_coco.py` - COCO数据集评估工具
- `engine_dataloader.py` - TensorRT引擎数据加载器

### 配置和接口
- `__init__.py` - 模块导入和API定义

## 变更日志 (Changelog)

**2025-10-09** - 完成ColorLayerONNX和OCRONNX重构 (004-refactor-colorlayeronnx-ocronnx)
- ✅ ColorLayerONNX和OCRONNX成功继承BaseOnnx
- ✅ 统一的`__call__()`接口替代旧版`infer()`方法
- ✅ 所有预处理和后处理函数迁移到类内部作为静态方法
- ✅ 删除utils/ocr_image_processing.py和utils/ocr_post_processing.py依赖
- ✅ 支持Polygraphy懒加载和provider自动检测
- ✅ 27个单元测试全部通过,115/122集成测试通过
- ⚠️ 旧版`infer()`方法保留但已标记为deprecated

**2025-09-15 20:01:23 CST** - 初始化推理引擎模块文档，建立多模型架构文档

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/infer_onnx/`*