[根目录](../CLAUDE.md) > **models**

# 模型配置模块 (models)

## 模块职责

存储和管理ONNX模型文件、TensorRT引擎、配置文件和模型相关的元数据，为整个系统提供模型资源和配置支持。

## 入口和启动

- **检测配置**: `det_config.yaml` - 类别名称和可视化配色方案
- **OCR配置**: `plate.yaml` - 字符字典和颜色层级映射
- **模型文件**: `*.onnx` - ONNX格式模型文件
- **引擎文件**: `*.engine` - TensorRT优化引擎文件

## 外部接口

### 配置文件加载
```python
import yaml

# 加载检测配置
with open('configs/det_config.yaml', 'r') as f:
    det_config = yaml.safe_load(f)
    class_names = det_config['class_names']
    colors = det_config['visual_colors']

# 加载OCR配置
with open('configs/plate.yaml', 'r') as f:
    plate_config = yaml.safe_load(f)
    character_dict = plate_config['ocr_dict']
    color_dict = plate_config['color_dict']
```

### 模型文件使用
```python
# ONNX模型路径
detection_model = 'models/rtdetr-2024080100.onnx'
ocr_model = 'models/ocr.onnx'
color_model = 'models/color_layer.onnx'

# TensorRT引擎路径
detection_engine = 'models/rtdetr-2024080100.engine'
```

## 关键依赖和配置

### 模型文件格式要求
- **检测模型**: ONNX格式，支持YOLO/RT-DETR/RF-DETR架构
- **OCR模型**: ONNX格式，CRNN架构，输入[1,3,48,320]
- **颜色分类**: ONNX格式，分类网络，输入[1,3,224,224]

### 配置文件规范
- **YAML格式**: 统一使用UTF-8编码的YAML配置
- **类别映射**: 支持中英文类别名称
- **颜色方案**: 十六进制颜色代码，支持16种可视化颜色

## 数据模型

### 检测配置结构
```yaml
# det_config.yaml
class_names:
  - car           # 汽车
  - truck         # 货车
  - heavy_truck   # 重型卡车
  - van           # 面包车
  - bus           # 公交车
  - bicycle       # 自行车
  - cyclist       # 骑行者
  - tricycle      # 三轮车
  - trolley       # 手推车
  - pedestrain    # 行人
  - cone          # 锥形桶
  - animal        # 动物
  - other         # 其他
  - plate         # 车牌
  - motorcycle    # 摩托车

visual_colors:
  - "#FF3838"     # 鲜红
  - "#FF9D97"     # 珊瑚粉
  # ... 更多颜色
```

### OCR配置结构
```yaml
# plate.yaml
ocr_dict:
  - "0"           # 数字0-9
  - "1"
  # ... 数字和字母
  - A
  - B
  # ... 英文字母
  - 京            # 省份简称
  - 沪
  # ... 中文字符

color_dict:
  0: black        # 黑色车牌
  1: blue         # 蓝色车牌
  2: green        # 绿色车牌
  3: white        # 白色车牌
  4: yellow       # 黄色车牌

layer_dict:
  0: single       # 单层车牌
  1: double       # 双层车牌
```

### 模型元数据
```python
model_metadata = {
    'input_shape': [1, 3, 640, 640],     # 输入形状
    'output_names': ['boxes', 'scores'],  # 输出名称
    'num_classes': 15,                    # 类别数量
    'preprocessing': {                    # 预处理参数
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'format': 'RGB'
    }
}
```

## 测试和质量

### 模型文件验证
- [ ] ONNX模型结构完整性检查
- [ ] 输入输出形状验证
- [ ] TensorRT引擎版本兼容性
- [ ] 配置文件格式验证

### 性能指标
- [ ] 模型文件大小合理性 (< 500MB for detection)
- [ ] 模型加载时间 (< 5s for ONNX, < 2s for TensorRT)
- [ ] 推理精度保持 (ONNX vs TensorRT差异 < 1%)

### 配置完整性
- [ ] 所有类别名称定义完整
- [ ] 颜色配置数量匹配类别数
- [ ] OCR字典包含所有必需字符

## 常见问题 (FAQ)

### Q: 如何添加新的检测类别？
A: 1) 在 `det_config.yaml` 的 `class_names` 中添加新类别; 2) 在 `visual_colors` 中分配对应颜色; 3) 重新训练或更新模型

### Q: OCR识别支持哪些字符？
A: 支持数字0-9、英文字母A-Z（除I、O）、中国省份简称和特殊用途字符（学、警、使、领等）

### Q: TensorRT引擎文件可以跨平台使用吗？
A: 不可以，TensorRT引擎与特定GPU架构和驱动版本绑定，需要在目标平台上重新构建

### Q: 如何优化模型文件大小？
A: 1) 使用模型压缩技术; 2) 选择合适的精度（FP16/INT8）; 3) 使用onnxslim工具优化ONNX模型

## 相关文件列表

### 配置文件
- `det_config.yaml` - 检测模型类别和可视化配置
- `plate.yaml` - OCR字典和车牌颜色层级配置

### ONNX模型文件
- `rtdetr-2024080100.onnx` - RT-DETR检测模型
- `rfdetr-20250811.onnx` - RF-DETR检测模型
- `yolo11n.onnx` - YOLOv11nano检测模型
- `ocr.onnx` - 车牌OCR识别模型
- `color_layer.onnx` - 车牌颜色层级分类模型

### 优化模型文件
- `*slim.onnx` - 使用onnxslim优化的轻量化模型
- `rtdetr-20250729.onnx` - 最新版本RT-DETR模型

### TensorRT引擎文件
- `*.engine` - FP32精度TensorRT引擎
- `*_fp32.engine` - 显式FP32精度引擎
- `rtdetr-20250729topk.engine` - TopK优化引擎

## 变更日志 (Changelog)

**2025-09-15 20:01:23 CST** - 初始化模型配置模块文档，建立模型文件和配置管理规范

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/models/`*