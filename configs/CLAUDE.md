[根目录](../CLAUDE.md) > **configs**

# 配置文件模块 (configs)

## 模块职责

提供YAML格式的配置文件,包含检测类别映射、OCR字符字典、颜色/层级映射和Supervision可视化预设,为推理引擎和可视化工具提供统一的配置源。

## 入口和启动

配置文件通过Python的`yaml`库加载:

```python
import yaml

# 加载检测配置
with open('configs/det_config.yaml') as f:
    det_config = yaml.safe_load(f)
    class_names = det_config['class_names']
    visual_colors = det_config['visual_colors']

# 加载OCR配置
with open('configs/plate.yaml') as f:
    plate_config = yaml.safe_load(f)
    ocr_dict = plate_config['ocr_dict']
    color_map = plate_config['color_dict']
    layer_map = plate_config['layer_dict']

# 加载可视化预设
with open('configs/visualization_presets.yaml') as f:
    presets = yaml.safe_load(f)['presets']
    debug_preset = presets['debug']
```

## 外部接口

### 1. det_config.yaml - 检测类别和颜色配置

```yaml
class_names:
  - car           # 类别0
  - truck         # 类别1
  - heavy_truck   # 类别2
  - van           # 类别3
  # ... 共16个类别

visual_colors:
  - "#FF3838"     # 类别0的可视化颜色(十六进制RGB)
  - "#FF9D97"     # 类别1的可视化颜色
  # ... 共16个颜色
```

**用途**:
- 为检测器提供类别ID到名称的映射
- 为Supervision Annotator提供类别专属颜色

**使用场景**:
```python
from onnxtools import create_detector
import yaml

with open('configs/det_config.yaml') as f:
    config = yaml.safe_load(f)

detector = create_detector('rtdetr', 'models/rtdetr.onnx')
result = detector(image)

# 使用类别名称
for class_id in result.class_ids:
    class_name = config['class_names'][class_id]
    print(f"Detected: {class_name}")

# 使用可视化颜色
import supervision as sv
detections = result.to_supervision()
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
annotated = box_annotator.annotate(image, detections)
```

### 2. plate.yaml - OCR字典和映射配置

```yaml
ocr_dict:
  - "0"           # 数字0-9
  - "1"
  # ...
  - A             # 字母A-Z(不含I和O)
  - B
  # ...
  - 京            # 省份简称
  - 沪
  # ...
  - 学            # 特殊字符(学、警、使、领等)
  # 共85个字符

color_dict:
  0: black        # 黑牌
  1: blue         # 蓝牌
  2: green        # 绿牌
  3: white        # 白牌
  4: yellow       # 黄牌

layer_dict:
  0: single       # 单层车牌
  1: double       # 双层车牌
```

**用途**:
- OCR模型的字符解码字典
- 颜色/层级分类器的输出映射

**使用场景**:
```python
from onnxtools import OcrORT, ColorLayerORT
import yaml

with open('configs/plate.yaml') as f:
    config = yaml.safe_load(f)

# OCR识别
ocr_model = OcrORT('models/ocr.onnx', character=config['ocr_dict'])
text, conf, char_scores = ocr_model(plate_image)
print(f"识别结果: {text}")

# 颜色/层级分类
classifier = ColorLayerORT(
    'models/color_layer.onnx',
    color_map=config['color_dict'],
    layer_map=config['layer_dict']
)
color, layer, conf = classifier(plate_image)
print(f"颜色: {color}, 层级: {layer}")
```

### 3. visualization_presets.yaml - Supervision可视化预设

```yaml
presets:
  standard:
    name: "标准检测模式"
    description: "默认边框+标签,适用于通用检测场景"
    annotators:
      - type: box_corner
        thickness: 2
      - type: rich_label
        font_size: 25

  lightweight:
    name: "简洁轻量模式"
    description: "点标记+简单标签,最小视觉干扰"
    annotators:
      - type: dot
        radius: 5
        position: CENTER
      - type: rich_label
        font_size: 14

  privacy:
    name: "隐私保护模式"
    description: "边框+车牌模糊,保护敏感信息"
    annotators:
      - type: box
        thickness: 2
      - type: blur
        kernel_size: 15

  debug:
    name: "调试分析模式"
    description: "圆角框+置信度条+详细标签,展示所有信息"
    annotators:
      - type: round_box
        thickness: 3
        roundness: 0.3
      - type: percentage_bar
        height: 16
        width: 80
      - type: rich_label
        font_size: 18

  high_contrast:
    name: "高对比展示模式"
    description: "光晕效果+背景变暗,突出检测对象"
    annotators:
      - type: halo
        opacity: 0.8
      - type: background_overlay
        opacity: 0.5
        color: black
```

**用途**:
- 为Supervision Annotator提供预定义的可视化组合
- 支持5种场景: standard、lightweight、privacy、debug、high_contrast

**使用场景**:
```python
from onnxtools.utils import load_visualization_preset

# 方式1: 加载预设
annotators = load_visualization_preset('debug')
for annotator in annotators:
    image = annotator.annotate(image, detections)

# 方式2: 命令行使用
# python main.py --annotator-preset debug --input test.jpg
```

## 模块结构

```
configs/
├── det_config.yaml              # 检测类别和颜色(16个类别)
├── plate.yaml                   # OCR字典(85字符)和映射
└── visualization_presets.yaml   # Supervision预设(5种场景)
```

## 关键依赖和配置

### 配置加载依赖
- `pyyaml>=6.0.2` - YAML解析库

### 配置验证
- 类别名称列表长度必须与visual_colors长度一致
- OCR字典字符必须唯一且不能包含空字符
- 颜色/层级映射的键必须是整数索引

## 数据模型

### 检测配置模型
```python
det_config = {
    'class_names': List[str],        # 类别名称列表,索引对应类别ID
    'visual_colors': List[str]       # 十六进制颜色字符串,如"#FF3838"
}

# 使用示例
class_id = 0
class_name = det_config['class_names'][class_id]  # "car"
color = det_config['visual_colors'][class_id]     # "#FF3838"
```

### OCR配置模型
```python
plate_config = {
    'ocr_dict': List[str],           # OCR字符字典,索引对应CTC输出
    'color_dict': Dict[int, str],    # 颜色映射 {索引: 颜色名}
    'layer_dict': Dict[int, str]     # 层级映射 {索引: 层级名}
}

# 使用示例
char_index = 36
character = plate_config['ocr_dict'][char_index]  # "京"

color_index = 1
color_name = plate_config['color_dict'][color_index]  # "blue"

layer_index = 0
layer_name = plate_config['layer_dict'][layer_index]  # "single"
```

### 可视化预设模型
```python
visualization_presets = {
    'presets': {
        'preset_name': {
            'name': str,              # 预设显示名称
            'description': str,       # 预设描述
            'annotators': List[Dict]  # Annotator配置列表
        }
    }
}

# 单个annotator配置
annotator_config = {
    'type': str,                      # Annotator类型(如'box', 'round_box')
    # 类型特定参数
    'thickness': int,                 # 线条粗细
    'roundness': float,               # 圆角程度
    'font_size': int,                 # 字体大小
    # ...
}
```

## 测试和质量

### 配置验证测试
```python
import yaml
import pytest

def test_det_config_valid():
    """验证检测配置格式正确"""
    with open('configs/det_config.yaml') as f:
        config = yaml.safe_load(f)

    assert 'class_names' in config
    assert 'visual_colors' in config
    assert len(config['class_names']) == len(config['visual_colors'])
    assert len(config['class_names']) == 16

    # 验证颜色格式
    for color in config['visual_colors']:
        assert color.startswith('#')
        assert len(color) == 7

def test_plate_config_valid():
    """验证OCR配置格式正确"""
    with open('configs/plate.yaml') as f:
        config = yaml.safe_load(f)

    assert 'ocr_dict' in config
    assert 'color_dict' in config
    assert 'layer_dict' in config

    # 验证OCR字典唯一性
    ocr_dict = config['ocr_dict']
    assert len(ocr_dict) == len(set(ocr_dict))
    assert len(ocr_dict) == 85

    # 验证颜色映射
    color_dict = config['color_dict']
    expected_colors = {'black', 'blue', 'green', 'white', 'yellow'}
    assert set(color_dict.values()) == expected_colors

    # 验证层级映射
    layer_dict = config['layer_dict']
    expected_layers = {'single', 'double'}
    assert set(layer_dict.values()) == expected_layers

def test_visualization_presets_valid():
    """验证可视化预设格式正确"""
    with open('configs/visualization_presets.yaml') as f:
        config = yaml.safe_load(f)

    assert 'presets' in config
    presets = config['presets']

    expected_presets = {'standard', 'lightweight', 'privacy', 'debug', 'high_contrast'}
    assert set(presets.keys()) == expected_presets

    for preset_name, preset in presets.items():
        assert 'name' in preset
        assert 'description' in preset
        assert 'annotators' in preset
        assert len(preset['annotators']) > 0

        for annotator in preset['annotators']:
            assert 'type' in annotator
```

### 运行测试
```bash
# 创建测试文件并运行
pytest tests/unit/test_configs.py -v
```

## 常见问题 (FAQ)

### Q: 如何添加新的检测类别?
A:
1. 在`det_config.yaml`的`class_names`列表末尾添加新类别名
2. 在`visual_colors`列表末尾添加对应的十六进制颜色
3. 确保两个列表长度一致

```yaml
class_names:
  # ... 现有类别
  - new_class       # 新类别

visual_colors:
  # ... 现有颜色
  - "#123456"       # 新颜色
```

### Q: 如何修改OCR支持的字符?
A: 修改`plate.yaml`中的`ocr_dict`列表,但需注意:
1. 字符顺序必须与训练时使用的字典一致
2. 不能有重复字符
3. 修改后需要重新训练OCR模型

### Q: 如何自定义可视化预设?
A: 在`visualization_presets.yaml`的`presets`下添加新预设:

```yaml
presets:
  # ... 现有预设
  my_custom:
    name: "自定义模式"
    description: "我的自定义可视化组合"
    annotators:
      - type: box
        thickness: 3
        color: red
      - type: rich_label
        font_size: 20
```

支持的annotator类型: `box`, `round_box`, `box_corner`, `circle`, `triangle`, `ellipse`, `dot`, `color`, `background_overlay`, `halo`, `percentage_bar`, `blur`, `pixelate`

### Q: 配置文件修改后需要重启程序吗?
A: 是的,配置文件在程序启动时加载,修改后需要重新运行推理程序或评估脚本。

### Q: 如何验证配置文件格式正确?
A: 使用Python YAML库验证:

```bash
python -c "import yaml; yaml.safe_load(open('configs/det_config.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/plate.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/visualization_presets.yaml'))"
```

无报错则格式正确。

### Q: 为什么OCR字典有85个字符?
A: 包含:
- 数字: 0-9 (10个)
- 字母: A-Z (不含I和O,24个)
- 省份简称: 京沪津渝等 (31个)
- 特殊字符: 学警使领港澳等 (20个)

覆盖中国大陆车牌的所有可能字符组合。

## 相关文件列表

### 配置文件
- `configs/det_config.yaml` - 检测类别和颜色配置(16个类别)
- `configs/plate.yaml` - OCR字典和映射配置(85字符)
- `configs/visualization_presets.yaml` - Supervision预设(5种场景)

### 使用配置的模块
- `onnxtools/infer_onnx/onnx_ocr.py` - 使用ocr_dict和color/layer映射
- `onnxtools/utils/supervision_preset.py` - 加载visualization_presets
- `main.py` - 加载det_config获取类别名称

### 测试文件
- `tests/unit/test_configs.py` - 配置文件格式验证(建议创建)
- `tests/conftest.py` - 使用plate_config fixture

## 使用示例

### 完整配置加载示例
```python
import yaml
from pathlib import Path

def load_all_configs():
    """加载所有配置文件"""
    config_dir = Path('configs')

    # 检测配置
    with open(config_dir / 'det_config.yaml') as f:
        det_config = yaml.safe_load(f)

    # OCR配置
    with open(config_dir / 'plate.yaml') as f:
        plate_config = yaml.safe_load(f)

    # 可视化预设
    with open(config_dir / 'visualization_presets.yaml') as f:
        vis_config = yaml.safe_load(f)

    return {
        'detection': det_config,
        'ocr': plate_config,
        'visualization': vis_config
    }

# 使用配置
configs = load_all_configs()
print(f"检测类别数: {len(configs['detection']['class_names'])}")
print(f"OCR字符数: {len(configs['ocr']['ocr_dict'])}")
print(f"可视化预设数: {len(configs['visualization']['presets'])}")
```

### 在推理管道中使用配置
```python
import yaml
from onnxtools import create_detector, OcrORT
from onnxtools.utils import load_visualization_preset

# 加载配置
with open('configs/det_config.yaml') as f:
    det_config = yaml.safe_load(f)

with open('configs/plate.yaml') as f:
    plate_config = yaml.safe_load(f)

# 创建检测器
detector = create_detector('rtdetr', 'models/rtdetr.onnx')

# 创建OCR模型
ocr_model = OcrORT('models/ocr.onnx', character=plate_config['ocr_dict'])

# 加载可视化预设
annotators = load_visualization_preset('debug')

# 推理
result = detector(image)
for class_id in result.class_ids:
    class_name = det_config['class_names'][class_id]
    print(f"Detected: {class_name}")

# 可视化
detections = result.to_supervision()
for annotator in annotators:
    image = annotator.annotate(image, detections)
```

## 变更日志 (Changelog)

**2025-11-07** - 创建配置文件模块文档
- 初始化完整的configs模块文档
- 记录3个配置文件的格式和用途
- 补充使用示例和常见问题
- 添加配置验证测试代码

**2025-09-30** - 新增visualization_presets.yaml
- 5种Supervision可视化预设场景
- 13种Annotator类型支持

**2025-09-15** - 初始化配置文件
- det_config.yaml: 16个检测类别
- plate.yaml: 85个OCR字符

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/configs/`*
*最后更新: 2025-11-07 16:35:25*
