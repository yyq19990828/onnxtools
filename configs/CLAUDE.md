[根目录](../CLAUDE.md) > **configs**

# 配置文件模块 (configs)

## 模块职责

YAML 配置源,为推理引擎和可视化工具提供检测类别映射、OCR 字符字典、颜色/层级映射和 Supervision 可视化预设。通过 `yaml.safe_load` 加载。

## 配置优先级

代码内置默认值;若存在外部 YAML 则外部值覆盖默认值。修改配置文件后需重启程序(仅启动时加载一次)。

## 配置文件清单

| 文件 | 用途 | 规模 |
|------|------|------|
| `det_config.yaml` | 检测类别名与可视化颜色 | 16 个类别 |
| `plate.yaml` | OCR 字符字典 + 颜色/层级映射 | 85 字符 |
| `visualization_presets.yaml` | Supervision 可视化预设组合 | 5 种场景 |

使用方:
- `onnxtools/infer_onnx/onnx_ocr.py` — 使用 `ocr_dict` 和颜色/层级映射
- `onnxtools/utils/supervision_preset.py` — 加载 `visualization_presets`
- `examples/demo_pipeline.py` — 加载 `det_config` 获取类别名称
- `tests/conftest.py` — `plate_config` fixture

## 字段参考

### det_config.yaml

| 字段 | 类型 | 说明 |
|------|------|------|
| `class_names` | `List[str]` | 类别名,索引对应类别 ID |
| `visual_colors` | `List[str]` | 十六进制颜色(如 `#FF3838`),索引对应类别 ID |

约束:两列表长度必须一致(均为 16);颜色为 7 字符 `#RRGGBB`。

### plate.yaml

| 字段 | 类型 | 说明 |
|------|------|------|
| `ocr_dict` | `List[str]` | OCR 字符字典,索引对应 CTC 输出 |
| `color_dict` | `Dict[int, str]` | 颜色映射 `{索引: 颜色名}` |
| `layer_dict` | `Dict[int, str]` | 层级映射 `{索引: 层级名}` |

- `ocr_dict`:85 字符 = 数字 0-9(10)+ 字母 A-Z 去除 I/O(24)+ 省份简称(31)+ 特殊字符 学警使领港澳等(20)。顺序必须与训练字典一致且唯一,改动需重训模型。
- `color_dict`:`0 black / 1 blue / 2 green / 3 white / 4 yellow`
- `layer_dict`:`0 single / 1 double`

### visualization_presets.yaml

结构:`presets.<名称> = { name: str, description: str, annotators: List[Dict] }`。
每个 annotator:`{ type: str, ...类型特定参数 }`(如 `thickness`、`roundness`、`font_size`、`radius`、`kernel_size`、`opacity`、`color` 等)。

| 预设 | 组合 |
|------|------|
| `standard` | box_corner + rich_label |
| `lightweight` | dot + rich_label |
| `privacy` | box + blur |
| `debug` | round_box + percentage_bar + rich_label |
| `high_contrast` | halo + background_overlay |

支持的 annotator 类型:`box`、`round_box`、`box_corner`、`circle`、`triangle`、`ellipse`、`dot`、`color`、`background_overlay`、`halo`、`percentage_bar`、`blur`、`pixelate`。

## 加载示例

```python
import yaml
from onnxtools import create_detector, OcrORT
from onnxtools.utils import load_visualization_preset

with open('configs/det_config.yaml') as f:
    det = yaml.safe_load(f)
with open('configs/plate.yaml') as f:
    plate = yaml.safe_load(f)

detector = create_detector('rtdetr', 'models/rtdetr.onnx')
ocr = OcrORT('models/ocr.onnx', character=plate['ocr_dict'])
annotators = load_visualization_preset('debug')

result = detector(image)
names = [det['class_names'][i] for i in result.class_ids]
detections = result.to_supervision()
for a in annotators:
    image = a.annotate(image, detections)
```

命令行加载预设:`python examples/demo_pipeline.py --annotator-preset debug --input test.jpg`

## 校验

- 类别名列表长度 == `visual_colors` 长度;颜色格式 `#RRGGBB`。
- `ocr_dict` 字符唯一、无空字符;颜色/层级映射键为整数索引。
- 快速语法检查:`python -c "import yaml; yaml.safe_load(open('configs/det_config.yaml'))"`
- 校验测试建议放在 `tests/unit/test_configs.py`。

## 依赖

`pyyaml>=6.0.2`
