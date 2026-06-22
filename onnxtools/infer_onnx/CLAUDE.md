[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **infer_onnx**

# 推理引擎模块 (onnxtools.infer_onnx)

## 模块职责

核心 ONNX 推理引擎。提供三类统一推理接口：检测器（YOLO / RT-DETR / RF-DETR）、分类器（车牌颜色/层级、车辆属性、头盔）、OCR 序列识别。检测器走 `BaseORT` 模板方法，分类器走 `BaseClsORT` 模板方法，OCR 为独立类。

## 三类推理架构

```
BaseORT (检测, 抽象)         → Result
├── YoloORT / RtdetrORT / RfdetrORT
└── RfdetrUnifiedORT (实验性, experiment.py)

BaseClsORT (分类, 抽象)      → ClsResult
├── ColorLayerORT (颜色+层级双分支)
├── VehicleAttributeORT (车型+颜色)
└── HelmetORT

OcrORT (独立, 序列识别)      → Optional[(text, conf, char_scores)]
```

## 核心契约 / 不变量

- **`__call__` 由基类实现，子类禁止重写**（模板方法）。检测器 `__call__(image, conf_thres=None, **kwargs) -> Result`。
- **`BaseORT` 子类必须实现两个静态抽象方法**（注意是 `@staticmethod @abstractmethod`）：
  - `preprocess(image, input_shape, **kwargs) -> tuple` （至少返回 `(input_tensor, scale, original_shape)`）
  - `postprocess(prediction, input_shape, conf_thres, **kwargs) -> list[np.ndarray]`
- **`BaseClsORT` 子类必须实现**：
  - `preprocess(image, input_shape, **kwargs)`（静态）
  - `postprocess(outputs, conf_thres, **kwargs) -> ClsResult`
- 未实现时基类抛 `NotImplementedError`，不静默回退。
- **检测器只能通过根模块 `create_detector(model_type, onnx_path, **kwargs)` 创建**，不要直接实例化。`model_type`: `yolo`/`yolov5`/`yolov8`/`yolov11`、`rtdetr`/`rt-detr`、`rfdetr`/`rf-detr`、`rfdetr_unified`/`rfdetr-unified`。
- Polygraphy 懒加载：`InferenceSession` 在 `__init__` 创建并缓存；`.engine` TensorRT 引擎仅在检测到时按需加载，勿在 `__init__` 中急加载。

## 模块结构

```
infer_onnx/
├── onnx_base.py         BaseORT 抽象基类 (检测), 提供 __call__
├── onnx_yolo.py         YoloORT
├── onnx_rtdetr.py       RtdetrORT
├── onnx_rfdetr.py       RfdetrORT
├── experiment.py        RfdetrUnifiedORT (实验性统一检测)
├── onnx_cls.py          BaseClsORT, ClsResult, ColorLayerORT, VehicleAttributeORT, HelmetORT
├── onnx_ocr.py          OcrORT (独立序列识别)
├── result.py            Result 检测结果类
├── infer_utils.py       推理辅助函数
├── engine_dataloader.py TensorRT 引擎数据加载
└── __init__.py          懒加载导出 (__getattr__)
```

## 数据类

### Result（检测）
字段：`boxes [N,4] xyxy`、`scores [N]`、`class_ids [N]`、`orig_shape (H,W)`、`names {int:str}`、`path`、`orig_img`。新字段须向后兼容（默认值或 Optional）。
方法：`len()`、索引/切片/迭代（返回子集 Result）、`plot(annotator_preset=...)`、`show()`、`save(path)`、`filter(conf_threshold, classes)`、`summary()`、`to_supervision()`。
可视化预设：`standard` / `debug` / `lightweight` / `privacy`。

### ClsResult（分类，定义于 onnx_cls.py）
字段：`labels: List[str]`、`confidences: List[float]`、`avg_confidence: float`、`logits: Optional`。
支持 `len()`、`result[i] -> (label, conf)`、迭代，以及向后兼容的元组解包：
- 双分支：`color, layer, conf = result`
- 单分支：`label, conf = result`

### OCR 输出
`Optional[(text: str, confidence: float, char_scores: List[float])]`，识别失败返回 `None`。

## 用法速览

```python
from onnxtools import create_detector, ColorLayerORT, OcrORT

# 检测
det = create_detector('rtdetr', 'models/rtdetr.onnx', conf_thres=0.5, iou_thres=0.5)
result = det(image)                 # Result
result.save('out.jpg', annotator_preset='debug')
for d in result.filter(conf_threshold=0.8):
    print(d.boxes[0], d.scores[0], d.names[d.class_ids[0]])

# 分类
cls = ColorLayerORT('models/color_layer.onnx')
color, layer, conf = cls(plate_img)  # 或 r=cls(plate_img); r.labels[0]

# OCR
ocr = OcrORT('models/ocr.onnx')
r = ocr(plate_img, is_double_layer=True)
if r:
    text, conf, char_scores = r
```

## 模型 I/O 要求

| 类型 | 输入 | 输出 |
|------|------|------|
| YOLO | [1,3,640,640] | [1,N,85] (x,y,w,h,conf,classes) |
| RT-DETR / RF-DETR | [1,3,640,640] | [1,N,6] (x1,y1,x2,y2,score,cls) |
| OCR | [1,3,48,168] | [1,T,C] |
| ColorLayer | [1,3,48,168] | [[1,5],[1,2]] 颜色+层级 |
| VehicleAttribute | [1,3,224,224] | [1,24] 车型(13)+颜色(11) |

## 配置

- `configs/det_config.yaml`：检测类别名称与可视化颜色。
- `configs/plate.yaml`：OCR 字典与颜色/层级映射。
- 默认值内置于代码，外部 YAML 优先。

## 新增模型架构

1. 检测：在 `onnx_<name>.py` 继承 `BaseORT`，实现静态 `preprocess` / `postprocess` 返回 `list[np.ndarray]`（基类再包成 `Result`）。
2. 分类：继承 `BaseClsORT`，实现 `preprocess` / `postprocess` 返回 `ClsResult`。
3. 在 `__init__.py` 的 `_LAZY_EXPORTS` + `__all__` 中导出；检测器还需在根模块 `create_detector()` 注册新分支。
4. 补测试：`tests/unit/test_<name>.py` + `tests/integration/test_<name>_pipeline.py`。

## 常见问题

- **OcrORT 为何独立？** 序列识别任务，返回可变长字符序列 + 字符级置信度，预处理含双层车牌/倾斜校正等特殊逻辑，不套用分类模板。
- **分类返回 ClsResult 还是元组？** 返回 `ClsResult`，同时支持元组解包以向后兼容。
