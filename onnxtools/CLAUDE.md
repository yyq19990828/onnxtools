[根目录](../CLAUDE.md) > **onnxtools**

# onnxtools 核心模块

项目的 Python 包核心，提供统一的 ONNX 推理接口、多架构检测器、OCR 识别、分类器和数据集评估。整合推理引擎 (infer_onnx)、工具函数 (utils) 和评估工具 (eval)，对外暴露简洁 API。

- 根模块入口：`onnxtools/__init__.py`
- 推理管道：`pipeline.py` 中的 `InferencePipeline`（推荐接口）
- 演示脚本：`examples/demo_pipeline.py`

## 核心入口

三类推理对象，返回类型各不相同：
- **检测器**（继承 `BaseORT`）：目标检测，返回 `Result`
- **分类器**（继承 `BaseClsORT`）：分类，返回 `ClsResult`（也支持元组解包）
- **OCR**（独立 `OcrORT`）：序列识别，返回 `Optional[Tuple]`

```python
from onnxtools import (
    InferencePipeline, create_detector, setup_logger,
    BaseORT, YoloORT, RtdetrORT, RfdetrORT, Result,        # 检测
    BaseClsORT, ClsResult, ColorLayerORT, VehicleAttributeORT,  # 分类
    OcrORT,                                                 # OCR
    DetDatasetEvaluator, OCRDatasetEvaluator, SampleEvaluation,  # 评估
)

# 完整管道（检测 + OCR + 分类 + 可视化），开箱即用
pipeline = InferencePipeline(
    model_type='rtdetr', model_path='models/rtdetr.onnx',
    ocr_model_path='models/ocr.onnx', color_model_path='models/color_layer.onnx',
    conf_thres=0.5, annotator_preset='standard',
)
result_img, output_data = pipeline(image)  # (标注图像, 数据字典)

# 仅创建检测器（灵活，适合自定义流程/研究）
detector = create_detector('rtdetr', 'models/rtdetr.onnx', conf_thres=0.5)
result = detector(image)  # Result 对象

# 分类器
color, layer, conf = ColorLayerORT('models/color_layer.onnx')(plate_image)
vtype, vcolor, conf = VehicleAttributeORT('models/vehicle_attribute.onnx')(vehicle_image)

# OCR
ocr_result = OcrORT('models/ocr.onnx')(plate_image)
if ocr_result:
    text, conf, char_scores = ocr_result

# 评估
DetDatasetEvaluator(detector).evaluate_dataset(dataset_path)
OCRDatasetEvaluator(ocr_model).evaluate_dataset(label_file, dataset_base)
```

`InferencePipeline.__init__` 关键参数：`model_type`('yolo'/'rtdetr'/'rfdetr')、`model_path`、可选的 `ocr_model_path`/`color_model_path`/`vehicle_attr_model_path`、`config_path`、`plate_config_path`、`conf_thres`、`iou_thres`、`annotator_preset`('standard'/'debug'/'lightweight'/'privacy'/'high_contrast')。

新模型架构扩展：在 `infer_onnx/` 创建推理类继承 `BaseORT` 或 `BaseClsORT`，实现抽象方法，在 `__init__.py` 导出（检测器还需在 `create_detector()` 注册）。

## 数据模型

```python
# Result（检测器）
result.boxes        # np.ndarray [N,4] xyxy
result.scores       # np.ndarray [N]
result.class_ids    # np.ndarray [N]
result.orig_shape   # Tuple[int,int]
# 富 API：len/索引/切片/迭代、filter()、summary()、plot()/show()/save()

# ClsResult（分类器）
result.labels           # List[str]
result.confidences      # List[float]
result.avg_confidence   # float
# 向后兼容：color, layer, conf = classifier(img)

# OCR：Optional[Tuple] -> (text: str, confidence: float, char_scores: List[float])
```

完整 Result API 见 [infer_onnx/CLAUDE.md](infer_onnx/CLAUDE.md)。

## 模块结构

```
onnxtools/
├── __init__.py        # 根模块，导出公共 API
├── pipeline.py        # InferencePipeline
├── config.py          # 配置管理 (VEHICLE_TYPE_MAP, VEHICLE_COLOR_MAP)
├── infer_onnx/        # 推理引擎：BaseORT/检测器、BaseClsORT/分类器、OcrORT、Result/ClsResult
│   ├── onnx_base.py   # BaseORT 抽象基类
│   ├── onnx_yolo.py · onnx_rtdetr.py · onnx_rfdetr.py  # 检测器
│   ├── onnx_cls.py    # BaseClsORT, ClsResult, ColorLayerORT, VehicleAttributeORT
│   ├── onnx_ocr.py    # OcrORT
│   ├── result.py      # Result
│   └── CLAUDE.md
├── eval/              # 数据集评估：DetDatasetEvaluator (COCO)、OCRDatasetEvaluator
│   └── CLAUDE.md
└── utils/             # 图像处理、Supervision 可视化(annotator/preset/labels)、NMS、OCR 指标、日志、字体
    └── CLAUDE.md
```

## 依赖与配置

- 核心：`onnxruntime-gpu==1.22.0`、`opencv-contrib-python>=4.12.0.88`、`numpy>=2.2.6`、`supervision==0.26.1`、`pyyaml`、`python-levenshtein`、`colorlog`、`pillow`
- 可选：`[trt]` extra（`tensorrt==8.6.1.post1` 等，自动识别 `.engine` 文件，无需改代码）
- 配置文件：`configs/det_config.yaml`（检测类别/颜色）、`configs/plate.yaml`（OCR 字典/映射）、`configs/visualization_presets.yaml`（可视化预设）

## 检测器选型

- 实时优先：YOLO 系列
- 精度优先：RF-DETR
- 平衡：RT-DETR
