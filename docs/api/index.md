# API 参考

下列页面由 [`mkdocstrings`](https://mkdocstrings.github.io/) 从源码 docstring 自动生成,与代码同步更新。

## 顶层导出

```python
from onnxtools import (
    create_detector,
    InferencePipeline,
    BaseORT, YoloORT, RtdetrORT, RfdetrORT,
    BaseClsORT, ColorLayerORT, VehicleAttributeORT, ClsResult,
    OcrORT,
    Result,
    DetDatasetEvaluator, OCRDatasetEvaluator,
    setup_logger,
)
```

## 分类导航

| 区块 | 内容 |
|---|---|
| [Detectors](detectors.md) | `BaseORT` 与三种检测器 (`YoloORT` / `RtdetrORT` / `RfdetrORT`) |
| [Classifiers](classifiers.md) | `BaseClsORT` / `ColorLayerORT` / `VehicleAttributeORT` / `ClsResult` |
| [OCR](ocr.md) | `OcrORT` 序列识别 |
| [Result](result.md) | 检测结果包装类 |
| [Pipeline](pipeline.md) | 端到端 `InferencePipeline` |
| [Utils](utils.md) | 图像预处理、可视化、NMS、OCR 指标等工具 |
| [Eval](eval.md) | COCO / OCR / 分类数据集评估器 |
| [Config](config.md) | 全局配置加载(检测类别、OCR 字典、可视化预设) |
