# Detectors

所有检测器继承自 [`BaseORT`](#onnxtools.infer_onnx.onnx_base.BaseORT),
通过模板方法模式串联预处理 → ONNX 推理 → 后处理 → [`Result`](result.md) 包装。

## 工厂函数

::: onnxtools.create_detector
    options:
      heading_level: 3

## BaseORT

::: onnxtools.infer_onnx.onnx_base.BaseORT
    options:
      heading_level: 3
      show_root_heading: true
      members_order: source

## YoloORT

::: onnxtools.infer_onnx.onnx_yolo.YoloORT
    options:
      heading_level: 3
      members_order: source

## RtdetrORT

::: onnxtools.infer_onnx.onnx_rtdetr.RtdetrORT
    options:
      heading_level: 3
      members_order: source

## RfdetrORT

::: onnxtools.infer_onnx.onnx_rfdetr.RfdetrORT
    options:
      heading_level: 3
      members_order: source

## RfdetrUnifiedORT (实验性)

通过 ``create_detector('rfdetr_unified', ...)`` 创建。

::: onnxtools.infer_onnx.experiment.RfdetrUnifiedORT
    options:
      heading_level: 3
      members_order: source
