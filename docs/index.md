# onnxtools

基于 ONNX 的车辆与车牌识别工具包,统一封装 YOLO / RT-DETR / RF-DETR 检测、OCR 识别、颜色/层级/属性分类与可视化能力。

## 核心特性

- **多检测架构统一接口**:`YoloORT` / `RtdetrORT` / `RfdetrORT` 共享 `BaseORT` 模板方法,均返回 [`Result`](api/result.md) 对象
- **完整识别管道**:[`InferencePipeline`](api/pipeline.md) 一站式串接检测 + OCR + 颜色/属性分类 + 可视化
- **统一分类抽象**:`BaseClsORT` + `ClsResult`,覆盖车牌颜色层级、车辆属性多标签
- **Supervision 可视化**:13 种 annotator + 5 套预设(`standard` / `debug` / `lightweight` / `privacy` / `high_contrast`)
- **TensorRT 加速**:`tools/trt/build_engine.py` 一键构建 FP16 引擎,推理代码无需改动
- **Polygraphy 调试集成**:ONNX↔TensorRT 输出对齐校验

## 快速链接

- [快速上手](getting-started.md) — 5 分钟跑通第一张图
- [API 参考](api/index.md) — 从源码 docstring 自动生成
- [可视化标注指南](guides/annotator_usage.md)
- [模型评估指南](guides/evaluation_guide.md)
- [模型支持列表](models/model_support_list.md)

## 最小示例

```python
from onnxtools import create_detector
import cv2

detector = create_detector('rtdetr', 'models/rtdetr.onnx', conf_thres=0.5)
result = detector(cv2.imread('test.jpg'))
print(result)                                    # Result(N detections, K classes)
result.save('out.jpg', annotator_preset='debug')
```
