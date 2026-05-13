# 快速上手

## 安装

```bash
# 推荐使用 uv
uv sync

# 验证安装
python -c "import onnxtools; print('OK')"
```

TensorRT 加速(可选,需要本地 NVIDIA 环境):

```bash
uv pip install pip setuptools wheel
uv pip install -e ".[trt]"
```

## 5 分钟体验

### 方式 1:工厂函数 + Result 对象(推荐研究/自定义流程)

```python
from onnxtools import create_detector
import cv2

detector = create_detector(
    model_type='rtdetr',          # 'yolo' / 'rtdetr' / 'rfdetr'
    onnx_path='models/rtdetr.onnx',
    conf_thres=0.5,
    iou_thres=0.7,
)

image = cv2.imread('data/sample.jpg')
result = detector(image)                          # 返回 Result 对象

print(result)                                     # Result(10 detections, 2 classes)
print(result.boxes.shape, result.scores.shape)    # (10, 4) (10,)

# 过滤 + 可视化
high_conf = result.filter(conf_threshold=0.8)
high_conf.save('out.jpg', annotator_preset='debug')
```

### 方式 2:完整推理管道(推荐应用开发)

```python
from onnxtools import InferencePipeline
import cv2

pipeline = InferencePipeline(
    model_type='rtdetr',
    model_path='models/rtdetr.onnx',
    ocr_model_path='models/ocr.onnx',
    color_model_path='models/color_layer.onnx',
    conf_thres=0.5,
    annotator_preset='debug',
)

annotated, data = pipeline(cv2.imread('data/sample.jpg'))
cv2.imwrite('annotated.jpg', annotated)
```

## 命令行示例

```bash
# 单图推理
python examples/demo_pipeline.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show

# 视频推理(跳帧)
python examples/demo_pipeline.py \
    --model-path models/yolo11n.onnx \
    --model-type yolo \
    --input video.mp4 \
    --source-type video \
    --output-mode save \
    --frame-skip 2
```

## 下一步

- 检测器/分类器/OCR 完整 API:[API 参考](api/index.md)
- 自定义可视化:[可视化标注指南](guides/annotator_usage.md)
- 模型精度评估:[模型评估指南](guides/evaluation_guide.md)
