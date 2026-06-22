# CLI 工具

`tools/` 下的命令行脚本用于评估、TensorRT 引擎构建与 ONNX↔TRT 对比。

## 模型评估

### 检测(COCO 格式)

```bash
python tools/eval/eval.py \
    --model-type rtdetr \
    --model-path models/rtdetr.onnx \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25 \
    --iou-threshold 0.7
```

输出 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall 等。详见
[模型评估指南](evaluation_guide.md)。

### OCR

```bash
python tools/eval/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5
```

输出完全匹配率、归一化编辑距离、编辑距离相似度。

## TensorRT 引擎构建

```bash
# FP16 引擎
python tools/trt/build_engine.py \
    --onnx-path models/rtdetr.onnx \
    --engine-path models/rtdetr_fp16.engine \
    --fp16

# 构建后自动与 ONNX 对齐校验
python tools/trt/build_engine.py \
    --onnx-path models/yolov8s_640.onnx \
    --compare
```

## ONNX ↔ TensorRT 对比

```bash
python tools/trt/compare_onnx_engine.py \
    --onnx  models/yolov8s_640.onnx \
    --engine models/yolov8s_640.engine
```

底层基于 [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy),
输出层级 cosine 相似度与最大/平均误差。仓库内 `docs/polygraphy使用指南/`
有完整中文翻译(未纳入本站)。

## Shell 包装脚本

`tools/scripts/` 下有三个便捷脚本:

| 脚本 | 用途 |
|---|---|
| `build.sh` | ONNX 优化 + TensorRT 构建一站式 |
| `eval.sh` | 模型评估 |
| `third_party.sh` | 第三方库(Ultralytics / RF-DETR)初始化 |

## 其他工具

| 脚本 | 用途 |
|---|---|
| `tools/trt/draw_engine.py` | 可视化 TensorRT engine 计算图 |
| `tools/trt/layer_statistics.py` | 各 layer 的耗时、精度统计 |

每个脚本都支持 `--help`,可直接查看完整参数。
