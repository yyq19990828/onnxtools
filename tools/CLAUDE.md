[根目录](../CLAUDE.md) > **tools**

# 调试工具模块 (tools)

## 模块职责

模型评估（COCO / OCR / 分类 / MOT）、TensorRT 引擎构建、ONNX vs TRT 性能对比，以及 Polygraphy 子图调试。引擎构建/对比依赖 `tensorrt` + `polygraphy`（可选 `[trt]` extra）。

## 脚本清单

Python 工具按衍生模块分目录归类。

### eval/ — 数据集评估（onnxtools.eval 衍生）
- `eval/eval.py` — COCO 数据集检测评估主入口（mAP/mAP50/75）。
- `eval/eval_ocr.py` — OCR 数据集评估，支持深度错误分析与 JSON 导出。
- `eval/eval_cls.py` — 颜色/层数分类模型评估。
- `eval/eval_mot.py` — MOT 跟踪评估（HOTA/MOTA/IDF1，MOTChallenge 格式）。

### trt/ — TensorRT 引擎构建与分析（polygraphy/tensorrt/trex）
- `trt/build_engine.py` — ONNX 转 TensorRT 引擎构建。
- `trt/compare_onnx_engine.py` — ONNX vs TensorRT 延迟/吞吐/精度对比。
- `trt/draw_engine.py` — TensorRT 引擎结构可视化。
- `trt/layer_statistics.py` — 模型层统计分析。
- `trt/tensor_selector.py` — 张量选择与分析。
- `trt/network_postprocess.py` — TensorRT 网络后处理/精度优化。

### onnx/ — ONNX 图结构改造（onnx-graphsurgeon）
- `onnx/modify_onnx_io_names.py` — ONNX 输入/输出张量重命名。
- `onnx/modify_rfdetr.py` — RF-DETR ONNX 结构调整。

### tracking/ — 跟踪器调参（onnxtools.tracking 衍生）
- `tracking/measure_class_voting.py`、`tracking/sweep_{bytetrack,iou_age,q}.py` — ByteTrack/卡尔曼参数 sweep。

### Shell 脚本 (scripts/)
- `scripts/build.sh` — 批量引擎构建。
- `scripts/eval.sh` — 批量评估。
- `scripts/third_party.sh` — 第三方库初始化。
- `scripts/rsync_export.sh` — 交互式 rsync 迁移（本地/远程，默认排除 git 元数据、缓存、构建产物）。

### 调试脚本 (debug/)
- `debug/01_debug_subonnx_fp16.sh` / `debug/01_debug_subonnx_fp32.sh` / `debug/02_debug_subonnx_fp32.sh` — Polygraphy 子图精度调试。
- `debug/debug_fp16.sh` — FP16 精度调试。
- `debug/data_loader.py.template` — 校准/调试数据加载器模板。

## 示例命令

### COCO 检测评估
```bash
python tools/eval/eval.py \
    --model-path models/rtdetr-2024080100.onnx --model-type rtdetr \
    --dataset-path /path/to/coco --annotations-path /path/to/annotations.json
```

### OCR 评估
```bash
python tools/eval/eval_ocr.py \
    --label-file data/val.txt --dataset-base data/ \
    --ocr-model models/ocr.onnx --config configs/plate.yaml \
    --conf-threshold 0.5
# 错误分析: --error-analysis report.json ；JSON 导出: --output-format json
```

### 分类评估
```bash
python tools/eval/eval_cls.py --model models/color_layer.onnx --config configs/plate.yaml --dataset-base data/cls/
```

### MOT 跟踪评估
```bash
# 模式 A: 用 GT 框作理想检测，现场跑某跟踪后端评估关联质量（无需检测器）
python tools/eval/eval_mot.py --gt-root data/track/MOT_dataset --tracker bytetrack_native --frame-rate 5

# 模式 B: 评估已有跟踪结果目录（每序列 <seq>.txt，MOTChallenge 格式）
python tools/eval/eval_mot.py --gt-root data/track/MOT_dataset \
    --predictions runs/tracker_out --metrics hota identity --output runs/mot_eval/result.json
```

### TensorRT 引擎构建
```bash
python tools/trt/build_engine.py \
    --onnx-path models/rtdetr-2024080100.onnx --engine-path models/rtdetr-2024080100.engine \
    --precision fp16 --max-batch-size 8
```

### ONNX vs TensorRT 性能对比
```bash
python tools/trt/compare_onnx_engine.py \
    --onnx-path models/x.onnx --engine-path models/x.engine \
    --input-shape 1,3,640,640 --iterations 100
```

### 引擎结构可视化
```bash
python tools/trt/draw_engine.py --engine-path models/x.engine
```

### 清洁同步/迁移
```bash
bash tools/scripts/rsync_export.sh   # 交互式选源目录 + 本地/远程目标
```

## 数据模型

- 检测评估输出: `mAP / mAP_50 / mAP_75 / mAP_{small,medium,large} / per_class_ap / inference_time / total_images`。
- 性能对比输出: `onnx_runtime` 与 `tensorrt_runtime` 各含 `mean_latency / std_latency / throughput / memory_usage`，外加 `speedup_ratio`。
- 引擎构建配置: `precision('fp32'|'fp16'|'int8') / max_batch_size / max_workspace_size / input_shapes / optimization_level`。

## FAQ

- **TRT 引擎构建失败**：检查 ONNX 兼容性与 TensorRT 版本，调大工作空间，用 Polygraphy 定位。
- **评估精度异常**：核对标注格式、预处理一致性、类别映射；对比少量样本推理结果。
- **Polygraphy 深度调试**：见 `../docs/polygraphy使用指南/`。
