# 调试和优化工具 (tools/)

项目调试、评估和优化工具集合。

## 目录结构

```
tools/
├── eval/                      # 数据集评估（onnxtools.eval 衍生）
│   ├── eval.py                # COCO 数据集检测评估
│   ├── eval_ocr.py            # OCR 数据集评估（支持错误分析）
│   ├── eval_cls.py            # 颜色/层级分类模型评估
│   └── eval_mot.py            # MOT 跟踪评估（HOTA/MOTA/IDF1）
├── trt/                       # TensorRT 引擎构建与分析（polygraphy/trex）
│   ├── build_engine.py        # ONNX → TensorRT 引擎构建
│   ├── compare_onnx_engine.py # ONNX vs TensorRT 性能/精度对比
│   ├── draw_engine.py         # TensorRT 引擎结构可视化
│   ├── layer_statistics.py    # 模型层统计分析
│   ├── network_postprocess.py # TensorRT 网络精度优化
│   └── tensor_selector.py     # 张量选择和分析
├── onnx/                      # ONNX 图结构改造（onnx-graphsurgeon）
│   ├── modify_onnx_io_names.py # ONNX 输入/输出张量重命名
│   └── modify_rfdetr.py       # RF-DETR ONNX 结构改造
├── tracking/                  # 跟踪器调参（onnxtools.tracking 衍生）
│   ├── measure_class_voting.py
│   ├── sweep_bytetrack.py
│   ├── sweep_iou_age.py
│   └── sweep_q.py
├── scripts/                   # Shell 脚本
│   ├── build.sh               # ONNX 优化 + TensorRT 引擎构建
│   ├── eval.sh                # 模型评估快捷脚本
│   ├── third_party.sh         # 第三方库初始化
│   └── rsync_export.sh        # 交互式清洁同步/迁移脚本
├── debug/                     # Polygraphy 调试脚本
│   ├── 01_debug_subonnx_fp16.sh
│   ├── 01_debug_subonnx_fp32.sh
│   ├── 02_debug_subonnx_fp32.sh
│   ├── debug_fp16.sh
│   └── data_loader.py.template
├── CLAUDE.md                  # 模块文档
└── README.md                  # 本文件
```

## 常用命令

### 模型评估

#### COCO 数据集评估
用于评估检测模型（YOLO, RT-DETR 等）在标准数据集上的 mAP。
```bash
python tools/eval/eval.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25
```

#### OCR 数据集评估
评估车牌 OCR 模型的准确率，支持详细的错误 analysis 和多种输出格式。
```bash
python tools/eval/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --error-analysis error_report.json
```

#### 分类模型评估
评估颜色分类和层级分类模型的精度。
```bash
python tools/eval/eval_cls.py \
    --model-path models/color_layer.onnx \
    --dataset-path data/cls_test/
```

### TensorRT 引擎构建

#### 使用 Python 工具构建
提供细粒度的构建选项，包括 FP16/INT8 精度切换。
```bash
python tools/trt/build_engine.py \
    --onnx-path models/rtdetr.onnx \
    --engine-path models/rtdetr.engine \
    --fp16
```

#### 使用快捷脚本构建
集成 ONNX 优化和 TensorRT 构建的自动化脚本。
```bash
bash tools/scripts/build.sh models/rtdetr.onnx models/rtdetr.engine
```

### 性能分析

#### ONNX vs TensorRT 对比
自动运行推理并比较两者的延迟、吞吐量和输出一致性。
```bash
python tools/trt/compare_onnx_engine.py \
    --onnx models/rtdetr.onnx \
    --engine models/rtdetr.engine
```

#### 模型层统计分析
详细分析 ONNX 和 TensorRT 网络中的每一层，用于排查性能瓶颈。
```bash
python tools/trt/layer_statistics.py --model models/rtdetr.onnx --build-trt
```

### ONNX 模型工具

#### 修改输入输出名称
在模型部署前重命名 I/O 张量，确保与推理后端匹配。
```bash
python tools/onnx/modify_onnx_io_names.py \
    --input models/model.onnx \
    --output models/model_fixed.onnx \
    --new-names "images:0,output:0"
```

### 第三方库初始化
快速配置项目所需的第三方子模块（如 Polygraphy, Ultralytics 等）。
```bash
bash tools/scripts/third_party.sh
```

### 清洁同步/迁移
使用 rsync 交互式同步项目目录到本地目录或远程服务器，默认排除 `.git`、缓存、
虚拟环境、构建产物、运行输出、模型和数据目录。远程模式会询问 Host/IP、用户名、
SSH 端口、私钥路径和目标目录，自动创建远程目录，然后直接执行同步。
```bash
bash tools/scripts/rsync_export.sh
```
