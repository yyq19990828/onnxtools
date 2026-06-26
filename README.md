# onnxtools — ONNX 车辆与车牌识别工具集

> 基于 ONNX 的高性能车辆 / 车牌识别工具集。支持图像、视频、摄像头输入，提供车辆与车牌检测、车牌 OCR、颜色 / 层级分类，以及 2D 多目标跟踪。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime--gpu-1.22.0-green.svg)](https://onnxruntime.ai/)
[![Supervision](https://img.shields.io/badge/supervision-0.26.1-orange.svg)](https://supervision.roboflow.com/)

## 核心特性

- **多检测架构**：YOLO v8/v11、RT-DETR、RF-DETR，通过统一工厂函数创建。
- **全流程识别**：车辆 / 车牌检测 → 车牌 OCR → 颜色（蓝/黄/白/黑/绿）与层级（单/双层）分类。
- **2D 多目标跟踪**：内置 ByteTrack，为视频 / 摄像头中的目标分配持久 ID。
- **专业可视化**：集成 Supervision，提供 7 种预设和 15 种 annotator。
- **多源输入**：图像、文件夹、视频文件、摄像头、RTSP（输入源自动识别）。
- **可选 TensorRT 加速**：检测到 `.engine` 时按需懒加载，推理提速 2–5 倍。
- **框架无关**：纯 ONNX Runtime 本地运行，不依赖特定训练框架。

## 安装

```bash
git clone https://github.com/your-username/onnxtools.git
cd onnxtools

# 方式 1：uv（推荐，完整推理环境）
uv sync --extra inference

# 方式 2：pip（完整推理环境）
python -m venv .venv && source .venv/bin/activate
pip install -e ".[inference]"
```

**环境要求**：Python ≥ 3.10。基础安装只包含通用轻依赖；ONNX 推理链路使用 `[inference]` extra，主要依赖 `onnxruntime-gpu==1.22.0`、`supervision==0.26.1`、`opencv-contrib-python>=4.12.0`、`numpy>=2.2.6`、`pyyaml>=6.0.2`。

### 纯 Tracking 安装

如果只使用 `onnxtools.tracking`，无需安装 ONNX / ONNX Runtime / Polygraphy：

```bash
uv pip install -e ".[tracking]"

# 可选：使用 lap.lapjv 加速匹配
uv pip install -e ".[tracking-fast]"

# 可选：BoT-SORT 相机运动补偿(camera_motion=True)所需 OpenCV
uv pip install -e ".[tracking-cmc]"
```

可用后端:`bytetrack`、`bytetrack_native`、`ocsort`、`botsort`。`botsort` 的 ReID 模型由调用方外部提供 embedding,不会强制安装 PyTorch / ONNX Runtime。

### MOT 评估安装

评估 MOTChallenge 格式跟踪结果（HOTA / MOTA / IDF1）需要额外的评估库，同样无需 ONNX：

```bash
uv pip install -e ".[mot]"   # motmetrics(CLEAR/IDF1) + trackeval(HOTA) + supervision
```

### TensorRT 加速（可选）

TensorRT 为可选 extra `[trt]`，依赖本地 NVIDIA GPU 及 NVIDIA PyPI 源。安装前需在 `pyproject.toml` 中取消注释 `[tool.uv]` 的 `extra-index-url` 与 `[project.optional-dependencies].trt` 三个包：

```bash
uv pip install pip setuptools wheel
uv pip install -e ".[trt]"
```

> 远程 / 云开发环境（如 Claude Code on the web）通常无法访问 NVIDIA PyPI 源，TensorRT 功能不可用。详见 [TensorRT 官方安装指南](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

## 快速开始

主脚本为 `examples/demo_pipeline.py`，输入源类型由 `--input` 自动识别。

```bash
# 单张图片（RT-DETR，保存结果）
python examples/demo_pipeline.py --model-path models/rtdetr-2024080100.onnx --input data/sample.jpg

# 文件夹批量处理 + debug 可视化预设
python examples/demo_pipeline.py --model-path models/rtdetr-2024080100.onnx \
    --model-type rtdetr --input data/苏州图片 --annotator-preset debug

# 视频 + 实时显示 + 开启跟踪
python examples/demo_pipeline.py --model-path models/yolov8s_640.onnx --model-type yolo \
    --input video.mp4 --output-mode show --enable-tracking

# 摄像头（YOLO11，跳帧）
python examples/demo_pipeline.py --model-path models/yolo11n.onnx --model-type yolo \
    --input 0 --output-mode show --frame-skip 2

# 也可直接运行预配置脚本
./run.sh
```

### 常用命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 检测 ONNX 模型路径（必需） | — |
| `--model-type` | `rtdetr` / `yolo` / `rfdetr` / `rfdetr_unified` | `rtdetr` |
| `--input` | 图像 / 文件夹 / 视频路径或摄像头 ID（如 `0`） | `data/sample.jpg` |
| `--output-mode` | `save` 保存 / `show` 窗口显示 | `save` |
| `--output-dir` | 结果输出目录 | `runs` |
| `--conf-thres` / `--iou-thres` | 检测置信度 / NMS IoU 阈值 | `0.5` / `0.5` |
| `--plate-conf-thres` | 车牌专用置信度（默认同 `--conf-thres`） | `None` |
| `--roi-top-ratio` | 检测 ROI 顶部比例 [0–1]，0.5 表示仅检测下半幅 | `0.5` |
| `--det-config` | `coco80` 或检测配置 YAML 路径 | `None` |
| `--ocr-model` / `--color-layer-model` | OCR / 颜色层级模型路径 | 见 `--help` |
| `--annotator-preset` | `standard`/`debug`/`lightweight`/`privacy`/`high_contrast`/`box_only`/`tracking` | `standard` |
| `--frame-skip` | 视频跳帧数 | `0` |
| `--save-frame` / `--save-json` | （视频）逐帧保存图像 / JSON | off |
| `--enable-tracking` | 开启 2D 跟踪（视频 / 摄像头） | off |

完整参数见 `python examples/demo_pipeline.py --help`。

### Python API

```python
import cv2
from onnxtools import create_detector, setup_logger

setup_logger("INFO")

# 工厂函数是唯一创建入口
detector = create_detector(
    model_type="rtdetr",  # yolo / rtdetr / rfdetr / rfdetr_unified
    onnx_path="models/rtdetr-2024080100.onnx",
    conf_thres=0.5,
    iou_thres=0.5,
)

image = cv2.imread("data/sample.jpg")
result = detector(image)        # 返回统一的 Result 对象
boxes = result.boxes            # [N, 4] xyxy
scores = result.scores          # [N]
class_ids = result.class_ids    # [N]
```

车牌 OCR：

```python
import yaml
from onnxtools import OcrORT

config = yaml.safe_load(open("configs/plate.yaml"))
ocr = OcrORT(
    onnx_path="models/ocr.onnx",
    character=config["plate_dict"]["character"],
    conf_thres=0.7,
)

x1, y1, x2, y2 = boxes[0].astype(int)
out = ocr(image[y1:y2, x1:x2])
if out:
    text, conf, char_scores = out
    print(f"{text}  ({conf:.2f})")
```

### 车辆属性二阶段预标

`VehicleAttributePipeline` 串联检测与车辆属性分类：检测框出目标 → 对机动车
（car / truck / heavy_truck / van / bus / motorcycle）裁剪 ROI → 用车辆属性模型得到
车型（13 类）与颜色（11 类），写入该框属性。非机动车只输出几何，不做二次推理。

```python
import cv2
from onnxtools import VehicleAttributePipeline, setup_logger

setup_logger("INFO")

pipeline = VehicleAttributePipeline(
    model_type="rtdetr",
    model_path="models/rtdetr-2024080100.onnx",
    va_model_path="models/va_260612.onnx",
    conf_thres=0.5,
)

output = pipeline(cv2.imread("data/sample.jpg"))
for item in output:
    if "vehicle_type" in item:        # 机动车
        print(item["type"], item["vehicle_type"], item["color"])
```

命令行：`python examples/demo_vehicle_attribute.py --input data/sample.jpg`。

## 模型与配置

模型放置在 `models/` 目录，需要三类 ONNX：

| 模型 | 作用 | 输入尺寸 |
|------|------|---------|
| 检测模型 | 检测车辆与车牌（YOLO / RT-DETR / RF-DETR） | 640×640 |
| `color_layer.onnx` | 车牌颜色 + 单/双层分类 | 48×168 |
| `ocr.onnx` | 车牌号 OCR | 48×320 |

**配置内置于代码，无需额外文件即可运行。** 如需自定义，可在 `configs/` 提供 YAML（外部配置优先级更高）：

- `det_config.yaml` — 检测类别名与可视化颜色
- `plate.yaml` — OCR 字符字典、颜色 / 层级映射
- `visualization_presets.yaml` — Supervision 可视化预设

字段说明见 [configs/CLAUDE.md](configs/CLAUDE.md)。

## 输出结果

`save` 模式在 `--output-dir`（默认 `runs/`）下生成：

- **标注图像 / 视频**：绘制边界框，车牌附带号码、颜色、层级（开启跟踪时带 ID）。
- **JSON**（图像默认输出，视频需 `--save-json`）：每个检测目标的结构化信息。

```json
{
    "detections": [
        {
            "box": [420, 529, 509, 562],
            "width": 89, "height": 33,
            "confidence": 0.93,
            "class_id": 0, "class_name": "plate",
            "plate_text": "苏A88888", "plate_conf": 0.95,
            "color": "blue", "layer": "single"
        }
    ]
}
```

## 项目结构

```
onnxtools/
├── onnxtools/          # 核心 Python 包（推理引擎、工具、跟踪、评估）
├── configs/            # 配置文件（检测类别、OCR 字典、可视化预设）
├── models/             # ONNX 模型与 TensorRT 引擎（.gitignore）
├── tools/              # 评估、TensorRT 构建、ONNX/TRT 对比
├── tests/              # 单元 / 集成 / 合约 / 性能测试
├── docs/               # 用户文档、Polygraphy 指南、API 参考
├── examples/           # 演示脚本（demo_pipeline / demo_detect / demo_crop）
├── mcp_tools/          # MCP 服务，LLM 工具接口
├── third_party/        # Ultralytics / Polygraphy / RF-DETR 集成
├── specs/ · openspec/  # 规范驱动开发（spec-kit / OpenSpec）
├── pyproject.toml      # 项目配置（uv）
└── CLAUDE.md           # AI 助手开发指南（含各模块导航）
```

各模块详细文档见对应目录下的 `CLAUDE.md`。

## 高级用法

```bash
# 构建 TensorRT FP16 引擎
python tools/trt/build_engine.py --onnx models/yolov8s_640.onnx \
    --output models/yolov8s_640_fp16.engine --precision fp16

# 模型评估（COCO mAP / OCR 指标）
python tools/eval/eval.py --model-path models/rtdetr-2024080100.onnx --test-dir data/test/

# MOT 跟踪评估（HOTA / MOTA / IDF1，MOTChallenge 格式）
# 用 GT 框作为理想检测现场跑某跟踪后端，对比关联质量：
python tools/eval/eval_mot.py --gt-root data/track/MOT_dataset --tracker botsort --frame-rate 5
# 或评估已有跟踪结果目录（每序列一个 <seq>.txt）：
python tools/eval/eval_mot.py --gt-root data/track/MOT_dataset --predictions runs/tracker_out

# ONNX vs TensorRT 性能对比
python tools/trt/compare_onnx_engine.py --onnx models/yolov8s_640.onnx --engine models/yolov8s_640.engine
```

Polygraphy 调试详见 [docs/polygraphy使用指南/](docs/polygraphy使用指南/)。

## 贡献指南

提交 PR 前请确保：遵循 PEP 8、补全类型提示与 Google 风格 docstring、添加对应测试。新功能 / 架构变更请先走 OpenSpec 提案流程（`openspec list --specs`）。

开发约定与架构契约见 [CLAUDE.md](CLAUDE.md)。

## 许可证

MIT License，详见 [LICENSE](LICENSE)。

## 相关资源

- [ONNX Runtime](https://onnxruntime.ai/docs/) · [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) · [Supervision](https://supervision.roboflow.com/) · [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)
