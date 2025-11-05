<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# ONNX车辆牌照识别系统

## 变更日志 (Changelog)

**2025-10-11** - 🐛 Bug修复 + ⚙️ 配置优化
- 🐛 **JSON数组格式支持**: 修复OCR评估器无法处理JSON数组格式label文件的问题
  - 问题: `load_label_file()` 将 `["img1.jpg", "img2.jpg"]` 作为单个文件路径处理
  - 修复: 添加JSON数组检测和解析，自动展开多图片到独立样本
  - 兼容性: 完全向后兼容原有单图片格式，支持混合格式
  - 测试: 新增12个单元测试用例，覆盖所有边界情况（12/12通过）
- ⚙️ **TensorRT可选依赖**: 将TensorRT从核心依赖改为可选依赖 `[trt]`
  - 配置: 在 `pyproject.toml` 添加 `[project.optional-dependencies]` 和 `no-build-isolation-package`
  - 安装: `uv pip install pip setuptools wheel && uv pip install -e ".[trt]"`
  - 文档: 更新 `README.md` 和 `requirements.txt` 安装说明
  - 脚本: 创建 `install.sh` 便捷安装脚本和 `verify_installation.py` 验证脚本
- 📝 **文件变更**:
  - 修改: `infer_onnx/eval_ocr.py` - `load_label_file()` 添加JSON支持
  - 新增: `tests/unit/test_load_label_file.py` - 12个单元测试
  - 更新: `pyproject.toml` - TensorRT可选依赖配置
  - 更新: `requirements.txt` - 依赖说明和安装指南
  - 更新: `README.md` - 安装指南重构

**2025-10-10** - ✅ 完成OCR指标评估功能 (006-make-ocr-metrics) - Phase 4已交付,字符级分析完成
- ✅ **核心功能**: OCRDatasetEvaluator类提供完整的OCR模型性能评估
- ✅ **三大指标**: 完全准确率、归一化编辑距离、编辑距离相似度 (基于python-Levenshtein 0.27.1)
- ✅ **双输出模式**: 表格对齐终端输出(支持中文列宽20/15) + JSON导出格式
- ✅ **置信度过滤**: 可配置阈值,自动统计过滤样本数
- ✅ **进度日志**: 每50张图像记录进度,支持大规模数据集评估
- ✅ **Tab分隔格式**: 支持标准OCR数据集(train.txt/val.txt)
- ✅ **字符级分析** (Phase 4): SampleEvaluation数据类提供每样本详细指标,per_sample_results字段输出完整分析
- 📊 **实施进度**: Phase 1-4完成(18个任务,67%),Phase 5-7待实施(9个任务,33%)
- 📝 **新增文件**:
  - `infer_onnx/eval_ocr.py` - OCR评估器模块(324行,含per_sample_results)
  - `utils/ocr_metrics.py` - OCR指标计算函数(201行,含中文对齐修复)
  - `tests/contract/test_ocr_evaluator_contract.py` - 合约测试(11个测试用例,含per_sample_results验证)
  - `tests/integration/test_ocr_evaluation_integration.py` - 集成测试(8个测试用例)
  - `tests/unit/test_ocr_metrics.py` - 单元测试(23个测试用例,覆盖23种边界情况)
- 🔧 **模块导出**: OCRDatasetEvaluator、SampleEvaluation已添加到infer_onnx.__all__
- ✅ **测试覆盖**:
  - 合约测试: 11/11通过(基础流程、编辑距离、置信度过滤、JSON导出、表格格式、per_sample_results)
  - 集成测试: 8/8通过(端到端评估、参数验证、性能测试、边界情况处理)
  - 单元测试: 23/23通过(空字符串、长度差异、插入删除替换、中文字符、真实OCR场景等)
- ⏭️ **下一步**: Phase 5-7增强功能(置信度过滤优化、性能报告、文档完善、CLI工具)

**2025-10-09** - ✅ 完成BaseOnnx抽象方法强制实现与__call__优化 (005-baseonnx-postprocess-call)
- ✅ **抽象方法强制实现**: `_postprocess()`和`_preprocess_static()`添加@abstractmethod装饰器,强制所有子类实现
- ✅ **__call__方法重构**: 代码行数减少83.3% (60→10行),提取3个阶段方法(_prepare_inference, _execute_inference, _finalize_inference)
- ✅ **子类完整性验证**: 所有5个子类(YoloOnnx/RTDETROnnx/RFDETROnnx/ColorLayerONNX/OCRONNX)验证通过,修复2个子类缺失的实现
- ✅ **错误提示优化**: 统一的NotImplementedError格式,包含类名、方法名、职责描述和docstring引用
- ✅ **测试验证**: 单元测试100% (27/27),集成测试96.6% (170/176),无回归问题
- ✅ **代码质量**: 向后兼容性完整保持,代码结构大幅优化,模板方法模式清晰
- 📊 **性能指标**: 测试通过率96.6%,代码减少83.3%,5个子类全部验证
- 📝 **文档完善**: 创建COMPLETION_SUMMARY.md总结文档,抽象方法docstring完整(Args/Returns/Raises/Example)

**2025-10-09** - 完成ColorLayerONNX和OCRONNX重构 (004-refactor-colorlayeronnx-ocronnx)
- ✅ **核心重构**: ColorLayerONNX和OCRONNX成功继承BaseOnnx,统一初始化模式和会话管理
- ✅ **API统一**: 使用`__call__()`接口替代旧版`infer()`,符合Python惯例和BaseOnnx规范
- ✅ **函数迁移**: 所有预处理和后处理函数迁移到类内部,13个静态方法封装完整OCR流程
- ✅ **代码清理**: 删除utils/ocr_image_processing.py (245行) 和 utils/ocr_post_processing.py (98行)
- ✅ **依赖解耦**: 移除utils模块对OCR的循环依赖,改用infer_onnx模块统一管理
- ✅ **测试验证**: 27个单元测试全部通过,115/122集成测试通过 (7个失败为非核心功能)
- ✅ **性能优化**: Polygraphy懒加载减少初始化时间93%+ (800ms → 50ms)
- ⚠️ **向后兼容**: 保留旧版`infer()`方法并添加DeprecationWarning
- 📝 **文档更新**: 更新infer_onnx/CLAUDE.md、utils/CLAUDE.md和quickstart.md迁移指南

**2025-09-30 17:30:00 CST** - 完成Supervision Annotators扩展集成 (003-add-more-annotators)
- 新增13种annotator类型支持：RoundBox, BoxCorner, Circle, Triangle, Ellipse, Dot, Color, BackgroundOverlay, Halo, PercentageBar, Blur, Pixelate
- 实现AnnotatorFactory统一工厂模式和AnnotatorPipeline组合管道
- 创建5种预设场景：standard, lightweight, privacy, debug, high_contrast
- 完成性能基准测试：12种annotator通过测试（最快75μs，最慢1.5ms）
- 扩展supervision_config.py添加get_default_annotator_config()便捷函数
- 新增文件：
  - `utils/annotator_factory.py` - Annotator工厂和管道类
  - `utils/visualization_preset.py` - 可视化预设加载器
  - `tests/performance/test_annotator_benchmark.py` - 性能基准测试
  - `specs/003-add-more-annotators/performance_report.md` - 性能分析报告

**2025-09-30 11:05:14 CST** - 完整初始化AI上下文架构
- 全面扫描项目结构，识别8个主要模块
- 生成完整的模块结构图和索引
- 创建/更新所有模块级CLAUDE.md文档
- 建立测试和规范(specs)文档体系
- 统计项目规模：100+ Python文件，覆盖核心推理、工具、测试和MCP集成

**2025-09-15 当前** - 正在进行supervision库可视化集成
- 分支: `001-supervision-plate-box`
- 状态: Phase 1设计阶段，已完成002-delete-old-draw重构
- 目标: 使用supervision库替换utils/drawing.py自定义可视化功能
- 进展: 完成技术调研，正在设计API合约和数据模型

**2025-09-15 20:01:23 CST** - 初始化AI上下文架构，生成项目结构图和模块索引

---

## 项目愿景

基于ONNX模型的车辆和车牌识别系统，支持多种输入源（图像、视频、摄像头），提供高精度的车辆检测、车牌识别、字符OCR和颜色/层级分类功能。该项目采用模块化架构设计，支持多种模型架构（YOLO、RT-DETR、RF-DETR），提供TensorRT加速优化，并通过MCP协议实现标准化服务集成。

## 架构概览

该项目采用模块化设计，分为推理引擎、工具集、第三方库、MCP集成和测试规范五个主要层次：

- **核心推理引擎** (`infer_onnx/`): 多模型架构支持（YOLO、RT-DETR、RF-DETR），基于Polygraphy懒加载
- **工具与实用程序** (`utils/`): 图像处理、模型评估、可视化工具、13种supervision annotators集成
- **调试和优化工具** (`tools/`): TensorRT引擎构建、性能评估、精度调试
- **模型资源管理** (`models/`): ONNX模型文件、配置文件、TensorRT引擎
- **MCP服务扩展** (`mcp_vehicle_detection/`): 模型上下文协议标准化服务接口
- **第三方集成** (`third_party/`): Ultralytics、Polygraphy、RF-DETR、TRT Engine Explorer
- **测试和规范** (`tests/`, `specs/`): 单元测试、集成测试、性能测试、功能规范

## 模块结构图

```mermaid
graph TD
    A["(根目录) ONNX车辆牌照识别系统"] --> B["infer_onnx"];
    A --> C["utils"];
    A --> D["tools"];
    A --> E["models"];
    A --> F["third_party"];
    A --> G["mcp_vehicle_detection"];
    A --> H["docs"];
    A --> I["tests"];
    A --> J["specs"];
    A --> K["runs"];
    A --> L["data"];

    B --> B1["base_onnx.py - 基础推理引擎"];
    B --> B2["yolo_onnx.py - YOLO模型推理"];
    B --> B3["rtdetr_onnx.py - RT-DETR推理"];
    B --> B4["rfdetr_onnx.py - RF-DETR推理"];
    B --> B5["ocr_onnx.py - OCR与颜色分类"];
    B --> B6["infer_models.py - 模型工厂"];
    B --> B7["eval_coco.py - 数据集评估"];

    C --> C1["pipeline.py - 处理管道"];
    C --> C2["image_processing.py - 图像预处理"];
    C --> C3["ocr_post_processing.py - OCR后处理"];
    C --> C4["logging_config.py - 日志配置"];
    C --> C5["detection_metrics.py - 检测指标"];
    C --> C6["nms.py - 非极大值抑制"];
    C --> C7["annotator_factory.py - Annotator工厂和管道"];
    C --> C8["visualization_preset.py - 可视化预设"];
    C --> C9["supervision_config.py - Supervision配置"];

    D --> D1["eval.py - 模型评估"];
    D --> D2["build_engine.py - TensorRT构建"];
    D --> D3["compare_onnx_engine.py - 模型比较"];
    D --> D4["draw_engine.py - 引擎可视化"];
    D --> D5["layer_statistics.py - 层统计"];
    D --> D6["debug/ - 调试脚本集"];

    E --> E1["*.onnx - ONNX模型文件"];
    E --> E2["det_config.yaml - 检测配置"];
    E --> E3["plate.yaml - OCR配置"];
    E --> E4["*.engine - TensorRT引擎"];

    F --> F1["ultralytics - YOLO实现"];
    F --> F2["Polygraphy - NVIDIA调试工具"];
    F --> F3["rfdetr - RF-DETR实现"];
    F --> F4["trt-engine-explorer - 引擎分析"];

    G --> G1["server.py - MCP服务器"];
    G --> G2["main.py - 检测服务"];
    G --> G3["models/ - 数据模型"];
    G --> G4["services/ - 服务层"];
    G --> G5["mcp_utils/ - MCP工具"];

    H --> H1["evaluation_guide.md - 评估指南"];
    H --> H2["polygraphy使用指南/ - Polygraphy文档"];

    I --> I1["integration/ - 集成测试"];
    I --> I2["contract/ - 合约测试"];
    I --> I3["unit/ - 单元测试"];
    I --> I4["performance/ - 性能测试"];
    I --> I5["conftest.py - 测试配置"];

    J --> J1["001-supervision-plate-box/ - 可视化规范"];
    J --> J2["002-delete-old-draw/ - 重构规范"];

    K --> K1["rfdetr-*/  - 评估结果目录"];
    K --> K2["rtdetr-*/  - 评估结果目录"];
    K --> K3["*.json - 检测结果JSON"];

    L --> L1["sample.jpg - 示例数据"];

    click B "./infer_onnx/CLAUDE.md" "查看推理引擎模块文档"
    click C "./utils/CLAUDE.md" "查看工具模块文档"
    click D "./tools/CLAUDE.md" "查看调试工具文档"
    click E "./models/CLAUDE.md" "查看模型配置文档"
    click G "./mcp_vehicle_detection/CLAUDE.md" "查看MCP模块文档"
    click I "./tests/CLAUDE.md" "查看测试模块文档"
    click J "./specs/CLAUDE.md" "查看规范文档"
```

## 模块索引

| 模块路径 | 职责 | 入口文件 | 主要功能 | 状态 |
|---------|------|----------|---------|------|
| [`infer_onnx/`](./infer_onnx/CLAUDE.md) | 核心推理引擎 | `infer_models.py::create_detector()` | 多模型架构支持、OCR识别、颜色分类 | ✅ 活跃 |
| [`utils/`](./utils/CLAUDE.md) | 通用工具库 | `pipeline.py::process_frame()` | 图像处理、可视化、OCR后处理 | ✅ 活跃 |
| [`tools/`](./tools/CLAUDE.md) | 调试和优化 | `eval.py`, `build_engine.py` | 模型评估、TensorRT构建、性能分析 | ✅ 活跃 |
| [`models/`](./models/CLAUDE.md) | 模型资源 | `det_config.yaml`, `plate.yaml` | 模型文件、配置文件存储 | ✅ 活跃 |
| [`mcp_vehicle_detection/`](./mcp_vehicle_detection/CLAUDE.md) | MCP服务集成 | `server.py`, `main.py` | MCP协议车辆检测服务 | ✅ 活跃 |
| [`third_party/`](./third_party/CLAUDE.md) | 第三方集成 | 各子模块独立 | Ultralytics、Polygraphy、RF-DETR | ✅ 稳定 |
| [`docs/`](./docs/CLAUDE.md) | 项目文档 | `evaluation_guide.md` | 使用指南、Polygraphy文档 | ✅ 维护中 |
| [`tests/`](./tests/CLAUDE.md) | 测试体系 | `conftest.py` | 单元、集成、性能、合约测试 | ✅ 活跃 |
| [`specs/`](./specs/CLAUDE.md) | 功能规范 | 各规范独立 | 设计文档、API合约、任务计划 | ✅ 活跃 |
| `runs/` | 运行结果 | - | 评估结果、检测输出存储 | ✅ 自动生成 |
| `data/` | 数据资源 | - | 示例图像、测试数据 | ✅ 稳定 |

## 运行和开发

### 环境要求
```yaml
基础环境:
  - Python: ">= 3.10"
  - CUDA: "11.8+"
  - TensorRT: "8.6.1"

依赖管理:
  - uv: "推荐包管理器"
  - pip: "备用包管理器"
```

### 安装依赖
```bash
# 使用uv（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt

# 安装MCP子项目依赖
cd mcp_vehicle_detection
uv sync
# 或
pip install -r requirements.txt
```

### 基本使用
```bash
# 图像推理示例
python main.py --model-path models/rtdetr-2024080100.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show

# 视频推理示例（使用RF-DETR）
python main.py --model-path models/rfdetr-20250811.onnx \
    --model-type rfdetr \
    --input /path/to/video.mp4 \
    --output-mode save

# 摄像头实时推理
python main.py --model-path models/yolo11n.onnx \
    --model-type yolo \
    --input 0 \
    --output-mode show

# 使用新的annotator可视化（预设场景）
python main.py --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show \
    --annotator-preset debug

# 自定义annotator组合
python main.py --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show \
    --annotator-types round_box percentage_bar rich_label \
    --box-thickness 3 \
    --roundness 0.4
```

### Annotator可视化选项

项目支持13种annotator类型和5种预设场景：

**预设场景**：
- `standard` - 标准检测模式（默认边框+标签）
- `lightweight` - 轻量级模式（点标记+简单标签）
- `privacy` - 隐私保护模式（边框+车牌模糊）
- `debug` - 调试模式（圆角框+置信度条+详细标签）
- `high_contrast` - 高对比度模式（区域填充+背景变暗）

**Annotator类型**：
- 边框类: `box`, `round_box`, `box_corner`
- 几何标记: `circle`, `triangle`, `ellipse`, `dot`
- 填充类: `color`, `background_overlay`
- 特效类: `halo`, `percentage_bar`
- 隐私保护: `blur`, `pixelate`

详细使用说明请参考 [`docs/annotator_usage.md`](docs/annotator_usage.md)

### 模型评估
```bash
# COCO数据集评估
python tools/eval.py \
    --model-type rtdetr \
    --model-path models/rtdetr-2024080100.onnx \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25 \
    --iou-threshold 0.7
```

### TensorRT引擎构建
```bash
# 构建FP16引擎
python tools/build_engine.py \
    --onnx-path models/rtdetr-2024080100.onnx \
    --engine-path models/rtdetr-2024080100.engine \
    --fp16

# 构建并对比精度
python tools/build_engine.py \
    --onnx-path models/rtdetr-2024080100.onnx \
    --compare
```

### MCP服务启动
```bash
# 启动MCP服务器
cd mcp_vehicle_detection
python server.py

# 快速测试
python quick_test.py
```

### 模型类型支持
| 模型架构 | 特点 | 输入尺寸 | 推荐场景 |
|---------|------|---------|---------|
| **RT-DETR** | 实时DETR，平衡精度和速度 | 640x640 | 通用检测 |
| **RF-DETR** | 增强RF-DETR，高精度检测 | 640x640 | 高精度需求 |
| **YOLO** | YOLOv8/v11系列，快速检测 | 640x640 | 实时性要求高 |

## 测试策略

### 测试体系架构
```
tests/
├── unit/          # 单元测试 - 功能组件测试
├── integration/   # 集成测试 - 端到端流程测试
├── contract/      # 合约测试 - API接口验证
├── performance/   # 性能测试 - 基准测试和性能分析
└── conftest.py    # 测试配置和fixtures
```

### 单元测试
- 推理引擎模块测试 (`infer_onnx/`)
- 图像处理工具测试 (`utils/`)
- OCR后处理逻辑测试
- 模型工厂函数测试

### 集成测试
```bash
# 运行集成测试
pytest tests/integration/ -v

# 测试覆盖:
# - test_pipeline_integration.py: 端到端推理管道
# - test_ocr_integration.py: OCR识别流程
# - test_supervision_only.py: Supervision库集成
```

### 合约测试
```bash
# 运行合约测试
pytest tests/contract/ -v

# 测试覆盖:
# - test_convert_detections_contract.py: 数据转换合约
# - test_draw_detections_contract.py: 可视化API合约
# - test_benchmark_contract.py: 性能基准合约
```

### 性能测试
```bash
# 运行性能基准测试
pytest tests/performance/ -v --benchmark-only

# 性能指标:
# - 模型推理延迟 (< 50ms for 640x640)
# - GPU内存使用 (< 2GB for batch_size=1)
# - 可视化渲染性能 (< 30ms for 20 objects)
```

## 编码标准

### Python代码规范
- **PEP 8**: 遵循PEP 8编码风格
- **类型提示**: 使用Python 3.10+类型提示
- **文档字符串**: Google风格docstring
- **命名约定**:
  - 类名: PascalCase (如 `BaseOnnx`)
  - 函数名: snake_case (如 `create_detector`)
  - 常量: UPPER_CASE (如 `RUN`)

### 模型集成规范
- 所有模型推理类继承自 `BaseOnnx`
- 实现 `predict()` 和 `postprocess()` 抽象方法
- 统一的配置文件格式（YAML）
- 标准化的后处理接口

### 错误处理
- 使用Python `logging` 模块记录日志
- 关键路径添加异常处理
- 优雅的模型加载失败处理
- 提供有意义的错误信息

### 日志规范
```python
# 使用colorlog彩色日志
from utils.logging_config import setup_logger
setup_logger(log_level='INFO')

# 日志级别:
# DEBUG - 详细调试信息
# INFO - 一般信息（默认）
# WARNING - 警告信息
# ERROR - 错误信息
# CRITICAL - 严重错误
```

## AI使用指南

### 代码分析
- **推理引擎优化**: 专注于 `infer_onnx/` 模块的多模型架构设计
- **图像处理**: 重点关注 `utils/` 模块的预处理和后处理流程
- **TensorRT优化**: 理解 `tools/build_engine.py` 的引擎构建流程
- **MCP集成**: 研究 `mcp_vehicle_detection/` 的服务化实现

### 调试辅助
- **Polygraphy工具**: 使用 `docs/polygraphy使用指南/` 进行深度调试
- **精度问题**: 利用 `tools/compare_onnx_engine.py` 对比ONNX和TensorRT
- **性能分析**: 使用 `tools/layer_statistics.py` 分析模型层性能
- **引擎检查**: 通过 `third_party/trt-engine-explorer/` 分析引擎结构

### 功能扩展
- **新模型架构**: 在 `infer_onnx/` 添加新的推理类
- **图像处理**: 扩展 `utils/` 模块的处理功能
- **MCP工具**: 在 `mcp_vehicle_detection/` 添加新的MCP工具
- **测试覆盖**: 在 `tests/` 添加对应的测试用例

### 规范驱动开发
- **功能设计**: 在 `specs/` 创建规范文档（参考001和002示例）
- **合约测试**: 在 `tests/contract/` 编写合约测试验证API
- **渐进实现**: 按照规范的Phase划分逐步实现功能

## 项目统计

### 代码规模
```
核心代码:
  - Python文件: 100+ 个
  - ONNX模型: 10+ 个
  - TensorRT引擎: 5+ 个
  - 配置文件: 100+ 个

测试覆盖:
  - 集成测试: 5个测试套件 (新增OCR评估集成测试8个用例)
  - 合约测试: 4个测试套件 (新增OCR评估合约测试11个用例)
  - 单元测试: 1个测试套件 (OCR指标计算23个用例)
  - 性能测试: 1个测试套件 (Annotator性能基准)

文档体系:
  - 模块文档: 8个CLAUDE.md
  - 功能规范: 2个specs
  - 使用指南: 多个markdown文档
```

### 第三方依赖
```
核心库:
  - onnxruntime-gpu: 1.22.0
  - tensorrt: 8.6.1.post1
  - opencv-contrib-python: 4.12.0+
  - numpy: 2.2.6+
  - supervision: 0.26.1

工具库:
  - polygraphy: 0.49.26+
  - onnxslim: 0.1.65+
  - pyyaml: 6.0.2+
  - colorlog: 6.9.0+
```

## 常见问题

### Q: 如何选择合适的模型架构？
**A**:
- **实时性优先**: 选择YOLO系列（yolo11n.onnx）
- **精度优先**: 选择RF-DETR（rfdetr-20250811.onnx）
- **平衡需求**: 选择RT-DETR（rtdetr-2024080100.onnx）

### Q: TensorRT引擎构建失败怎么办？
**A**:
1. 检查ONNX模型兼容性（opset版本）
2. 验证TensorRT版本匹配（8.6.1）
3. 使用 `tools/build_engine.py --compare` 进行精度对比
4. 查看Polygraphy调试指南

### Q: 如何提高推理速度？
**A**:
1. 使用TensorRT引擎替代ONNX模型
2. 启用FP16精度 (`--fp16`)
3. 调整输入分辨率
4. 使用批处理推理
5. 确保GPU资源充足

### Q: OCR识别准确率低怎么改善？
**A**:
1. 检查车牌图像预处理质量
2. 调整OCR模型置信度阈值
3. 使用更大的OCR模型
4. 验证OCR字典完整性
5. 增加训练数据覆盖

### Q: 如何添加新的检测类别？
**A**:
1. 在 `configs/det_config.yaml` 添加类别名称
2. 在 `visual_colors` 分配对应颜色
3. 重新训练或更新模型
4. 更新测试用例

---

*最后更新: 2025-09-30 11:05:14 CST*
*项目路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/`*
