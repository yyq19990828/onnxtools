# Implementation Plan: BaseORT结果包装类

**Branch**: `001-baseort-result-third` | **Date**: 2025-11-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-baseort-result-third/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

实现一个精简的Result类来包装BaseORT子类的检测后处理输出，提供面向对象的API访问、可视化方法和数据转换功能。该类强制作为BaseORT.__call__()的返回值，替代现有的字典格式，同时提供临时的to_dict()方法用于向后兼容（2个迭代后移除）。采用浅层不可变设计（属性只读但内部数组可修改），使用numpy视图优化内存性能，避免不必要的数组拷贝。

**技术要点**:
- 继承SimpleClass基类（参考Ultralytics设计模式）
- 使用@property装饰器实现只读属性保护
- 集成现有Supervision可视化工具链（AnnotatorFactory、draw_detections_supervision）
- 支持None初始化并自动转换为形状正确的空numpy数组
- 目标创建开销<5ms，可视化<1秒（20个目标，640x640图像）

## Technical Context

**Language/Version**: Python 3.10+ （项目现有版本）
**Primary Dependencies**: numpy>=2.2.6, opencv-contrib-python>=4.12.0, supervision==0.26.1 （项目现有依赖）
**Storage**: N/A （内存中的数据结构，不涉及持久化）
**Testing**: pytest （项目现有测试框架）
**Target Platform**: Linux/Windows，CPU/GPU推理环境
**Project Type**: Single （单体Python包，onnxtools模块）
**Performance Goals**:
  - Result对象创建开销 < 5ms （640x640图像，20个检测目标）
  - plot()可视化方法 < 1秒 （20个目标，640x640图像）
  - 内存占用 < 原始字典结构的120%
**Constraints**:
  - 必须与现有BaseORT子类（YoloORT、RtdetrORT、RfdetrORT、OcrORT、ColorLayerORT）兼容
  - 不能破坏现有的推理管道性能
  - 必须复用现有Supervision集成，不重复实现可视化逻辑
**Scale/Scope**:
  - 单个Result类 + BaseORT基类修改
  - 影响5个BaseORT子类的__call__()返回值
  - 约500行核心代码 + 1000行测试代码

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Modular Architecture ✅ **PASS**
- Result类将作为独立模块位于`onnxtools/infer_onnx/result.py`
- 继承SimpleClass基类，符合项目统一抽象模式
- 公共接口通过`onnxtools/__init__.py`导出
- 可独立测试，无需完整推理管道

**评估**: 完全符合模块化原则，Result类边界清晰，仅依赖numpy和项目现有工具。

### II. Configuration-Driven Design ✅ **PASS**
- Result类不包含硬编码配置
- 可视化依赖现有的`configs/visualization_presets.yaml`
- annotator_preset参数外部化

**评估**: 无配置需求，符合原则。

### III. Performance First ✅ **PASS**
- 使用numpy视图避免数组拷贝（FR-004, FR-005, FR-009）
- 浅层不可变设计减少内存分配
- 目标创建开销<5ms（SC-006）
- 可视化<1秒（SC-002）

**评估**: 性能优化策略明确，符合<50ms推理延迟要求的补充约束。

### IV. Type Safety and Contract Validation ✅ **PASS**
- 所有公共方法必须添加类型提示（Python 3.10+ typing）
- 输入验证：boxes shape检查、orig_img非None验证
- 属性访问错误：尝试赋值抛出AttributeError（FR-017）

**评估**: 符合类型安全原则，需在实现时严格执行mypy检查。

### V. Test-Driven Development (TDD) ✅ **PASS**
- 计划单元测试覆盖率>90%（SC-003）
- 需要合约测试验证BaseORT子类返回Result对象
- 需要集成测试验证可视化和Supervision集成

**评估**: 测试优先策略明确，符合TDD原则。

### VI. Observability and Debugging ⚠️ **PARTIAL**
- Result类本身无需日志（纯数据结构）
- 但BaseORT.__call__()修改需添加调试日志（记录Result对象创建时机）
- 错误消息需清晰（如orig_img为None时的ValueError）

**评估**: 部分符合，需在BaseORT集成时补充日志。

### VII. Simplicity and Incremental Growth (YAGNI) ✅ **PASS**
- 明确排除verbose()、to_json()、save_crop()等非MVP功能
- 不实现设备转换方法（cpu/cuda/to）
- 浅层不可变优先于深拷贝（避免过度优化）

**评估**: 严格遵循YAGNI原则，功能精简。

**总体评估**: ✅ **APPROVED** - 7/7原则通过，1项部分符合（可在实现中补充）

## Project Structure

### Documentation (this feature)

```text
specs/001-baseort-result-third/
├── spec.md              # 功能规范（已完成）
├── plan.md              # 本文件（当前正在填写）
├── research.md          # Phase 0 输出（即将生成）
├── data-model.md        # Phase 1 输出（即将生成）
├── quickstart.md        # Phase 1 输出（即将生成）
├── contracts/           # Phase 1 输出（即将生成）
│   └── result_api.yaml  # Result类API合约
└── tasks.md             # Phase 2 输出（需单独执行/speckit.tasks命令）
```

### Source Code (repository root)

```text
onnxtools/                          # 核心Python包
├── __init__.py                     # 导出Result类
├── infer_onnx/                     # 推理引擎模块
│   ├── __init__.py                 # 更新导出Result
│   ├── onnx_base.py                # 修改：BaseORT.__call__()返回Result对象
│   ├── result.py                   # 新增：Result类实现（核心文件）
│   ├── onnx_yolo.py                # 修改：返回Result对象
│   ├── onnx_rtdetr.py              # 修改：返回Result对象
│   ├── onnx_rfdetr.py              # 修改：返回Result对象
│   ├── onnx_ocr.py                 # 修改：返回Result对象（OCR和Color）
│   └── CLAUDE.md                   # 更新：Result类文档
│
├── utils/                          # 工具模块（复用现有）
│   ├── drawing.py                  # 复用：draw_detections_supervision()
│   ├── supervision_converter.py   # 复用：convert_to_supervision_detections()
│   └── annotator_factory.py       # 复用：AnnotatorFactory
│
└── CLAUDE.md                       # 更新：根模块文档

tests/                              # 测试套件
├── unit/                           # 单元测试
│   ├── test_result.py              # 新增：Result类单元测试
│   └── test_result_property.py    # 新增：只读属性测试
│
├── integration/                    # 集成测试
│   ├── test_result_integration.py # 新增：Result与BaseORT集成测试
│   └── test_result_visualization.py # 新增：可视化集成测试
│
└── contract/                       # 合约测试
    └── test_result_contract.py    # 新增：Result类API合约测试

main.py                             # 修改：更新Result对象使用示例
```

**Structure Decision**: 采用Single项目结构，Result类作为`onnxtools.infer_onnx`子模块的新增文件。选择此结构的原因：
1. Result类与BaseORT紧密耦合，属于推理引擎的一部分
2. 复用现有utils模块，无需新增依赖
3. 测试套件按照项目现有结构组织（unit/integration/contract三层）
4. 符合项目模块化架构原则（Principle I）

## Complexity Tracking

> **无宪章违规，本节为空**

所有设计决策均符合项目宪章7项核心原则，无需复杂性豁免。
