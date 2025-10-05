# Implementation Plan: 添加更多Supervision Annotators类型

**Branch**: `003-add-more-annotators` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/tyjt/桌面/onnx_vehicle_plate_recognition/specs/003-add-more-annotators/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → Feature spec loaded successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project Type: Single project (Python vehicle detection system)
   → Structure Decision: Option 1 (DEFAULT)
3. Fill Constitution Check section
   → Evaluated against 7 core principles
4. Evaluate Constitution Check section
   → Initial Constitution Check: PENDING
5. Execute Phase 0 → research.md
   → Research supervision annotators API usage via DeepWiki
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
7. Re-evaluate Constitution Check section
   → Post-Design Constitution Check: PENDING
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 9. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

扩展项目的supervision annotators支持，在现有BoxAnnotator和RichLabelAnnotator基础上添加13种可视化类型，包括圆角边框、角点标注、置信度条形图、几何标记(圆形/三角形/椭圆)、点标注、区域填充、背景叠加、光晕效果、以及隐私保护功能(模糊/像素化)。同时提供5种预设场景配置(标准检测、简洁轻量、隐私保护、调试分析、高对比展示)，通过统一的AnnotatorFactory和AnnotatorPipeline实现灵活的annotator组合和渲染顺序控制。

技术方法基于roboflow/supervision库的官方annotators API，采用配置驱动设计模式，通过扩展现有`utils/supervision_config.py`配置系统实现向后兼容的渐进式集成。

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: supervision (>=0.26.0), opencv-python, numpy, PIL
**Storage**: N/A (可视化功能无数据持久化需求)
**Testing**: pytest (contract tests, integration tests, performance benchmarks)
**Target Platform**: Linux (primary), Windows/macOS compatible
**Project Type**: single - Python ONNX vehicle detection library
**Performance Goals**: 记录和报告各annotator渲染时间，无强制性能指标
**Constraints**:
  - 保持与现有supervision_config.py配置系统兼容
  - 不破坏已有BoxAnnotator和RichLabelAnnotator功能
  - 支持annotator冲突警告机制(用户决策执行)
**Scale/Scope**:
  - 新增13种annotator类型
  - 5种预设场景配置
  - 扩展现有utils/模块(~200行新增代码)
  - 用户可使用DeepWiki查询supervision API: https://deepwiki.com/roboflow/supervision

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Modular Architecture ✅
- 所有新增annotator配置通过`supervision_config.py`统一管理
- AnnotatorFactory提供明确的创建接口
- AnnotatorPipeline封装组合逻辑，独立可测试
- 继承现有配置类设计模式

### II. Configuration-Driven Design ✅
- 5种预设场景通过YAML或配置类定义
- AnnotatorType枚举外部化，无硬编码
- 用户可通过配置文件/命令行参数切换annotator

### III. Performance First ✅
- 提供性能基准测试工具(FR-015)
- 记录各annotator性能特征(FR-016)
- 无强制性能要求，灵活适配不同场景
- 渲染顺序可配置以优化图层叠加

### IV. Type Safety and Contract Validation ✅
- 所有新增函数使用Python类型提示
- AnnotatorFactory输入验证和参数校验
- 合约测试验证API签名和返回类型
- 明确的错误消息(如冲突警告)

### V. Test-Driven Development (TDD) ✅
- Phase 1生成合约测试(先写测试后实现)
- 集成测试验证annotator组合场景
- 性能基准测试度量渲染时间
- 保持现有功能的回归测试

### VI. Observability and Debugging ✅
- 冲突警告日志输出(FR-024)
- 性能特征记录(FR-016)
- 使用Python logging模块
- DeepWiki文档引用支持调试

### VII. Simplicity and Incremental Growth (YAGNI) ✅
- 选择性扩展策略(13种而非全部22种)
- 基于现有配置系统渐进式集成
- 无premature优化(记录性能而非强制优化)
- 删除旧版不兼容代码(基于002规范已完成)

**Initial Check Status**: ✅ PASS - 符合所有7项核心原则

## Project Structure

### Documentation (this feature)
```
specs/003-add-more-annotators/
├── plan.md              # This file (/plan command output)
├── spec.md              # Feature specification (input)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── annotator_api.yaml
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
utils/
├── supervision_config.py       # 扩展：新增13种annotator配置类
├── supervision_converter.py    # 保持：现有检测格式转换
├── font_utils.py              # 保持：字体处理工具
├── drawing.py                 # 保持：主绘制函数(使用supervision)
└── annotator_factory.py       # 新增：AnnotatorFactory和AnnotatorPipeline

tests/
├── contract/
│   ├── test_annotator_factory_contract.py    # 新增：AnnotatorFactory合约测试
│   └── test_annotator_pipeline_contract.py   # 新增：AnnotatorPipeline合约测试
├── integration/
│   ├── test_supervision_only.py              # 扩展：新增annotator集成测试
│   └── test_annotator_presets.py             # 新增：预设场景测试
└── performance/
    └── test_annotator_benchmark.py           # 新增：性能基准测试

models/
└── visualization_presets.yaml                # 新增：5种预设场景配置
```

**Structure Decision**: Option 1 (Single project) - 本项目为Python ONNX推理库，所有模块位于单一代码库。新功能通过扩展`utils/`模块实现，保持现有模块化架构。

## Phase 0: Outline & Research

### Research Tasks
1. **Supervision Annotators API研究** (已通过DeepWiki完成)
   - ✅ 13种annotator的初始化参数
   - ✅ annotate()方法签名
   - ✅ 各annotator特定配置选项

2. **现有配置系统集成模式**
   - 分析`supervision_config.py`现有设计
   - 确定扩展点和兼容性策略
   - 定义AnnotatorFactory创建模式

3. **性能测试框架选择**
   - pytest-benchmark用于annotator性能测试
   - 时间度量和统计报告方法
   - 基准测试数据集设计

4. **冲突检测策略**
   - 识别annotator类型冲突规则
   - 警告消息格式和日志级别
   - 用户决策交互模式

**Output**: research.md with detailed API analysis and integration patterns

## Phase 1: Design & Contracts

### Design Artifacts

1. **data-model.md** - 数据模型定义:
   - `AnnotatorType` 枚举(13种类型)
   - `AnnotatorConfig` 配置实体
   - `AnnotatorFactory` 工厂类接口
   - `AnnotatorPipeline` 管道类接口
   - `VisualizationPreset` 预设场景模型

2. **contracts/annotator_api.yaml** - API合约:
   - `create_annotator(type, config)` → annotator实例
   - `create_pipeline(annotators_config)` → pipeline实例
   - `get_preset(name)` → preset配置
   - 输入验证规则和错误处理

3. **quickstart.md** - 快速入门指南:
   - 基础使用示例(单个annotator)
   - 预设场景使用示例
   - 自定义annotator组合示例
   - 性能基准测试示例

4. **CLAUDE.md更新** - AI助手上下文:
   - 运行`.specify/scripts/bash/update-agent-context.sh claude`
   - 新增技术栈: supervision annotators API
   - 更新最近变更: 003-add-more-annotators功能
   - 保持文档<150行

### Contract Tests (Phase 1生成，初始FAIL)
- `test_annotator_factory_contract.py`: 验证factory创建所有13种annotator
- `test_annotator_pipeline_contract.py`: 验证pipeline组合和渲染顺序
- `test_preset_loading_contract.py`: 验证5种预设场景加载

**Output**: data-model.md, /contracts/*, failing contract tests, quickstart.md, CLAUDE.md updated

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

### Task Generation Strategy

**从Phase 1设计生成任务**:
1. **Contract Test Tasks** (优先级P，可并行):
   - T001 [P]: 实现`test_annotator_factory_contract.py`
   - T002 [P]: 实现`test_annotator_pipeline_contract.py`
   - T003 [P]: 实现`test_preset_loading_contract.py`

2. **Model/Entity Tasks** (优先级P):
   - T004 [P]: 实现`AnnotatorType`枚举
   - T005 [P]: 实现`AnnotatorConfig`基类和13个子类
   - T006 [P]: 实现`VisualizationPreset`模型

3. **Core Implementation Tasks** (依赖Model层):
   - T007: 实现`AnnotatorFactory.create_annotator()`
   - T008: 实现`AnnotatorFactory`参数验证
   - T009: 实现`AnnotatorPipeline.add_annotator()`
   - T010: 实现`AnnotatorPipeline.annotate()`渲染逻辑
   - T011: 实现冲突检测和警告机制

4. **Configuration Tasks**:
   - T012 [P]: 创建`visualization_presets.yaml`
   - T013 [P]: 实现预设场景加载器
   - T014: 扩展`supervision_config.py`集成新annotator

5. **Integration Test Tasks**:
   - T015: 测试圆角边框annotator集成
   - T016: 测试角点标注annotator集成
   - T017: 测试置信度条形图annotator集成
   - T018: 测试几何标记annotators(Circle/Triangle/Ellipse)
   - T019: 测试点标注annotator集成
   - T020: 测试区域填充annotator集成
   - T021: 测试背景叠加annotator集成
   - T022: 测试隐私保护annotators(Blur/Pixelate)
   - T023: 测试光晕效果annotator集成
   - T024: 测试5种预设场景
   - T025: 测试annotator组合和渲染顺序

6. **Performance Benchmark Tasks**:
   - T026: 实现annotator性能基准测试框架
   - T027: 基准测试所有13种annotator
   - T028: 生成性能报告和优化建议

7. **Documentation Tasks**:
   - T029: 更新quickstart.md实际示例
   - T030: 更新CLAUDE.md with 最终实现细节

### Ordering Strategy
- **TDD顺序**: Contract tests (T001-T003) → Models (T004-T006) → Implementation (T007-T014) → Integration tests (T015-T025) → Performance (T026-T028) → Docs (T029-T030)
- **依赖顺序**: AnnotatorType → AnnotatorConfig → Factory/Pipeline → Presets
- **并行标记 [P]**: 独立文件可并行实现(contract tests, models, config files)

### Estimated Output
**30个任务**，分为7个逻辑组，清晰的依赖关系和优先级标记

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

无违规项 - 本功能完全符合宪章7项核心原则。

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved: N/A (no ambiguities)
- [x] Complexity deviations documented: N/A (no violations)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*