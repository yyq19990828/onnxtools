# Implementation Plan: BaseOnnx抽象方法强制实现与__call__优化

**Branch**: `005-baseonnx-postprocess-call` | **Date**: 2025-10-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-baseonnx-postprocess-call/spec.md`

**User Guidance**: __call__提取的3个阶段方法, 在BaseOnnx中提供基础的模板方法, 子类中按需重写或直接继承

## Summary

本功能将BaseOnnx推理基类重构为更严格的抽象基类架构:
1. **抽象方法强化**: 将`_postprocess()`和`_preprocess_static()`标记为@abstractmethod,强制子类实现
2. **__call__优化**: 保持__call__为具体模板方法,但重构为3个清晰的阶段方法(`_prepare_inference()`, `_execute_inference()`, `_finalize_inference()`),基类提供默认实现,子类可按需重写
3. **代码清理**: 基于测试覆盖率分析,删除未使用的旧版本分支逻辑,降低代码复杂度
4. **子类适配**: 验证并适配所有5个子类(YoloOnnx/RTDETROnnx/RFDETROnnx/ColorLayerONNX/OCRONNX),确保正确实现抽象方法

**技术方法**: 使用Python ABC模块的@abstractmethod装饰器 + 模板方法设计模式 + pytest-cov代码覆盖分析

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- abc (Python标准库) - 抽象基类支持
- onnxruntime-gpu 1.22.0 - ONNX推理引擎
- numpy 2.2.6+ - 张量操作
- pytest 8.0+ - 单元和集成测试
- pytest-cov 4.0+ - 代码覆盖率分析
- mypy - 静态类型检查
- pylint - 代码质量检查

**Storage**: N/A (代码重构,不涉及数据存储)
**Testing**: pytest + pytest-cov (覆盖率分析) + mypy (类型检查)
**Target Platform**: Linux/Windows (跨平台,支持GPU和CPU)
**Project Type**: 单一Python项目 (机器学习推理库)
**Performance Goals**:
- 推理延迟保持< 50ms (640x640输入)
- GPU内存< 2GB (batch_size=1)
- __call__方法代码行数减少30%+
- 圈复杂度降低

**Constraints**:
- 必须保持向后兼容性(deprecated方法保留)
- 不能破坏现有5个子类的功能
- 测试通过率: 集成测试100%(115/122), 单元测试100%(27/27)
- 不能引入性能退化

**Scale/Scope**:
- 修改1个基类(BaseOnnx)
- 适配5个子类
- 影响约60行核心代码(__call__方法)
- 新增3个阶段方法(约80行)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Phase 0 Check (Pre-Research)

| Principle | Compliance | Justification |
|-----------|-----------|---------------|
| **I. Modular Architecture** | ✅ PASS | BaseOnnx作为基类已是模块化架构的核心,本次重构强化其接口契约,增强模块化 |
| **II. Configuration-Driven** | ✅ PASS | 不涉及配置变更,保持现有YAML配置驱动设计 |
| **III. Performance First** | ✅ PASS | 性能是成功标准之一(SC-006),重构不能引入退化 |
| **IV. Type Safety** | ✅ PASS | 使用@abstractmethod提供编译时检查,增强类型安全 |
| **V. Test-Driven Development** | ✅ PASS | 基于测试覆盖率指导重构(FR-014),先测试后重构 |
| **VI. Observability** | ✅ PASS | 保留现有logging,优化错误消息格式(FR-004) |
| **VII. Simplicity (YAGNI)** | ✅ PASS | 删除未使用代码(基于覆盖率),遵循YAGNI |

**Overall Status**: ✅ **PASS** - No violations

### Phase 1 Check (Post-Design)

*To be completed after data-model.md and contracts generation*

## Project Structure

### Documentation (this feature)

```
specs/005-baseonnx-postprocess-call/
├── spec.md              # 功能规格 (已完成)
├── plan.md              # 本文件 (实施计划)
├── research.md          # Phase 0 技术调研
├── data-model.md        # Phase 1 数据模型
├── quickstart.md        # Phase 1 快速入门
├── contracts/           # Phase 1 API合约
│   └── baseonnx_api.md  # BaseOnnx接口合约
└── tasks.md             # Phase 2 任务清单 (由/speckit.tasks生成)
```

### Source Code (repository root)

```
infer_onnx/
├── onnx_base.py            # BaseOnnx基类 (核心修改)
├── onnx_yolo.py            # YoloOnnx子类 (适配)
├── onnx_rtdetr.py          # RTDETROnnx子类 (适配)
├── onnx_rfdetr.py          # RFDETROnnx子类 (适配)
├── onnx_ocr.py             # ColorLayerONNX/OCRONNX子类 (适配)
└── CLAUDE.md               # 模块文档更新

tests/
├── unit/                   # 单元测试 (新增抽象方法测试)
├── integration/            # 集成测试 (验证子类推理)
└── contract/               # 合约测试 (验证接口合约)
```

**Structure Decision**: 采用Option 1 (单一项目结构),因为这是单一Python推理库,所有代码在`infer_onnx/`模块下,测试在`tests/`目录。

## Complexity Tracking

*本节仅在Constitution Check有违规时填写*

**Status**: 无违规需要追踪

---

## Phase 0: Research & Technical Decisions

*See [research.md](./research.md) for detailed findings*

### Research Tasks

1. **@abstractmethod与@staticmethod组合验证**
   - 研究Python 3.10+中@staticmethod和@abstractmethod的正确组合顺序
   - 验证实例化时的TypeError行为
   - 确认各Python版本(3.10/3.11/3.12)的一致性

2. **pytest-cov覆盖率分析最佳实践**
   - 研究pytest-cov的分支覆盖报告生成
   - 确定覆盖率阈值策略(删除0%分支 vs 保留低覆盖但关键分支)
   - 研究如何生成HTML报告和term-missing报告

3. **模板方法模式在推理管道中的应用**
   - 研究3阶段解耦的最佳粒度
   - 确定各阶段方法的参数传递策略
   - 研究子类重写策略(完全重写 vs 部分扩展)

4. **旧版本分支逻辑识别策略**
   - 识别onnx_base.py中的向后兼容代码
   - 确定哪些分支是真正未使用的
   - 研究如何安全移除兼容性代码

---

## Phase 1: Design & Contracts

*See [data-model.md](./data-model.md) and [contracts/](./contracts/) for detailed design*

### Core Entities

#### BaseOnnx (抽象基类)

**职责**: ONNX推理引擎的抽象基类,定义统一接口契约

**抽象方法**:
- `_postprocess(prediction, conf_thres, **kwargs) -> List[np.ndarray]` - 模型输出后处理
- `_preprocess_static(image, input_shape) -> Tuple` - 静态预处理方法

**具体模板方法**:
- `__call__(image, conf_thres, **kwargs) -> Tuple[List[np.ndarray], tuple]` - 推理入口

**新增阶段方法** (在BaseOnnx中提供默认实现,子类可重写):
- `_prepare_inference(image, conf_thres, **kwargs) -> Tuple` - 准备阶段
- `_execute_inference(input_tensor) -> List[np.ndarray]` - 执行阶段
- `_finalize_inference(outputs, scale, original_shape, conf_thres, **kwargs) -> List[np.ndarray]` - 完成阶段

#### SubClass (子类实现)

**必须实现**: `_postprocess()`, `_preprocess_static()`
**可选重写**: `_prepare_inference()`, `_execute_inference()`, `_finalize_inference()`
**默认继承**: `__call__()` (除非有特殊需求)

### API Contracts

*See [contracts/baseonnx_api.md](./contracts/baseonnx_api.md)*

核心合约:
1. 所有子类必须实现2个抽象方法,否则实例化时抛出TypeError
2. __call__方法调用3个阶段方法,按顺序执行推理流程
3. 3个阶段方法在BaseOnnx中有默认实现,子类可按需重写
4. 错误消息格式统一: "{ClassName}.{method_name}() must be implemented by subclass. {责任描述}"

---

## Phase 2: Task Breakdown

*Tasks will be generated by `/speckit.tasks` command. This section is a placeholder.*

预期任务类型:
- **模块化任务** (Principle I): 重构BaseOnnx的__call__方法为3个阶段方法
- **类型安全任务** (Principle IV): 添加@abstractmethod装饰器和详细的docstring
- **测试任务** (Principle V): 生成覆盖率报告,添加抽象方法检查的单元测试
- **简化任务** (Principle VII): 删除未覆盖的旧版本分支逻辑

---

## Implementation Notes

### 关键决策

1. **3个阶段方法在BaseOnnx中提供默认实现** (来自用户输入)
   - `_prepare_inference()`: 调用`_ensure_initialized()` + `_preprocess()` + 参数处理
   - `_execute_inference()`: Polygraphy推理执行 + batch维度处理
   - `_finalize_inference()`: 调用`_postprocess()` + 结果格式化 + batch过滤

   子类可以:
   - 直接继承使用(默认行为,适用于大多数子类)
   - 重写特定阶段方法(如RTDETROnnx可能重写_finalize_inference处理特殊输出格式)
   - 保持__call__不变(除非极特殊需求)

2. **渐进式重构策略**
   - 步骤1: 添加@abstractmethod装饰器到_postprocess和_preprocess_static
   - 步骤2: 运行测试确认所有子类已实现
   - 步骤3: 生成覆盖率报告
   - 步骤4: 提取3个阶段方法(保持__call__不变)
   - 步骤5: 删除0%覆盖的分支
   - 步骤6: 运行完整测试套件验证

3. **向后兼容性保证**
   - deprecated方法(如infer())保留并发出DeprecationWarning
   - 子类不需要修改代码(除非缺少抽象方法实现)
   - __call__接口签名保持不变

### 风险缓解

- **风险**: 重构破坏子类功能
  - **缓解**: 每步重构后运行完整测试套件,保持集成测试100%通过率

- **风险**: 3个阶段方法的粒度不合适
  - **缓解**: 在research.md中研究最佳实践,在data-model.md中详细设计接口

- **风险**: 删除了仍在使用的兼容性代码
  - **缓解**: 严格基于pytest-cov覆盖率报告,只删除0%覆盖的分支

---

*Plan generated: 2025-10-09*
*Next command: Research findings documented in research.md*
