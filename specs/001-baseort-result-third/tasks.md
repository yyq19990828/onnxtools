# Tasks: BaseORT结果包装类

**Input**: Design documents from `/specs/001-baseort-result-third/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/result_api.yaml

**Tests**: 根据spec.md，本功能**未明确要求TDD方式**，但需要高单元测试覆盖率（>90%），因此测试任务与实现任务并行进行，而非测试优先。

**Organization**: 任务按用户故事分组，支持每个故事的独立实现和测试。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行执行（不同文件，无依赖关系）
- **[Story]**: 任务所属的用户故事（US1, US2, US3）
- 任务描述中包含精确的文件路径

---

## Phase 1: Setup（项目初始化）

**目的**: 创建Result类的基础结构和测试框架

- [X] T001 在`onnxtools/infer_onnx/result.py`中创建Result类骨架，包含`__init__`方法和基础属性定义
- [X] T002 在`onnxtools/__init__.py`中导出Result类，使其可通过`from onnxtools import Result`访问
- [X] T003 [P] 在`onnxtools/infer_onnx/__init__.py`中导出Result类
- [X] T004 [P] 创建测试目录结构：`tests/unit/test_result.py`、`tests/integration/test_result_integration.py`、`tests/contract/test_result_contract.py`

---

## Phase 2: Foundational（阻塞性前置任务）

**目的**: 完成Result类的核心基础设施，所有用户故事依赖这些任务

**⚠️ 关键**: 在此阶段完成之前，用户故事实现无法开始

- [X] T005 在`onnxtools/infer_onnx/result.py`中实现`__init__`方法的参数验证逻辑（V1-V6规则，参考data-model.md）
- [X] T006 [P] 在`onnxtools/infer_onnx/result.py`中实现所有@property装饰器（boxes、scores、class_ids、orig_img、orig_shape、names、path），包含None到空数组的自动转换
- [X] T007 [P] 在`onnxtools/infer_onnx/result.py`中实现`__len__`魔术方法，返回检测目标数量
- [X] T008 [P] 在`onnxtools/infer_onnx/result.py`中实现`__repr__`和`__str__`方法，提供可读的对象表示
- [X] T009 在`tests/unit/test_result.py`中实现Result类初始化的单元测试（覆盖所有验证规则V1-V6）
- [X] T010 [P] 在`tests/contract/test_result_contract.py`中实现API合约测试（基于contracts/result_api.yaml的initialization测试集）

**Checkpoint**: Result类基础架构完成，可以创建Result对象并访问属性

---

## Phase 3: User Story 1 - 基础检测结果访问和操作 (Priority: P1) 🎯 MVP

**目标**: 开发人员可以通过Result对象以面向对象方式访问检测结果（boxes、scores、class_ids），使用索引和切片操作，获取检测数量

**独立测试**: 创建Result对象，通过属性访问数据，使用len()和索引操作，验证数据正确性和API便捷性

### 实现任务

- [X] T011 [P] [US1] 在`onnxtools/infer_onnx/result.py`中实现`__getitem__`方法，支持整数索引（返回单个检测的新Result对象）
- [X] T012 [P] [US1] 在`onnxtools/infer_onnx/result.py`中扩展`__getitem__`方法，支持切片操作（返回子集的新Result对象，使用numpy视图）
- [X] T013 [US1] 在`onnxtools/infer_onnx/result.py`中实现`numpy()`方法，确保所有内部数据为numpy.ndarray格式（幂等操作）
- [X] T014 [US1] 修改`onnxtools/infer_onnx/onnx_base.py`中的`BaseORT.__call__()`方法，使其返回Result对象而非字典（集成BaseORT与Result类）
- [X] T015 [US1] 在`onnxtools/infer_onnx/result.py`中实现`to_dict()`方法，添加DeprecationWarning并返回字典格式（向后兼容支持）

### 测试任务

- [x] T016 [P] [US1] 在`tests/unit/test_result.py`中实现属性访问测试（boxes、scores、class_ids、orig_shape、names、path）
- [x] T017 [P] [US1] 在`tests/unit/test_result_property.py`中创建只读属性保护测试（验证尝试赋值抛出AttributeError）
- [x] T018 [P] [US1] 在`tests/unit/test_result.py`中实现`__len__`测试（空结果返回0，有N个检测返回N）
- [x] T019 [P] [US1] 在`tests/unit/test_result.py`中实现`__getitem__`单个索引测试（正常索引、负索引、越界IndexError）
- [x] T020 [P] [US1] 在`tests/unit/test_result.py`中实现`__getitem__`切片测试（result[1:3]、result[:5]、result[-1]）
- [x] T021 [US1] 在`tests/unit/test_result.py`中实现空检测结果测试（None初始化、len()=0、索引抛出IndexError、属性访问返回空数组）
- [x] T022 [P] [US1] 在`tests/integration/test_result_integration.py`中实现BaseORT集成测试（验证YoloORT、RtdetrORT、RfdetrORT返回Result对象）
- [x] T023 [P] [US1] 在`tests/contract/test_result_contract.py`中实现索引和切片的合约测试（基于contracts/result_api.yaml的indexing测试集）
- [x] T024 [US1] 在`tests/unit/test_result.py`中实现to_dict()废弃警告测试（验证DeprecationWarning被触发）

**Checkpoint**: User Story 1完成 - Result对象可创建、属性可访问、支持索引和切片操作，BaseORT集成完成

---

## Phase 4: User Story 2 - 结果可视化和保存 (Priority: P2)

**目标**: 开发人员可以快速可视化检测结果（plot/show）并保存标注图像（save），无需手动编写绘制代码

**独立测试**: 创建Result对象，调用plot()获取标注图像，调用show()显示，调用save()保存到文件，验证输出图像正确性

### 实现任务

- [x] T025 [P] [US2] 在`onnxtools/infer_onnx/result.py`中实现`to_supervision()`方法，调用`onnxtools.utils.supervision_converter.convert_to_supervision_detections()`
- [x] T026 [US2] 在`onnxtools/infer_onnx/result.py`中实现`plot()`方法，集成Supervision可视化工具链（AnnotatorFactory、draw_detections_supervision），支持annotator_preset参数
- [x] T027 [US2] 在`onnxtools/infer_onnx/result.py`中实现`show()`方法，调用cv2.imshow()显示标注图像（内部调用plot()）
- [x] T028 [US2] 在`onnxtools/infer_onnx/result.py`中实现`save()`方法，调用cv2.imwrite()保存标注图像到指定路径（内部调用plot()）
- [x] T029 [US2] 在`plot()`、`show()`、`save()`方法中添加orig_img非None的前提条件验证，抛出ValueError并提供清晰错误消息

### 测试任务

- [x] T030 [P] [US2] 在`tests/unit/test_result.py`中实现`to_supervision()`转换测试（验证返回supervision.Detections对象，数据一致性）
- [x] T031 [P] [US2] 在`tests/unit/test_result.py`中实现`plot()`方法测试（默认annotator_preset、自定义preset、返回numpy数组类型验证）
- [x] T032 [P] [US2] 在`tests/unit/test_result.py`中实现可视化方法的错误处理测试（orig_img为None时抛出ValueError）
- [x] T033 [US2] 在`tests/integration/test_result_visualization.py`中实现可视化集成测试（端到端测试plot/show/save方法，验证输出图像质量）
- [x] T034 [P] [US2] 在`tests/contract/test_result_contract.py`中实现可视化的合约测试（基于contracts/result_api.yaml的visualization测试集）
- [x] T035 [US2] 在`tests/performance/test_result_plot_benchmark.py`中实现plot()性能基准测试（验证<50ms，20个目标，640x640图像）

**Checkpoint**: ✅ User Story 2完成 - Result对象支持可视化和保存功能，性能达标

---

## Phase 5: User Story 3 - 结果过滤和转换 (Priority: P3)

**目标**: 开发人员可以根据条件过滤检测结果（置信度、类别），将结果转换为其他格式（summary统计），便于后续处理

**独立测试**: 创建Result对象，应用过滤条件（conf_threshold、classes），验证返回的新Result对象仅包含符合条件的检测；调用summary()验证统计信息正确

### 实现任务

- [x] T036 [P] [US3] 在`onnxtools/infer_onnx/result.py`中实现`filter()`方法的置信度过滤逻辑（conf_threshold参数，使用numpy布尔索引）
- [x] T037 [P] [US3] 在`onnxtools/infer_onnx/result.py`中实现`filter()`方法的类别过滤逻辑（classes参数，使用numpy.isin()）
- [x] T038 [US3] 在`onnxtools/infer_onnx/result.py`中实现`filter()`方法的组合过滤（同时支持conf_threshold和classes，返回新Result对象）
- [x] T039 [US3] 在`filter()`方法中添加参数验证（conf_threshold必须在0-1之间，classes必须为整数列表）
- [x] T040 [US3] 在`onnxtools/infer_onnx/result.py`中实现`summary()`方法，返回包含total_detections、class_counts、avg_confidence、min_confidence、max_confidence的字典

### 测试任务

- [x] T041 [P] [US3] 在`tests/unit/test_result.py`中实现`filter()`置信度过滤测试（单一阈值、边界条件、空结果）
- [x] T042 [P] [US3] 在`tests/unit/test_result.py`中实现`filter()`类别过滤测试（单个类别、多个类别、不存在的类别）
- [x] T043 [P] [US3] 在`tests/unit/test_result.py`中实现`filter()`组合过滤测试（同时应用置信度和类别过滤）
- [x] T044 [P] [US3] 在`tests/unit/test_result.py`中实现`filter()`参数验证测试（无效的conf_threshold、无效的classes类型）
- [x] T045 [US3] 在`tests/unit/test_result.py`中实现`filter()`返回空结果测试（过滤后无匹配项，返回空Result对象而非None）
- [x] T046 [P] [US3] 在`tests/unit/test_result.py`中实现`summary()`方法测试（验证所有统计字段正确计算，空结果情况）
- [x] T047 [P] [US3] 在`tests/contract/test_result_contract.py`中实现过滤的合约测试（基于contracts/result_api.yaml的filtering测试集）

**Checkpoint**: User Story 3完成 - Result对象支持过滤和统计功能，所有核心功能齐全

---

## Phase 6: Polish & Cross-Cutting Concerns（优化和完善）

**目的**: 完善文档、代码清理、性能验证、全量测试

- [ ] T048 [P] 更新`onnxtools/infer_onnx/CLAUDE.md`文档，添加Result类的使用说明和示例
- [ ] T049 [P] 更新`onnxtools/CLAUDE.md`根模块文档，说明Result类的公共API
- [ ] T050 [P] 更新`main.py`示例代码，展示Result对象的使用方式（替代旧的字典访问）
- [ ] T051 代码审查和重构：检查Result类实现是否符合PEP 8规范、类型提示完整性、docstring完整性（Google风格）
- [ ] T052 [P] 在`tests/unit/test_result.py`中补充边界情况单元测试（切片越界、空结果遍历、视图修改行为）
- [ ] T053 运行完整测试套件，验证单元测试覆盖率>90%（使用pytest --cov=onnxtools.infer_onnx.result --cov-report=html）
- [ ] T054 [P] 在`tests/performance/test_result_performance.py`中实现Result对象创建的性能基准测试（验证<5ms，20个目标）
- [ ] T055 [P] 在`tests/performance/test_result_performance.py`中实现内存占用基准测试（验证<120%原始字典，使用memory_profiler）
- [ ] T056 验证`specs/001-baseort-result-third/quickstart.md`中的所有示例代码可执行（运行quickstart示例脚本）
- [ ] T057 修改所有BaseORT子类（YoloORT、RtdetrORT、RfdetrORT、OcrORT）的示例代码和测试，确保兼容Result对象返回值
- [ ] T058 [P] 运行mypy类型检查，确保Result类所有方法的类型提示正确（mypy onnxtools/infer_onnx/result.py --strict）

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (BLOCKS all user stories)
    ↓
Phase 3: User Story 1 (P1) ──┐
Phase 4: User Story 2 (P2) ──┼── Can run in parallel after Phase 2
Phase 5: User Story 3 (P3) ──┘
    ↓
Phase 6: Polish & Cross-Cutting Concerns
```

### User Story Dependencies

- **User Story 1 (P1)**: 依赖Phase 2完成 - 无其他用户故事依赖
- **User Story 2 (P2)**: 依赖Phase 2完成 - 部分依赖US1（需要to_supervision()），但可独立测试可视化功能
- **User Story 3 (P3)**: 依赖Phase 2完成 - 无其他用户故事依赖

### Within Each User Story

- 实现任务先于测试任务（非TDD模式，但测试与实现可并行进行）
- Result类核心方法 → BaseORT集成 → 合约测试验证
- 每个故事阶段完成后应达到独立可测试状态

### Parallel Opportunities

**Setup阶段**（Phase 1）:
- T002, T003, T004 可并行执行（不同文件）

**Foundational阶段**（Phase 2）:
- T006, T007, T008 可并行执行（同一文件的不同方法）
- T009, T010 可并行执行（不同测试文件）

**User Story 1阶段**（Phase 3）:
- T011, T012 可并行执行（同一方法的不同功能分支）
- T016, T017, T018, T019, T020 可并行执行（不同测试用例）
- T022, T023 可并行执行（不同测试文件）

**User Story 2阶段**（Phase 4）:
- T025, T026 可并行执行（不同方法）
- T030, T031, T032 可并行执行（不同测试用例）
- T033, T034 可并行执行（不同测试文件）

**User Story 3阶段**（Phase 5）:
- T036, T037 可并行执行（同一方法的不同功能分支）
- T041, T042, T043, T044 可并行执行（不同测试用例）
- T046, T047 可并行执行（不同测试文件）

**Polish阶段**（Phase 6）:
- T048, T049, T050 可并行执行（不同文档文件）
- T052, T054, T055 可并行执行（不同测试文件）
- T058 可与文档更新任务并行执行

**跨用户故事并行**:
- Phase 3、Phase 4、Phase 5 可在Phase 2完成后同时开始（如果团队有多个开发者）

---

## Parallel Example: User Story 1 Implementation

```bash
# 并行启动User Story 1的核心实现任务（T011, T012）：
Task: "在result.py中实现__getitem__方法的整数索引支持"
Task: "在result.py中实现__getitem__方法的切片支持"

# 并行启动User Story 1的测试任务（T016, T017, T018, T019, T020）：
Task: "实现属性访问测试"
Task: "实现只读属性保护测试"
Task: "实现__len__测试"
Task: "实现__getitem__单个索引测试"
Task: "实现__getitem__切片测试"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. 完成 Phase 1: Setup（基础结构）
2. 完成 Phase 2: Foundational（核心基础设施，CRITICAL）
3. 完成 Phase 3: User Story 1（基础检测结果访问和操作）
4. **STOP and VALIDATE**: 独立测试User Story 1，验证Result对象创建、属性访问、索引操作、BaseORT集成全部正常
5. 部署/演示MVP版本

### Incremental Delivery

1. Setup + Foundational → 基础架构就绪
2. 添加 User Story 1 → 独立测试 → 部署/演示（MVP发布）
3. 添加 User Story 2 → 独立测试 → 部署/演示（可视化功能增强）
4. 添加 User Story 3 → 独立测试 → 部署/演示（过滤和统计功能增强）
5. 每个故事增加价值，不破坏之前的功能

### Parallel Team Strategy

如果有多个开发者:

1. 团队共同完成 Setup + Foundational
2. Foundational完成后：
   - 开发者 A: User Story 1（基础功能，优先级最高）
   - 开发者 B: User Story 2（可视化功能）
   - 开发者 C: User Story 3（过滤和统计）
3. 各故事独立完成和集成

---

## Notes

- **[P]** = 可并行执行任务（不同文件或无依赖关系）
- **[Story]** = 任务所属用户故事（US1, US2, US3）
- 每个用户故事应独立完成和测试
- 测试与实现可并行进行（非严格TDD，但需高覆盖率）
- 在每个Checkpoint处验证故事独立功能
- 避免：模糊任务、同一文件冲突、跨故事依赖破坏独立性

---

## Task Count Summary

- **Phase 1: Setup**: 4 tasks
- **Phase 2: Foundational**: 6 tasks
- **Phase 3: User Story 1**: 14 tasks (5 implementation + 9 testing)
- **Phase 4: User Story 2**: 11 tasks (5 implementation + 6 testing)
- **Phase 5: User Story 3**: 12 tasks (5 implementation + 7 testing)
- **Phase 6: Polish**: 11 tasks
- **Total**: 58 tasks

**Parallel Opportunities**: 约30个任务可并行执行（标记[P]），理论上可将开发周期压缩40-50%

**MVP Scope (User Story 1 Only)**: 24 tasks（Phase 1 + Phase 2 + Phase 3），覆盖核心功能
