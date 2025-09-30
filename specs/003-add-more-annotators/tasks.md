# Tasks: 添加更多Supervision Annotators类型

**Input**: Design documents from `/home/tyjt/桌面/onnx_vehicle_plate_recognition/specs/003-add-more-annotators/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.10+, supervision>=0.26.0, pytest
   → Structure: Single project (utils/, tests/)
2. Load design documents:
   → data-model.md: 5 entities (AnnotatorType, Config, Factory, Pipeline, Preset)
   → contracts/: 1 API contract (annotator_api.yaml)
   → research.md: 13 annotator types, conflict detection, presets
3. Generate tasks by category:
   → Setup: dependencies, linting
   → Tests: 3 contract tests, 13 integration tests, performance tests
   → Core: enums, Factory, Pipeline, presets (配置类改用字典模式)
   → Polish: docs, cleanup
4. Apply task rules:
   → Different files = [P] for parallel
   → Tests before implementation (TDD)
5. Numbered tasks: T001-T030 (其中T006-T007已跳过，设计变更)
6. SUCCESS: 28/30 tasks executed (2 skipped by design decision)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All paths relative to repository root: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/`

---

## Phase 3.1: Setup & Dependencies

### T001 [X] [P] 验证项目依赖和环境
**File**: `requirements.txt`, `pyproject.toml`
**Description**:
- 验证`supervision>=0.26.0`已安装
- 验证`pytest`, `pytest-benchmark`可用
- 检查Python 3.10+环境
- 运行`pip list | grep supervision`确认版本

**Acceptance**:
- `supervision.__version__ >= 0.26.0`
- pytest可执行

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### T002 [X] [P] Contract test: AnnotatorFactory.create()
**File**: `tests/contract/test_annotator_factory_contract.py`
**Description**:
编写合约测试验证`AnnotatorFactory.create()`接口：
- 测试所有13种`AnnotatorType`可创建
- 测试配置参数验证（类型错误应raise TypeError）
- 测试返回类型为`sv.BaseAnnotator`子类
- 测试未知类型应raise ValueError

**Expected**: 测试失败（factory未实现）

**Test Outline**:
```python
def test_create_all_annotator_types():
    for ann_type in AnnotatorType:
        annotator = AnnotatorFactory.create(ann_type, default_config)
        assert isinstance(annotator, sv.BaseAnnotator)

def test_invalid_config_raises_error():
    with pytest.raises(TypeError):
        AnnotatorFactory.create(AnnotatorType.ROUND_BOX, {'invalid': 'param'})
```

---

### T003 [X] [P] Contract test: AnnotatorPipeline组合和渲染
**File**: `tests/contract/test_annotator_pipeline_contract.py`
**Description**:
编写合约测试验证`AnnotatorPipeline`接口：
- 测试`.add()`返回self支持链式调用
- 测试`.annotate()`返回np.ndarray且shape不变
- 测试空pipeline返回原图像副本
- 测试`.check_conflicts()`返回警告列表

**Expected**: 测试失败（pipeline未实现）

**Test Outline**:
```python
def test_pipeline_builder_pattern():
    pipeline = AnnotatorPipeline()
    result = pipeline.add(AnnotatorType.BOX, {})
    assert result is pipeline  # Builder pattern

def test_pipeline_annotate_preserves_shape():
    pipeline = AnnotatorPipeline().add(AnnotatorType.BOX, {})
    result = pipeline.annotate(test_image, test_detections)
    assert result.shape == test_image.shape
```

---

### T004 [X] [P] Contract test: VisualizationPreset加载和创建
**File**: `tests/contract/test_preset_loading_contract.py`
**Description**:
编写合约测试验证预设场景加载：
- 测试所有5种预设(`Presets.*`)可从YAML加载
- 测试未知预设raise ValueError
- 测试`.create_pipeline()`返回AnnotatorPipeline实例
- 测试预设包含`name`, `description`, `annotators`属性

**Expected**: 测试失败（preset加载器未实现）

**Test Outline**:
```python
def test_load_all_presets():
    for preset_name in [Presets.STANDARD, Presets.LIGHTWEIGHT, ...]:
        preset = VisualizationPreset.from_yaml(preset_name)
        assert preset.name
        assert preset.annotators

def test_preset_creates_pipeline():
    preset = VisualizationPreset.from_yaml(Presets.STANDARD)
    pipeline = preset.create_pipeline()
    assert isinstance(pipeline, AnnotatorPipeline)
```

---

## Phase 3.3: Core Implementation - Models & Enums (ONLY after tests are failing)

### T005 [X] [P] 实现AnnotatorType枚举
**File**: `utils/annotator_factory.py`
**Description**:
实现13种annotator类型枚举：
```python
class AnnotatorType(Enum):
    BOX = "box"
    RICH_LABEL = "rich_label"
    ROUND_BOX = "round_box"
    BOX_CORNER = "box_corner"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    ELLIPSE = "ellipse"
    DOT = "dot"
    COLOR = "color"
    BACKGROUND_OVERLAY = "background_overlay"
    HALO = "halo"
    PERCENTAGE_BAR = "percentage_bar"
    BLUR = "blur"
    PIXELATE = "pixelate"
```

**Acceptance**:
- 枚举包含13个值
- 可通过`AnnotatorType.ROUND_BOX`访问

---

### T006 [SKIPPED] [P] 实现BaseAnnotatorConfig和配置类(1-7)
**Status**: ❌ **设计变更 - 不实现**
**File**: `utils/annotator_configs.py` (未创建)
**Original Description**:
实现配置基类和前7个annotator配置类（略）

**Design Decision**:
采用**字典配置模式**替代dataclass配置类，理由：
1. ✅ **简洁性**：所有100+测试/代码使用`Dict[str, Any]`
2. ✅ **灵活性**：无需为13种annotator定义13个配置类
3. ✅ **Python惯用**：符合`**kwargs`和supervision库的设计风格
4. ✅ **可维护性**：减少13个配置类的维护负担

**Actual Implementation**:
- 使用字典直接传递配置参数
- Factory自动处理字典到annotator参数的映射
- 类型提示使用字符串引用：`Union[Dict[str, Any], 'BaseAnnotatorConfig']`

**Example**:
```python
# 实际使用方式（字典配置）
config = {'thickness': 2, 'roundness': 0.3}
annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, config)

# 或直接传递
pipeline.add(AnnotatorType.ROUND_BOX, {'thickness': 2, 'roundness': 0.3})
```

---

### T007 [SKIPPED] [P] 实现配置类(8-13)和特殊配置
**Status**: ❌ **设计变更 - 不实现**
**File**: `utils/annotator_configs.py` (未创建)
**Original Description**:
实现剩余6个annotator配置类和ConfigType联合（略）

**Design Decision**:
同T006，采用字典配置模式。无需创建配置类或ConfigType联合类型。

**Actual Implementation**:
所有配置通过字典传递，Factory内部自动验证和转换：
```python
# Factory内部实现 (annotator_factory.py:72-75)
if hasattr(config, '__dict__'):
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
else:
    config_dict = dict(config) if config else {}  # 字典模式
```

**Impact on Dependencies**:
- T008-T015: ✅ 不受影响，Factory已实现字典配置模式
- T016-T028: ✅ 所有测试使用字典配置，正常工作

---

## Phase 3.4: Core Implementation - Factory & Pipeline

### T008 [X] 实现AnnotatorFactory核心逻辑
**File**: `utils/annotator_factory.py`
**Description**:
实现`AnnotatorFactory.create()`方法：
- 实现类型dispatch映射（type → creator函数）
- 实现13个私有creator方法（`_create_round_box()`, etc.）
- 每个creator从config提取参数并创建sv.Annotator实例
- 处理config为dict或对象两种情况
- 参数验证和错误处理

参考research.md中的API签名

**Depends on**: T005, T006, T007

**Acceptance**:
- T002合约测试通过
- 所有13种类型可创建
- 错误配置正确raise异常

---

### T009 [X] 实现AnnotatorFactory参数验证
**File**: `utils/annotator_factory.py`
**Description**:
实现`AnnotatorFactory.validate_config()`静态方法：
- 验证config包含必需参数
- 验证参数类型正确
- 验证参数值范围（如roundness: 0-1.0）
- 返回验证结果和错误列表

**Depends on**: T008

**Acceptance**:
- 无效配置返回False和错误信息
- 有效配置返回True

---

### T010 [X] 实现AnnotatorPipeline.add()
**File**: `utils/annotator_factory.py` (或独立`utils/annotator_pipeline.py`)
**Description**:
实现`AnnotatorPipeline.add()`方法：
- 支持传入annotator实例或AnnotatorType
- 如果传入类型，调用Factory创建实例
- 存储annotator到内部列表
- 返回self支持Builder模式

**Depends on**: T008

**Acceptance**:
- T003部分测试通过
- 支持链式调用

---

### T011 [X] 实现AnnotatorPipeline.annotate()渲染逻辑
**File**: `utils/annotator_factory.py`
**Description**:
实现`AnnotatorPipeline.annotate()`方法：
- 复制输入图像（避免修改原图）
- 按顺序遍历所有annotator
- 每个annotator的输出作为下一个的输入
- 返回最终标注图像

**Depends on**: T010

**Acceptance**:
- T003所有测试通过
- 多annotator组合正确渲染

---

### T012 [X] 实现冲突检测和警告机制
**File**: `utils/annotator_factory.py`
**Description**:
实现`AnnotatorPipeline.check_conflicts()`方法：
- 定义冲突对列表（参考research.md）
- 检查pipeline中的annotator类型组合
- 生成警告消息
- 使用logging.warning输出日志

冲突对示例:
- (COLOR, BLUR), (COLOR, PIXELATE), (BOX, ROUND_BOX)

**Depends on**: T010

**Acceptance**:
- 冲突组合生成警告
- 非冲突组合不生成警告

---

## Phase 3.5: Configuration & Presets

### T013 [X] [P] 创建visualization_presets.yaml配置文件
**File**: `configs/visualization_presets.yaml`
**Description**:
创建YAML配置文件定义5种预设场景：
- `standard`: BoxAnnotator + RichLabelAnnotator
- `lightweight`: DotAnnotator + LabelAnnotator
- `privacy`: BoxAnnotator + BlurAnnotator (仅车牌)
- `debug`: RoundBoxAnnotator + PercentageBarAnnotator + RichLabelAnnotator
- `high_contrast`: ColorAnnotator + BackgroundOverlayAnnotator

参考research.md中的预设设计

**Acceptance**:
- YAML格式正确
- 包含5种预设
- 每个预设包含name, description, annotators字段

---

### T014 [X] [P] 实现VisualizationPreset模型和加载器
**File**: `utils/visualization_preset.py`
**Description**:
实现预设场景模型：
- `VisualizationPreset` dataclass (name, description, annotators)
- `VisualizationPreset.from_yaml()`类方法加载YAML
- `.create_pipeline()`方法创建AnnotatorPipeline
- `Presets`常量类定义5种预设名称

**Depends on**: T013, T010

**Acceptance**:
- T004所有测试通过
- 5种预设可正确加载和创建pipeline

---

### T015 [X] 扩展supervision_config.py集成新annotator
**File**: `utils/supervision_config.py`
**Description**:
扩展现有配置文件添加便捷函数：
- `get_default_annotator_config(type)`: 为每种类型返回默认配置
- 保留现有`create_box_annotator()`和`create_rich_label_annotator()`不变
- 添加文档字符串说明新功能

**Depends on**: T008

**Acceptance**:
- 现有函数完全兼容
- 新增13种默认配置getter

---

## Phase 3.6: Integration Tests (验证端到端功能)

### T016 [X] [P] 集成测试: RoundBoxAnnotator
**File**: `tests/integration/test_round_box_integration.py`
**Description**:
测试圆角边框annotator端到端功能：
- 创建测试图像和检测数据
- 使用Factory创建RoundBoxAnnotator
- 验证标注后图像shape不变
- 验证图像内容已修改（非原图副本）
- 测试不同roundness值效果

**Depends on**: T008

**Acceptance**:
- 测试通过
- 圆角边框正确绘制

---

### T017 [X] [P] 集成测试: BoxCornerAnnotator
**File**: `tests/integration/test_box_corner_integration.py`
**Description**:
测试角点标注annotator：
- 验证仅绘制四个角点
- 测试不同corner_length值
- 验证密集检测场景不重叠

**Depends on**: T008

---

### T018 [X] [P] 集成测试: PercentageBarAnnotator
**File**: `tests/integration/test_percentage_bar_integration.py`
**Description**:
测试置信度条形图annotator：
- 验证条形图按confidence绘制
- 测试custom_values参数
- 验证position配置生效

**Depends on**: T008

---

### T019 [X] [P] 集成测试: 几何标记Annotators (Circle/Triangle/Ellipse)
**File**: `tests/integration/test_geometric_annotators.py`
**Description**:
测试三种几何标记annotator：
- CircleAnnotator圆形绘制
- TriangleAnnotator三角标记
- EllipseAnnotator椭圆绘制
- 验证position参数生效

**Depends on**: T008

---

### T020 [X] [P] 集成测试: DotAnnotator
**File**: `tests/integration/test_dot_annotator.py`
**Description**:
测试点标注annotator：
- 验证点绘制在检测中心
- 测试不同radius和position
- 验证outline效果

**Depends on**: T008

---

### T021 [X] [P] 集成测试: ColorAnnotator和BackgroundOverlayAnnotator
**File**: `tests/integration/test_fill_annotators.py`
**Description**:
测试区域填充和背景叠加annotator：
- ColorAnnotator透明填充效果
- BackgroundOverlayAnnotator背景变暗
- 测试opacity参数
- 测试force_box模式

**Depends on**: T008

---

### T022 [X] [P] 集成测试: BlurAnnotator和PixelateAnnotator
**File**: `tests/integration/test_privacy_annotators.py`
**Description**:
测试隐私保护annotator：
- BlurAnnotator模糊效果验证
- PixelateAnnotator像素化效果
- 测试kernel_size和pixel_size参数
- 验证仅指定区域受影响

**Depends on**: T008

---

### T023 [X] [P] 集成测试: HaloAnnotator
**File**: `tests/integration/test_halo_annotator.py`
**Description**:
测试光晕效果annotator：
- 验证光晕围绕检测对象
- 测试kernel_size和opacity
- 验证与其他annotator组合效果

**Depends on**: T008

---

### T024 [X] [P] 集成测试: 5种预设场景
**File**: `tests/integration/test_preset_scenarios.py`
**Description**:
测试所有预设场景端到端：
- 加载并应用standard预设
- 加载并应用lightweight预设
- 加载并应用privacy预设
- 加载并应用debug预设
- 加载并应用high_contrast预设
- 验证每个预设渲染成功

**Depends on**: T014

**Test Outline**:
```python
@pytest.mark.parametrize("preset_name", [
    Presets.STANDARD,
    Presets.LIGHTWEIGHT,
    Presets.PRIVACY,
    Presets.DEBUG,
    Presets.HIGH_CONTRAST
])
def test_preset_rendering(preset_name, test_image, test_detections):
    preset = VisualizationPreset.from_yaml(preset_name)
    pipeline = preset.create_pipeline()
    result = pipeline.annotate(test_image, test_detections)
    assert result.shape == test_image.shape
```

---

### T025 [X] 集成测试: Annotator组合和渲染顺序
**File**: `tests/integration/test_annotator_pipeline.py`
**Description**:
测试annotator组合场景：
- 测试多annotator pipeline渲染
- 验证渲染顺序影响结果
- 测试冲突警告生成
- 测试复杂组合（5+个annotator）

**Depends on**: T011, T012

---

## Phase 3.7: Performance & Benchmarks

### T026 [X] [P] 实现annotator性能基准测试框架
**File**: `tests/performance/test_annotator_benchmark.py`
**Description**:
创建pytest-benchmark性能测试框架：
- 定义测试fixtures (640x640图像, 20个检测对象)
- 定义基准测试装饰器配置
- 创建性能报告生成工具

**Test Fixtures**:
```python
@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def test_detections():
    return sv.Detections(
        xyxy=np.random.rand(20, 4) * 640,
        confidence=np.random.rand(20),
        class_id=np.random.randint(0, 2, 20)
    )
```

**Depends on**: T008

---

### T027 [X] 基准测试所有13种Annotator
**File**: `tests/performance/test_annotator_benchmark.py` (扩展T026)
**Description**:
为每种annotator添加基准测试：
- 使用`@pytest.mark.benchmark(group="annotators")`
- 测试单annotator渲染时间
- 记录mean, std, min, max统计
- 对比不同参数配置性能

**示例测试**:
```python
@pytest.mark.benchmark(group="annotators")
def test_round_box_performance(benchmark, test_image, test_detections):
    annotator = AnnotatorFactory.create(
        AnnotatorType.ROUND_BOX,
        {'thickness': 2, 'roundness': 0.3}
    )
    result = benchmark(annotator.annotate, test_image, test_detections)
    assert result.shape == test_image.shape
```

**Depends on**: T026

**Acceptance**:
- 13个基准测试完成
- 生成性能报告

运行命令: `pytest tests/performance/ --benchmark-only --benchmark-autosave`

---

### T028 [X] 生成性能报告和优化建议
**File**: `specs/003-add-more-annotators/performance_report.md`
**Description**:
分析基准测试结果并生成报告：
- 汇总所有annotator性能数据
- 识别最快和最慢的annotator
- 提供性能优化建议
- 记录性能baseline供未来对比

**Depends on**: T027

**Acceptance**:
- 性能报告包含所有13种annotator数据
- 提供清晰的性能对比表格

---

## Phase 3.8: Polish & Documentation

### T029 [X] [P] 更新quickstart.md实际示例
**File**: `specs/003-add-more-annotators/quickstart.md`
**Description**:
更新快速入门指南使用实际代码：
- 替换占位示例为真实可运行代码
- 添加实际输出截图或描述
- 验证所有示例可执行
- 添加常见问题解答

**Depends on**: T008-T015 (核心功能完成)

**Acceptance**:
- 所有示例代码可执行
- 文档清晰易懂

---

### T030 [X] [P] 更新CLAUDE.md和项目文档
**File**: `CLAUDE.md`, `specs/003-add-more-annotators/CLAUDE.md`
**Description**:
更新AI助手上下文文档：
- 记录新增13种annotator使用方法
- 更新utils/模块文档
- 添加性能特征说明
- 更新测试覆盖状态

**Depends on**: T008-T028 (所有功能完成)

**Acceptance**:
- CLAUDE.md包含最新功能说明
- 模块级文档完整更新

---

## Dependencies

### 关键依赖链
```
Setup (T001) → Tests (T002-T004) → Models (T005) → Factory (T008-T009)
                                        ↓
                                   [T006-T007 SKIPPED]
                                        ↓
Factory → Pipeline (T010-T012) → Config (T013-T015) → Integration (T016-T025)
                                                              ↓
                                    Performance (T026-T028) → Docs (T029-T030)
```

**Note**: T006和T007已跳过（设计变更），不影响依赖链

### 详细依赖
- **T002-T004** (Tests): 无依赖，可立即开始（TDD）
- **T005** (AnnotatorType Enum): 无依赖，可与Tests并行
- **T006-T007** (Config Classes): ❌ **SKIPPED** - 采用字典配置模式
- **T008** (Factory): 依赖T005（不依赖T006/T007）
- **T009**: 依赖T008
- **T010-T012** (Pipeline): 依赖T008
- **T013**: 无依赖（独立YAML文件）
- **T014**: 依赖T013, T010
- **T015**: 依赖T008
- **T016-T023** (Integration): 依赖T008
- **T024**: 依赖T014
- **T025**: 依赖T011, T012
- **T026**: 依赖T008
- **T027**: 依赖T026
- **T028**: 依赖T027
- **T029-T030** (Docs): 依赖所有核心功能

---

## Parallel Execution Examples

### Wave 1: Tests + Models (可并行)
```bash
# 启动3个并行任务（不同文件）
Task: "Contract test AnnotatorFactory in tests/contract/test_annotator_factory_contract.py"
Task: "Contract test Pipeline in tests/contract/test_annotator_pipeline_contract.py"
Task: "Implement AnnotatorType enum in utils/annotator_factory.py"
```

### Wave 2: Config Classes (已跳过)
```bash
# T006-T007已跳过 - 采用字典配置模式
# 无需创建utils/annotator_configs.py
# 直接进入Factory和Pipeline实现
```

### Wave 3: Integration Tests (可并行)
```bash
# 所有集成测试独立文件，可完全并行
Task: "Integration test RoundBox in tests/integration/test_round_box_integration.py"
Task: "Integration test BoxCorner in tests/integration/test_box_corner_integration.py"
Task: "Integration test PercentageBar in tests/integration/test_percentage_bar_integration.py"
Task: "Integration test Geometric in tests/integration/test_geometric_annotators.py"
# ... 更多集成测试
```

---

## Notes

### 并行执行规则
- **[P]标记任务**: 不同文件且无依赖，可并行执行
- **同文件任务**: 顺序执行避免冲突（如T006/T007虽标记[P]但应协调）
- **依赖任务**: 必须等待依赖完成

### TDD原则
- ⚠️ **T002-T004必须先写且失败**，才能开始T008-T015实现
- 每个实现任务应使对应测试通过

### 提交策略
- 每个任务完成后提交一次
- 测试任务提交消息: `test: add contract test for AnnotatorFactory`
- 实现任务提交消息: `feat: implement AnnotatorFactory.create()`
- 文档任务提交消息: `docs: update quickstart with real examples`

### 任务时长预估
- **Setup (T001)**: 0.5小时
- **Tests (T002-T004)**: 每个1小时，共3小时
- **Models (T005-T007)**: 每个1小时，共3小时
- **Core (T008-T015)**: 每个1.5-2小时，共14小时
- **Integration (T016-T025)**: 每个0.5-1小时，共8小时
- **Performance (T026-T028)**: 共3小时
- **Docs (T029-T030)**: 共2小时
- **总计**: ~33.5小时 (约4-5个工作日)

---

## Validation Checklist
*GATE: Verified before task execution*

- [x] All contracts (annotator_api.yaml) have corresponding tests (T002-T004)
- [x] All entities (AnnotatorType, Configs, Factory, Pipeline, Preset) have creation tasks
- [x] All tests (T002-T004, T016-T025) come before implementation
- [x] Parallel tasks [P] truly independent (different files checked)
- [x] Each task specifies exact file path
- [x] No [P] task modifies same file as another [P] task
- [x] TDD principle enforced (tests T002-T004 before impl T008+)

---

**Generated**: 2025-09-30
**Last Updated**: 2025-09-30 (T006/T007设计变更)
**Execution Status**: ✅ 28/30 tasks completed (T006/T007 skipped by design)
**Design Decision**: 采用字典配置模式，无需创建13个配置类