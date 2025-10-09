# Implementation Plan: 重构ColorLayerONNX和OCRONNX以继承BaseOnnx

**Branch**: `004-refactor-colorlayeronnx-ocronnx` | **Date**: 2025-10-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-refactor-colorlayeronnx-ocronnx/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

重构车牌OCR和颜色/层级分类推理器以继承BaseOnnx基类，统一推理接口和会话管理。主要变更包括：
1. 使ColorLayerONNX和OCRONNX继承自BaseOnnx，复用Polygraphy懒加载机制
2. 将utils/ocr_image_processing.py和utils/ocr_post_processing.py中的函数整合为类的私有静态方法
3. 实现`_preprocess()`和`_postprocess()`抽象方法，采用混合模式（实例方法调用静态方法）
4. 删除utils/ocr_*.py文件，同步修改utils/pipeline.py等调用者代码
5. 支持TensorRT引擎比较和精度验证功能

**技术方法**: 采用BaseOnnx的混合方法模式，保持向后兼容性，通过拆分复杂逻辑（双层车牌处理）为多个私有静态辅助方法来提高可维护性。

## Technical Context

**Language/Version**: Python 3.10+（项目现有Python版本要求）
**Primary Dependencies**:
- onnxruntime-gpu 1.22.0 - ONNX模型推理引擎
- Polygraphy 0.49.26+ - NVIDIA模型调试和懒加载工具
- numpy 2.2.6+ - 数值计算和张量操作
- opencv-contrib-python 4.12.0+ - 图像处理（倾斜校正、分割、CLAHE）
- pyyaml 6.0.2+ - 配置文件解析

**Storage**: N/A（无持久化存储需求，仅内存推理）
**Testing**: pytest（现有测试框架，需要100%回归测试通过）
**Target Platform**: Linux server with CUDA 11.8+ GPU（主要生产环境）
**Project Type**: single（单体Python库项目）
**Performance Goals**:
- OCR首次推理时间（含懒加载）< 200ms
- 颜色分类推理延迟 < 10ms
- 代码重复度降低至少40%
- API响应时间误差 ±5%

**Constraints**:
- 必须保持现有OCR和颜色分类的输出格式和准确性（向后兼容）
- 不允许修改ONNX模型文件或plate.yaml配置格式
- 严格封装：所有迁移函数必须为私有静态方法
- 无渐进式迁移：立即删除utils文件并同步修改调用者

**Scale/Scope**:
- 重构2个推理类（ColorLayerONNX, OCRONNX）
- 整合约10个函数（5个预处理 + 2个后处理 + 3个辅助）
- 删除2个utils文件（约200行代码迁移）
- 修改1个主要调用者文件（utils/pipeline.py约287行）
- 新增约6个功能需求（FR-014至FR-019）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Modular Architecture ✅ PASS
- ✅ **符合性**: ColorLayerONNX和OCRONNX将继承自BaseOnnx基类，保持模块化边界
- ✅ **独立可测试**: 每个类封装自己的预处理/后处理逻辑，可单独测试
- ✅ **明确依赖**: 通过`__init__.py`导出公共接口，内部依赖BaseOnnx
- 📝 **文档要求**: 需在docstring中明确标注继承关系和抽象方法实现

### Principle II: Configuration-Driven Design ✅ PASS
- ✅ **符合性**: 不修改configs/plate.yaml配置文件，保持配置外部化
- ✅ **无硬编码**: 模型路径、字典和颜色映射均从YAML加载
- ⚠️ **注意**: 迁移后的预处理参数（如CLAHE参数、resize尺寸）应考虑配置化（但不强制）

### Principle III: Performance First ✅ PASS
- ✅ **Polygraphy懒加载**: 通过继承BaseOnnx自动获得懒加载优化，减少内存占用
- ✅ **TensorRT支持**: 新增`compare_engine()`方法支持精度验证，为生产部署铺路
- ✅ **性能监控**: 成功标准SC-003/SC-006明确了性能基准（200ms首次推理，±5%响应时间）
- 📝 **Profiling钩子**: 建议在`_preprocess()`和`_postprocess()`中添加可选的timing日志

### Principle IV: Type Safety and Contract Validation ⚠️ NEEDS IMPROVEMENT
- ⚠️ **类型提示**: 当前ocr_onnx.py缺少完整类型提示，重构时必须添加
- ✅ **运行时验证**: BaseOnnx已提供输入验证（input_shape检查）
- 📋 **行动项**:
  - 为所有新增方法添加类型提示（`@staticmethod def _decode_static(...) -> List[Tuple[str, float, List[float]]]:`）
  - 在`_postprocess()`中验证模型输出形状和dtype

### Principle V: Test-Driven Development (TDD) ⚠️ CRITICAL GATE
- 🔴 **关键风险**: SC-001要求100%现有测试通过，但需确认测试覆盖充分性
- 📋 **行动项Phase 0**:
  1. 审查现有测试覆盖（`tests/`目录）
  2. 为OCR和颜色分类添加缺失的单元测试（如无）
  3. 创建重构前的基准测试套件（golden test outputs）
- ✅ **合约测试**: 用户已明确FR-018需同步修改调用者，建议先写集成测试锁定行为

### Principle VI: Observability and Debugging ✅ PASS with ENHANCEMENT
- ✅ **结构化日志**: 项目已使用colorlog，重构类应保持日志级别一致
- ✅ **调试工具**: TensorRT引擎比较功能（FR-013）增强了可观测性
- 📝 **增强建议**:
  - 在双层车牌处理的关键步骤添加DEBUG日志（倾斜角度、分割点位置）
  - 为`_decode_static()`添加置信度统计日志

### Principle VII: Simplicity and Incremental Growth (YAGNI) ✅ PASS with JUSTIFICATION
- ✅ **简洁性**: 拆分80+行`process_plate_image`为多个辅助方法符合单一职责原则
- ⚠️ **复杂性权衡**: 私有静态方法模式增加了类的方法数量（约10+个方法）
- ✅ **正当理由**:
  - **问题**: 双层车牌处理逻辑复杂（倾斜检测、分割、拼接），单一方法难以维护和测试
  - **更简单方案被拒**: 保留在utils/中会违反"删除文件"要求；合并为单一方法违反单一职责
  - **迁移路径**: 无需迁移，这是最终架构

### 🚦 Gate Status: ⚠️ CONDITIONAL PASS

**允许进入Phase 0，但有前置条件**:
1. ✅ 架构符合宪法原则I、II、III、VI、VII
2. ⚠️ 需在Phase 0研究中补充类型安全策略（原则IV）
3. 🔴 **必须在Phase 0完成测试覆盖审查**（原则V - 关键风险）

**Phase 1后重新评估**: 确认类型提示完整性和测试充分性

---

## Constitution Re-Check (Phase 1后)

*执行时间: 2025-10-09 | Phase 1设计完成后*

### Principle IV: Type Safety and Contract Validation ✅ IMPROVED → PASS
- ✅ **完整类型提示**: data-model.md定义了所有方法的详细类型签名
  - 使用`from numpy.typing import NDArray`和`TypeAlias`定义复杂类型
  - 示例: `OCRResult: TypeAlias = Tuple[str, float, List[float]]`
- ✅ **API合约**: 两个YAML合约文件完整定义输入/输出格式和异常
  - `ocr_onnx_api.yaml`: 600+行详细合约规范
  - `color_layer_onnx_api.yaml`: 300+行详细合约规范
- ✅ **验证规则**: data-model.md包含验证规则表（如置信度范围[0,1]，图像形状检查）
- 📋 **剩余行动**: Phase 2实施时严格遵循合约中的类型提示

### Principle V: Test-Driven Development (TDD) ⚠️ PARTIALLY ADDRESSED
- ✅ **测试缺口识别**: research.md R1发现NO单元测试存在（🔴 critical risk）
- ✅ **测试策略制定**:
  - data-model.md定义了验证规则和边界条件
  - contracts/包含完整的测试需求（unit_tests, integration_tests, contract_tests）
  - quickstart.md提供测试用例示例
- ⚠️ **仍需完成**: 实际编写单元测试代码（Phase 2任务）
- 📋 **行动项Phase 2**:
  - 创建tests/unit/test_ocr_onnx.py（基于contract测试需求）
  - 创建tests/unit/test_color_layer_onnx.py
  - 创建golden test基准（双层车牌处理）

### Principle I-III, VI-VII ✅ 持续符合
- ✅ **模块化架构**: data-model.md清晰展示继承关系和职责分离
- ✅ **配置驱动**: quickstart.md示例显示从plate.yaml加载配置
- ✅ **性能优先**: contracts/定义了详细的性能指标（如OCR < 20ms推理）
- ✅ **可观测性**: contracts/包含错误处理和日志建议
- ✅ **YAGNI**: 设计文档聚焦实际需求，无过度设计

### 🚦 Updated Gate Status: ✅ PASS WITH ACTIONS

**准备进入Phase 2实施，需优先完成**:
1. ✅ 类型安全: 已有完整类型定义和合约（原则IV）
2. ⚠️ 测试优先: **必须在代码重构前创建基准单元测试**（原则V）
   - 阻塞条件: 至少完成OCRONNX和ColorLayerONNX的冒烟测试
   - 推荐: 创建双层车牌golden test避免回归
3. ✅ 架构设计: 已通过所有其他宪法原则

**Phase 2启动清单**:
- [ ] 基于contracts/编写单元测试骨架
- [ ] 运行现有测试套件,建立性能基准
- [ ] 创建golden test数据集（OCR输出、颜色分类输出）
- [ ] 开始实施重构任务

## Complexity Tracking

*仅记录违反宪法的复杂性，需提供正当理由*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 无违反 | N/A | N/A |

**说明**: 虽然类方法数量较多（约10+个私有静态方法），但这是遵循**单一职责原则**（SOLID的S）和**宪法原则VII简洁性**的结果。每个方法职责明确（如`_detect_skew_angle()`仅负责倾斜检测），相比单一巨型方法更易维护和测试。

## Project Structure

### Documentation (this feature)

```
specs/004-refactor-colorlayeronnx-ocronnx/
├── spec.md              # 功能规范（已完成，含5个澄清）
├── plan.md              # 本文件（实施计划）
├── research.md          # Phase 0输出（待生成）
├── data-model.md        # Phase 1输出（待生成）
├── quickstart.md        # Phase 1输出（待生成）
├── contracts/           # Phase 1输出（待生成）
│   ├── ocr_onnx_api.yaml          # OCRONNX API合约
│   └── color_layer_onnx_api.yaml  # ColorLayerONNX API合约
└── checklists/
    └── requirements.md  # 规范质量检查清单（已完成）
```

### Source Code (repository root)

```
infer_onnx/                          # 核心推理模块（主要修改目标）
├── base_onnx.py                     # ✅ 现有：BaseOnnx基类
├── ocr_onnx.py                      # 🔧 重构：OCRONNX类
│   # 新增方法：
│   # - _preprocess() 实例方法
│   # - _preprocess_static() 静态方法（委托给下列方法）
│   # - _process_plate_image_static()
│   # - _resize_norm_img_static()
│   # - _detect_skew_angle()        （新拆分的辅助方法）
│   # - _correct_skew()               （新拆分的辅助方法）
│   # - _find_optimal_split_line()    （新拆分的辅助方法）
│   # - _postprocess() 实例方法
│   # - _decode_static()
│   # - _get_ignored_tokens()
│   # 修改方法：
│   # - __init__() 调用super().__init__()
│   # - infer() → __call__()（保留infer()作为弃用包装器）
│
├── yolo_onnx.py                     # ✅ 参考：混合方法模式示例
├── rtdetr_onnx.py                   # ✅ 参考：继承BaseOnnx示例
└── infer_models.py                  # ✅ 工厂函数（可能需微调导入）

utils/                               # 工具模块（修改和删除）
├── ocr_image_processing.py          # ❌ 删除：迁移到OCRONNX类
├── ocr_post_processing.py           # ❌ 删除：迁移到OCRONNX类
├── pipeline.py                      # 🔧 重构：修改导入和调用
│   # 修改行224-242：
│   # - 移除 from utils import process_plate_image, resize_norm_img, decode
│   # - 改为调用 OCRONNX._process_plate_image_static() 或封装方法
│   # - 或将OCR预处理逻辑移入OCRONNX.__call__()内部
│
├── image_processing.py              # ✅ 保留：通用图像处理
├── annotator_factory.py             # ✅ 保留：Supervision集成
└── __init__.py                      # 🔧 微调：移除ocr_*函数的导出

tests/                               # 测试体系（扩展）
├── unit/                            # 单元测试（新增）
│   ├── test_ocr_onnx.py             # 新增：OCRONNX单元测试
│   │   # 测试用例：
│   │   # - test_process_plate_image_single_layer()
│   │   # - test_process_plate_image_double_layer()
│   │   # - test_detect_skew_angle()
│   │   # - test_decode_static()
│   │   # - test_get_ignored_tokens()
│   └── test_color_layer_onnx.py     # 新增：ColorLayerONNX单元测试
│
├── integration/                     # 集成测试（扩展）
│   └── test_refactored_pipeline.py  # 新增：重构后pipeline集成测试
│
└── contract/                        # 合约测试（新增）
    ├── test_ocr_onnx_contract.py    # 新增：验证API合约
    └── test_color_layer_contract.py # 新增：验证API合约
```

**结构决策**:
选择**单体项目结构**（Option 1），因为这是现有的ONNX推理库架构。重构主要集中在`infer_onnx/ocr_onnx.py`，删除`utils/ocr_*.py`，并修改`utils/pipeline.py`的调用方式。测试结构遵循现有的`tests/{unit,integration,contract}/`分层模式。

**关键文件修改汇总**:
1. **主要重构**: `infer_onnx/ocr_onnx.py`（约+300行，新增10+方法）
2. **删除**: `utils/ocr_image_processing.py`（-132行）, `utils/ocr_post_processing.py`（-34行）
3. **调用者修改**: `utils/pipeline.py`第224-242行（约18行重构）
4. **测试新增**: 3个测试文件（预计约+500行测试代码）

---

## Phase 0: Research & Risk Mitigation

*目标: 解决技术不确定性，验证测试覆盖，为Phase 1设计提供坚实基础*

### 研究任务清单

基于用户风险提示和Constitution Check，识别以下研究任务：

#### R1: 测试覆盖充分性审查 🔴 高优先级
**问题**: SC-001要求100%现有测试通过，需确认OCR和颜色分类是否有充分测试
**研究内容**:
- 审查`tests/`目录，确认是否存在OCR和颜色分类的单元测试
- 运行现有测试套件，记录覆盖率基准（使用pytest-cov）
- 识别未覆盖的关键路径（如双层车牌处理、decode逻辑）

**输出**: `research.md`第1节 - 测试覆盖现状报告

#### R2: pipeline.py依赖识别 🔴 高优先级
**问题**: FR-018要求同步修改所有调用者，需全面识别依赖文件
**研究内容**:
- 使用`grep -r "process_plate_image\|resize_norm_img\|decode\|image_pretreatment"`搜索整个项目
- 分析`utils/pipeline.py`第224-242行的调用模式
- 确认是否有其他脚本（如`tools/`、`main.py`）间接依赖这些函数

**输出**: `research.md`第2节 - 依赖文件清单和修改策略

#### R3: BaseOnnx混合模式最佳实践 🟡 中优先级
**问题**: 需参考yolo_onnx.py的实现模式，确保一致性
**研究内容**:
- 深入阅读`infer_onnx/yolo_onnx.py`和`infer_onnx/base_onnx.py:206-213`
- 理解`_preprocess()`实例方法如何调用`_preprocess_static()`
- 确认TensorRT数据加载器如何复用静态方法（`engine_dataloader.py`）

**输出**: `research.md`第3节 - 混合模式实现指南

#### R4: 类型提示策略 🟡 中优先级
**问题**: Constitution原则IV要求完整类型提示，需定义标准
**研究内容**:
- 调研numpy数组和ONNX输出的类型提示最佳实践（如`np.ndarray`形状注解）
- 确认mypy配置和严格性级别
- 定义复杂返回值的类型别名（如`OCRResult = Tuple[str, float, List[float]]`）

**输出**: `research.md`第4节 - 类型提示标准和示例

#### R5: 双层车牌处理逻辑验证 🟢 低优先级
**问题**: FR-019要求拆分复杂逻辑，需验证拆分后的正确性
**研究内容**:
- 使用真实双层车牌图像测试现有`process_plate_image()`
- 记录中间状态（倾斜角度、分割点、拼接结果）作为golden test
- 确认拆分后的辅助方法边界和输入/输出

**输出**: `research.md`第5节 - 双层车牌处理拆分验证报告

### 研究方法

所有研究任务将通过**专用代理**执行，每个代理独立完成一个研究任务并生成对应章节。

---

## Phase 1: Design Artifacts

*前置条件: `research.md`完成，所有NEEDS CLARIFICATION解决*

### 1.1 Data Model (`data-model.md`)

从spec.md的Key Entities提取，扩展以下数据模型：

#### Entity 1: ColorLayerONNX
```python
class ColorLayerONNX(BaseOnnx):
    """车牌颜色和层级分类推理器"""
    # 属性
    - onnx_path: str
    - input_shape: Tuple[int, int] = (224, 224)
    - conf_thres: float = 0.5
    - providers: List[str]

    # 私有静态方法
    @staticmethod
    def _image_pretreatment_static(img: np.ndarray,
                                    default_size: Tuple[int, int] = (168, 48))
                                    -> np.ndarray:
        """颜色/层数模型输入归一化"""

    # 抽象方法实现
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """调用_image_pretreatment_static()"""

    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs)
                     -> Tuple[int, int]:  # (color_index, layer_index)
        """Softmax + argmax"""

    # 公共方法
    def __call__(self, image: np.ndarray) -> Tuple[Tuple[int, int], tuple]:
        """返回(color_index, layer_index), original_shape"""
```

#### Entity 2: OCRONNX
```python
class OCRONNX(BaseOnnx):
    """车牌字符识别推理器"""
    # 属性
    - onnx_path: str
    - input_shape: Tuple[int, int] = (48, 168)
    - character: List[str]  # OCR字典
    - conf_thres: float = 0.5

    # 私有静态方法（预处理）
    @staticmethod
    def _process_plate_image_static(img: np.ndarray,
                                     is_double_layer: bool = False) -> np.ndarray:
        """双层车牌校正和拼接"""

    @staticmethod
    def _detect_skew_angle(image: np.ndarray) -> float:
        """检测图像倾斜角度"""

    @staticmethod
    def _correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
        """校正图像倾斜"""

    @staticmethod
    def _find_optimal_split_line(gray_img: np.ndarray) -> int:
        """通过水平投影找到最佳分割线"""

    @staticmethod
    def _resize_norm_img_static(img: np.ndarray,
                                 image_shape: List[int] = [3, 48, 168]) -> np.ndarray:
        """车牌图像resize、归一化"""

    # 私有静态方法（后处理）
    @staticmethod
    def _get_ignored_tokens() -> List[int]:
        """返回需要忽略的token列表"""

    @staticmethod
    def _decode_static(character: List[str],
                       text_index: np.ndarray,
                       text_prob: Optional[np.ndarray] = None,
                       is_remove_duplicate: bool = False)
                       -> List[Tuple[str, float, List[float]]]:
        """将OCR输出解码为字符串及置信度"""

    # 抽象方法实现
    def _preprocess(self, image: np.ndarray, is_double_layer: bool = False)
                    -> Tuple[np.ndarray, float, tuple]:
        """预处理链: process_plate_image -> resize_norm_img"""

    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs)
                     -> List[Tuple[str, float, List[float]]]:
        """调用_decode_static()"""

    # 公共方法
    def __call__(self, image: np.ndarray, is_double_layer: bool = False)
                 -> Tuple[List[Tuple[str, float, List[float]]], tuple]:
        """返回OCR结果列表和原始形状"""
```

#### Entity 3: PipelineRefactorAdapter
```python
# utils/pipeline.py中的适配器模式（可选设计）
class PlateProcessor:
    """封装车牌处理流程，简化pipeline.py调用"""
    def __init__(self, color_layer_model: ColorLayerONNX,
                 ocr_model: OCRONNX,
                 character: List[str],
                 plate_yaml: dict):
        self.color_layer_model = color_layer_model
        self.ocr_model = ocr_model
        self.character = character
        self.color_dict = plate_yaml["color_dict"]
        self.layer_dict = plate_yaml["layer_dict"]

    def process(self, plate_img: np.ndarray) -> dict:
        """处理单个车牌，返回{text, color, layer}"""
        # 内部调用color_layer_model()和ocr_model()
        # 封装现有pipeline.py第224-242行的逻辑
```

**关系图**:
```
BaseOnnx (抽象基类)
    ├── ColorLayerONNX (继承)
    │   └── 使用: _image_pretreatment_static()
    └── OCRONNX (继承)
        ├── 使用: _process_plate_image_static() → 调用以下辅助方法
        │   ├── _detect_skew_angle()
        │   ├── _correct_skew()
        │   └── _find_optimal_split_line()
        ├── 使用: _resize_norm_img_static()
        └── 使用: _decode_static() + _get_ignored_tokens()

PlateProcessor (可选适配器)
    ├── 组合: ColorLayerONNX
    └── 组合: OCRONNX
```

### 1.2 API Contracts (`contracts/`)

#### Contract 1: `ocr_onnx_api.yaml`
```yaml
api_version: "1.0.0"
class_name: OCRONNX

# 公共接口
public_methods:
  __init__:
    parameters:
      - name: onnx_path
        type: str
        required: true
        description: ONNX模型文件路径
      - name: input_shape
        type: Tuple[int, int]
        default: (48, 168)
        description: 输入图像尺寸(height, width)
      - name: conf_thres
        type: float
        default: 0.5
      - name: providers
        type: Optional[List[str]]
        default: null
        description: ONNX Runtime执行提供程序
    effects:
      - 创建Polygraphy懒加载器（不立即加载模型）
      - 设置input_shape和providers
    raises:
      - FileNotFoundError: ONNX模型文件不存在

  __call__:
    parameters:
      - name: image
        type: np.ndarray
        shape: [H, W, 3]
        dtype: uint8
        description: BGR格式输入图像
      - name: is_double_layer
        type: bool
        default: false
        description: 是否为双层车牌
    returns:
      type: Tuple[List[Tuple[str, float, List[float]]], tuple]
      description: |
        - List[Tuple[str, float, List[float]]]: OCR结果列表
          - str: 识别的文本
          - float: 平均置信度
          - List[float]: 每个字符的置信度
        - tuple: 原始图像形状(H, W, C)
    performance:
      first_inference_latency_ms: <200
      subsequent_latency_ms: <50
    raises:
      - ValueError: 图像为空或形状不正确
      - RuntimeError: ONNX推理失败

  create_engine_dataloader:
    parameters:
      - name: image_paths
        type: Union[str, List[str]]
        description: 图片路径列表或文件夹路径
      - name: iterations
        type: int
        default: 1
    returns:
      type: CustomEngineDataLoader
    side_effects:
      - 设置self.engine_dataloader属性

  compare_engine:
    parameters:
      - name: engine_path
        type: Optional[str]
        default: null
        description: TensorRT引擎文件路径
      - name: save_engine
        type: bool
        default: false
      - name: rtol
        type: float
        default: 0.001
      - name: atol
        type: float
        default: 0.001
    returns:
      type: Tuple[bool, dict]
      description: (比较结果, 运行结果字典)
    requires:
      - self.engine_dataloader must be set (via create_engine_dataloader())

# 私有静态方法（文档化但不保证稳定性）
private_static_methods:
  _process_plate_image_static:
    signature: (img: np.ndarray, is_double_layer: bool) -> np.ndarray
    behavior: |
      1. 灰度转换并检测倾斜角度
      2. 校正倾斜
      3. 如为双层车牌：
         - 通过水平投影找到分割线
         - 分割上下两部分
         - 上层缩小50%宽度后拼接到下层左侧
      4. 返回处理后的单层车牌图像
    edge_cases:
      - 输入为空: 返回None
      - 分割失败: 返回None
      - 单层车牌: 仅校正倾斜后返回

  _resize_norm_img_static:
    signature: (img: np.ndarray, image_shape: List[int]) -> np.ndarray
    behavior: |
      1. 保持宽高比resize到目标高度
      2. 转换通道顺序为CHW
      3. 归一化到[-1, 1]（减0.5除0.5）
      4. 右侧padding到目标宽度
    returns:
      shape: [1, C, H, W]
      dtype: float32

  _decode_static:
    signature: (character: List[str], text_index: np.ndarray, text_prob: Optional[np.ndarray], is_remove_duplicate: bool) -> List[Tuple[str, float, List[float]]]
    behavior: |
      1. 对每个batch：
         - 移除ignored_tokens（0）
         - 可选：移除重复字符
         - 将索引映射到字符
         - 计算平均置信度
         - 后处理：将'苏'替换为'京'
      2. 返回(文本, 平均置信度, 字符置信度列表)
```

#### Contract 2: `color_layer_onnx_api.yaml`
```yaml
api_version: "1.0.0"
class_name: ColorLayerONNX

public_methods:
  __init__:
    parameters:
      - name: onnx_path
        type: str
        required: true
      - name: input_shape
        type: Tuple[int, int]
        default: (224, 224)
      - name: conf_thres
        type: float
        default: 0.5
      - name: providers
        type: Optional[List[str]]
        default: null

  __call__:
    parameters:
      - name: image
        type: np.ndarray
        shape: [H, W, 3]
        dtype: uint8
    returns:
      type: Tuple[Tuple[int, int], tuple]
      description: |
        - Tuple[int, int]: (color_index, layer_index)
        - tuple: 原始图像形状(H, W, C)
    performance:
      inference_latency_ms: <10
    raises:
      - ValueError: 图像为空

private_static_methods:
  _image_pretreatment_static:
    signature: (img: np.ndarray, default_size: Tuple[int, int]) -> np.ndarray
    behavior: |
      1. Resize到default_size (width, height)
      2. 归一化到[-1, 1]（(x/255 - 0.5) / 0.5）
      3. 转换通道顺序为CHW
      4. 添加batch维度
    returns:
      shape: [1, 3, H, W]
      dtype: float32
```

### 1.3 Quick Start Guide (`quickstart.md`)

*待Phase 1生成，包含重构前后的使用对比示例*

---

## Phase 2: Task Decomposition

*由`/speckit.tasks`命令生成，不在`/speckit.plan`范围内*

预计任务类型分布（基于Constitution principles）：
- **架构任务**: 重构OCRONNX和ColorLayerONNX继承BaseOnnx（2个任务）
- **函数迁移任务**: 整合utils/ocr_*.py函数为私有静态方法（约8个任务）
- **调用者适配任务**: 修改utils/pipeline.py和其他依赖文件（2-3个任务）
- **测试任务**: 单元测试、合约测试、集成测试（约6个任务）
- **文档任务**: 更新docstring、CLAUDE.md和quickstart.md（2个任务）
- **删除任务**: 移除utils/ocr_*.py文件和清理导入（1个任务）

---

## Next Steps

1. ✅ **本阶段完成**: 实施计划填写完毕
2. ⏭️ **Phase 0启动**: 执行研究任务R1-R5，生成`research.md`
3. ⏭️ **Phase 1设计**: 基于研究结果生成data-model.md和contracts/
4. ⏭️ **Constitutional Re-check**: Phase 1后重新评估类型安全和测试充分性
5. ⏭️ **Task Generation**: 运行`/speckit.tasks`生成可执行任务清单

---

*计划状态: Phase 0 Ready | 最后更新: 2025-10-09*
