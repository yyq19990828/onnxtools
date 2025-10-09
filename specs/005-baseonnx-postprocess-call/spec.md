# Feature Specification: BaseOnnx抽象方法强制实现

**Feature Branch**: `005-baseonnx-postprocess-call`
**Created**: 2025-10-09
**Status**: Draft
**Input**: User description: "优化BaseOnnx类, _postprocess __call__ _preprocess_static 函数都需要子类实现, 基类不实现且raise为实施的错. 然后调整所有继承该类的子类"

## Clarifications

### Session 2025-10-09

- Q: `__call__`方法的处理策略 → A: 保持`__call__`为具体方法(模板方法模式),不标记@abstractmethod,只优化内部实现(解耦、删除旧逻辑)
- Q: "旧版本分支逻辑"的清理范围 → A: 先分析测试覆盖,仅删除测试中未覆盖的分支逻辑
- Q: `__call__`方法的解耦策略 → A: 提取3个主要阶段方法: `_prepare_inference()`, `_execute_inference()`, `_finalize_inference()`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 强制子类实现核心方法 (Priority: P1)

开发者在创建新的ONNX推理模型类时,必须实现所有核心抽象方法,否则在实例化或运行时立即收到明确的错误提示,避免运行时才发现缺失实现导致的不可预测行为。

**Why this priority**: 这是类型系统最基础的约束,确保所有子类都遵循相同的接口契约,是整个推理框架稳定性的基石。

**Independent Test**: 可以通过创建一个不完整的子类并尝试实例化来验证,应该在类定义时或实例化时抛出NotImplementedError异常。

**Acceptance Scenarios**:

1. **Given** 开发者创建了一个继承BaseOnnx但未实现_postprocess方法的子类, **When** 尝试调用该类实例的__call__方法进行推理, **Then** 系统抛出NotImplementedError并明确指出_postprocess方法未实现
2. **Given** 开发者创建了一个继承BaseOnnx但未实现_preprocess_static静态方法的子类, **When** 尝试调用该类实例进行推理, **Then** 系统抛出NotImplementedError并明确指出_preprocess_static方法未实现
3. **Given** BaseOnnx基类存在对应抽象方法的装饰器(@abstractmethod), **When** 开发者尝试实例化未完整实现的子类, **Then** Python解释器在实例化时抛出TypeError异常
4. **Given** BaseOnnx的__call__方法完成优化, **When** 任何子类调用__call__进行推理, **Then** 推理流程正常执行且代码结构清晰(无冗余旧版本分支逻辑)

---

### User Story 2 - 现有子类代码完整性验证 (Priority: P1)

确保当前所有继承BaseOnnx的子类(YoloOnnx, RTDETROnnx, RFDETROnnx, ColorLayerONNX, OCRONNX)都正确实现了所有必需的抽象方法,在重构后能够正常工作。

**Why this priority**: 保证重构不会破坏现有功能,是生产系统稳定性的基本要求。如果现有子类缺少实现,需要在重构过程中补全。

**Independent Test**: 运行现有的集成测试套件(tests/integration/),所有测试应继续通过,验证每个子类都能成功实例化和执行推理。

**Acceptance Scenarios**:

1. **Given** YoloOnnx/RTDETROnnx/RFDETROnnx/ColorLayerONNX/OCRONNX子类已完整实现所有抽象方法(_postprocess和_preprocess_static), **When** 执行现有的集成测试, **Then** 所有测试通过,功能表现与重构前一致
2. **Given** 某个子类缺少_postprocess或_preprocess_static的实现, **When** 重构过程中发现该问题, **Then** 开发者在重构PR中补全缺失的实现并通过测试验证
3. **Given** 所有子类都已正确实现抽象方法,且BaseOnnx的__call__完成优化, **When** 使用各种输入数据执行推理, **Then** 所有模型返回正确的预测结果,性能指标不降低

---

### User Story 3 - 明确错误提示和开发者体验 (Priority: P2)

当开发者违反抽象方法契约时,错误消息应该清晰指出哪个方法未实现、为什么需要实现,以及如何修复问题,减少调试时间。

**Why this priority**: 良好的开发者体验能显著提高团队生产力,虽然不如核心功能关键,但对长期维护很重要。

**Independent Test**: 创建测试用例验证NotImplementedError的错误消息包含方法名、类名和简要说明,可以独立于实际模型运行来测试。

**Acceptance Scenarios**:

1. **Given** 开发者创建了缺少_postprocess实现的子类MyDetector, **When** 调用MyDetector实例进行推理时抛出异常, **Then** 错误消息明确指出"MyDetector._postprocess() must be implemented by subclass. This method is responsible for processing model output."
2. **Given** NotImplementedError错误消息遵循统一格式, **When** 任何抽象方法未实现, **Then** 错误消息包含:类名.方法名、必须由子类实现的说明、该方法的职责描述
3. **Given** 开发者查看BaseOnnx源码中的docstring, **When** 阅读抽象方法的文档, **Then** 每个抽象方法都有清晰的文档说明其职责、参数和返回值

---

### Edge Cases

- **多重继承场景**: 如果有中间基类(如DetectorBaseOnnx继承BaseOnnx),子类继承中间基类时如何确保抽象方法正确传递和实现?
- **部分实现场景**: 如果子类只实现了部分抽象方法,Python的@abstractmethod装饰器应该在实例化时捕获,但需要测试验证所有方法都被正确标记
- **动态添加方法**: 如果开发者尝试在运行时动态添加方法来绕过抽象方法检查,系统应该如何处理?
- **向后兼容性**: 现有的deprecated方法(如旧版infer())是否会受到影响?如何确保渐进式迁移路径?
- **_preprocess实例方法**: 当前_preprocess是实例方法,调用_preprocess_static静态方法。如果子类覆盖_preprocess但未提供_preprocess_static,如何处理?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: BaseOnnx类必须将_postprocess方法标记为抽象方法(@abstractmethod),且方法体只包含raise NotImplementedError的实现
- **FR-002**: BaseOnnx类的__call__方法保持为具体方法(不标记@abstractmethod),实现模板方法模式,但需要进行内部优化:函数解耦和删除旧版本分支逻辑(基于测试覆盖率分析,仅删除未被测试覆盖的分支)
- **FR-003**: BaseOnnx类必须将_preprocess_static静态方法标记为抽象方法(@staticmethod + @abstractmethod),且方法体只包含raise NotImplementedError的实现
- **FR-004**: NotImplementedError错误消息必须遵循格式: "{ClassName}.{method_name}() must be implemented by subclass. {责任描述}"
- **FR-005**: YoloOnnx子类必须提供_postprocess和_preprocess_static的完整实现,如果缺失则在重构中补全
- **FR-006**: RTDETROnnx子类必须提供_postprocess和_preprocess_static的完整实现,如果缺失则在重构中补全
- **FR-007**: RFDETROnnx子类必须提供_postprocess和_preprocess_static的完整实现,如果缺失则在重构中补全
- **FR-008**: ColorLayerONNX子类必须提供_postprocess和_preprocess_static的完整实现,如果缺失则在重构中补全
- **FR-009**: OCRONNX子类必须提供_postprocess和_preprocess_static的完整实现,如果缺失则在重构中补全
- **FR-010**: 所有抽象方法必须包含详细的docstring,说明方法职责、参数类型、返回值类型和异常
- **FR-011**: 重构后所有现有的集成测试(tests/integration/)必须继续通过,验证功能完整性
- **FR-012**: 重构后所有现有的单元测试(tests/unit/)必须继续通过,验证单个组件功能
- **FR-013**: 系统必须保留向后兼容性,现有的deprecated方法(如infer())仍可使用但会发出警告
- **FR-014**: 在删除旧版本分支逻辑前,必须使用代码覆盖工具(如pytest-cov)分析__call__方法的测试覆盖率,生成覆盖报告,仅删除未被任何测试执行到的分支代码
- **FR-015**: __call__方法必须重构为调用3个主要阶段的私有方法:`_prepare_inference()`(预处理阶段)、`_execute_inference()`(推理执行阶段)、`_finalize_inference()`(后处理和结果整理阶段),保持方法间职责清晰

### Key Entities *(include if feature involves data)*

- **BaseOnnx**: 抽象基类,定义ONNX模型推理的统一接口契约,包含2个核心抽象方法(_postprocess, _preprocess_static)和1个具体模板方法(__call__)
- **AbstractMethod**: Python的抽象方法装饰器(@abstractmethod),在实例化时强制检查子类实现
- **SubclassImplementation**: 各个子类(YoloOnnx/RTDETROnnx/RFDETROnnx/ColorLayerONNX/OCRONNX)对抽象方法的具体实现
- **ErrorMessage**: NotImplementedError异常对象,包含结构化的错误信息(类名、方法名、职责描述)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有尝试实例化未完整实现抽象方法的BaseOnnx子类时,在实例化时立即抛出TypeError异常(Python标准行为)
- **SC-002**: 所有抽象方法的NotImplementedError错误消息格式统一且包含完整信息(类名+方法名+职责描述),开发者能在5秒内理解错误原因
- **SC-003**: 重构后运行完整测试套件,集成测试通过率保持100%(当前115/122通过的非核心失败不计入),单元测试通过率保持100%(当前27/27)
- **SC-004**: 所有5个子类(YoloOnnx/RTDETROnnx/RFDETROnnx/ColorLayerONNX/OCRONNX)都能成功实例化并执行至少一次完整推理,无抽象方法未实现错误
- **SC-005**: 代码审查确认2个抽象方法(_postprocess/_preprocess_static)都正确添加@abstractmethod装饰器且基类实现只包含raise NotImplementedError; __call__方法保持为具体方法且完成内部优化
- **SC-006**: 推理性能指标不降低,单次推理延迟保持在< 50ms (640x640输入),GPU内存使用保持< 2GB (batch_size=1)
- **SC-007**: 重构不引入新的pylint或mypy类型检查错误,代码质量分数不降低
- **SC-008**: __call__方法重构前生成测试覆盖报告,重构仅删除覆盖率为0%的分支代码,保留所有被测试执行到的逻辑路径
- **SC-009**: __call__方法重构后代码行数减少至少30%,方法复杂度(圈复杂度)降低,3个阶段方法职责清晰且独立可测试

## Assumptions *(optional, document defaults)*

### 技术假设
- Python的@abstractmethod装饰器会在实例化时自动检查所有抽象方法是否已实现,无需额外运行时检查
- 当前所有5个子类都使用BaseOnnx提供的__call__默认实现,不需要子类重写
- _preprocess实例方法调用_preprocess_static的模式在所有子类中一致,只需确保_preprocess_static被标记为抽象即可
- __call__方法内部的"旧版本分支逻辑"主要指兼容返回值格式的判断代码(如line 162-168处理3元组/4元组返回值兼容性)

### 业务假设
- 开发者熟悉Python的抽象基类(ABC)概念和@abstractmethod装饰器的使用
- 现有代码库没有动态创建BaseOnnx子类的场景(如使用type()函数动态创建类)
- 所有现有子类都已在生产环境中运行稳定,重构风险主要在于测试覆盖不足的边缘场景

### 默认行为
- 如果子类未实现抽象方法,Python在实例化时抛出TypeError而不是NotImplementedError
- NotImplementedError只在基类方法被意外调用时触发(例如通过super()调用或反射调用)
- 所有抽象方法的docstring遵循Google风格,包含Args、Returns、Raises三个部分

## Dependencies *(optional)*

### 内部依赖
- **测试框架**: 依赖pytest运行单元测试和集成测试,验证重构后的功能完整性
- **代码覆盖工具**: 依赖pytest-cov生成测试覆盖率报告,指导旧版本分支逻辑的安全删除
- **类型检查**: 依赖mypy静态类型检查工具验证抽象方法签名的类型一致性
- **代码质量**: 依赖pylint代码检查工具确保重构后代码符合项目编码规范

### 外部依赖
- **Python版本**: 项目要求Python 3.10+,确保@abstractmethod装饰器行为符合预期
- **ABC模块**: 依赖Python标准库的abc模块提供抽象基类功能

### 阻塞依赖
- 无阻塞依赖,重构可以独立进行

## Out of Scope *(optional)*

以下内容不在本次重构范围内:

- 修改抽象方法的参数签名或返回值类型(保持现有接口不变)
- 添加新的抽象方法或移除现有抽象方法
- 重构BaseOnnx的其他非核心方法(如_ensure_initialized、compare_engine等),本次只优化__call__
- 修改子类的业务逻辑或算法实现
- 优化推理性能或内存使用(除非重构导致退化需要修复)
- 添加新的子类或模型架构支持
- 修改TensorRT引擎构建逻辑或Polygraphy集成
- 重构_preprocess实例方法(保持其作为_preprocess_static的包装器)

## Risks and Mitigations *(optional)*

### 风险1: __call__方法优化引入回归问题
**描述**: 在对__call__方法进行函数解耦和删除旧版本分支逻辑时,可能意外破坏现有的兼容性逻辑或引入边缘情况bug。

**影响**: 中 - 可能导致某些子类或特定输入场景下的推理失败

**缓解策略**:
- 在重构前充分理解旧版本分支逻辑的存在原因和适用场景
- 使用渐进式重构:先添加测试覆盖,再移除旧逻辑
- 保留关键的向后兼容性逻辑(如deprecated方法的警告)
- 通过完整的集成测试验证所有子类在重构后仍正常工作

**决策**: 已确认保持__call__为具体方法(模板方法模式),只进行内部优化而非接口变更

### 风险2: 测试覆盖不足导致未发现的破坏
**描述**: 当前集成测试有7个失败(115/122通过),可能存在未充分测试的代码路径,重构可能破坏这些路径。

**影响**: 中 - 可能在生产环境中发现未预料的错误

**缓解策略**:
- 在重构前修复现有的7个失败测试,确保基准状态干净
- 添加针对抽象方法检查的单元测试
- 在staging环境进行充分的回归测试

### 风险3: 静态方法抽象化的Python版本兼容性
**描述**: @staticmethod和@abstractmethod的组合在旧版Python中可能有行为差异,虽然项目要求Python 3.10+,但需要验证。

**影响**: 低 - Python 3.10+对此支持良好

**缓解策略**:
- 在多个Python版本(3.10, 3.11, 3.12)上运行测试
- 参考Python官方文档验证装饰器组合的正确顺序

## Notes *(optional)*

### 实现注意事项
1. **装饰器顺序**: 对于_preprocess_static,正确的装饰器顺序是先@staticmethod后@abstractmethod:
   ```python
   @staticmethod
   @abstractmethod
   def _preprocess_static(...):
       raise NotImplementedError(...)
   ```

2. **__call__方法的处理**: 已确认保持__call__为具体方法(模板方法模式),不标记@abstractmethod。重构流程:
   - **第一步**: 运行`pytest --cov=infer_onnx.onnx_base --cov-report=html --cov-report=term-missing`生成覆盖报告
   - **第二步**: 分析覆盖报告,识别__call__方法中未被执行的分支(显示为红色或0% coverage)
   - **第三步**: 提取3个阶段方法:
     - `_prepare_inference(image, conf_thres, **kwargs)` - 负责模型初始化、预处理、验证输入
     - `_execute_inference(input_tensor)` - 负责Polygraphy推理执行、batch维度处理
     - `_finalize_inference(outputs, scale, original_shape, conf_thres, **kwargs)` - 负责后处理、结果过滤和格式化
   - **第四步**: 仅删除覆盖率为0%的分支逻辑(如完全未使用的兼容性代码)
   - **第五步**: 重构后的__call__方法结构清晰:
     ```python
     def __call__(self, image, conf_thres=None, **kwargs):
         # 准备阶段
         input_tensor, scale, original_shape, metadata = self._prepare_inference(image, conf_thres, **kwargs)
         # 执行阶段
         outputs = self._execute_inference(input_tensor)
         # 完成阶段
         detections = self._finalize_inference(outputs, scale, original_shape, conf_thres or self.conf_thres, **kwargs)
         return detections, original_shape
     ```

3. **错误消息模板**: 建议在BaseOnnx顶部定义统一的错误消息模板:
   ```python
   ERROR_MSG_TEMPLATE = "{cls_name}.{method_name}() must be implemented by subclass. {description}"
   ```

4. **渐进式迁移**: 如果发现某些子类缺少实现,可以分阶段进行:
   - Phase 1: 标记抽象方法但提供默认实现并发出警告
   - Phase 2: 确认所有子类完整实现后,移除默认实现只保留NotImplementedError

### 参考资料
- Python ABC文档: https://docs.python.org/3/library/abc.html
- PEP 3119 - Introducing Abstract Base Classes: https://www.python.org/dev/peps/pep-3119/
- SOLID原则中的里氏替换原则(LSP): 子类必须能替代父类,抽象方法强制确保接口一致性

---

*最后更新: 2025-10-09*
