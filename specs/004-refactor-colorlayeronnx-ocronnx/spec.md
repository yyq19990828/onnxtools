# Feature Specification: 重构ColorLayerONNX和OCRONNX以继承BaseOnnx

**Feature Branch**: `004-refactor-colorlayeronnx-ocronnx`
**Created**: 2025-10-09
**Status**: Draft
**Input**: User description: "refactor ColorLayerONNX, OCRONNX. inherit from baseonnx"

## Clarifications

### Session 2025-10-09

- Q: OCR预处理函数（`process_plate_image`, `resize_norm_img`, `image_pretreatment`）应该如何整合到重构后的类中？ → A: 混合方式 - 实例方法调用内部静态方法（如BaseOnnx和YoloOnnx的模式）
- Q: `decode()`和`get_ignored_tokens()`函数应该如何整合到OCRONNX类中？ → A: 作为独立的静态方法（如`_decode_static()`, `_get_ignored_tokens()`），由`_postprocess()`调用
- Q: 删除utils/ocr_image_processing.py和utils/ocr_post_processing.py后，如何处理现有调用者？ → A: 立即删除文件，并同步修改所有调用者代码（如pipeline.py改为调用OCRONNX和ColorLayerONNX的方法）
- Q: 整合的静态预处理方法应该是私有的还是公开的？ → A: 私有静态方法（如`_resize_norm_img_static()`, `_process_plate_image_static()`），仅供类内部使用，严格封装
- Q: 复杂的双层车牌预处理逻辑（`process_plate_image`含80+行）应该如何组织？ → A: 拆分为多个私有静态辅助方法（如`_detect_skew_angle()`, `_find_optimal_split_line()`, `_correct_skew()`），主方法`_process_plate_image_static()`调用它们

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 统一的模型初始化和管理 (Priority: P1)

作为系统维护者，我需要ColorLayerONNX和OCRONNX使用与其他检测器相同的初始化模式，这样所有ONNX模型都能享受Polygraphy懒加载、统一的provider选择和一致的错误处理机制。

**Why this priority**: 这是核心架构改进，影响整个推理引擎的一致性和可维护性。统一的初始化模式可以减少重复代码，提高代码质量。

**Independent Test**: 可以通过创建ColorLayerONNX和OCRONNX实例，验证它们是否正确继承了BaseOnnx的初始化逻辑，包括Polygraphy懒加载和provider配置，而无需依赖其他功能。

**Acceptance Scenarios**:

1. **Given** 用户提供一个OCR模型路径，**When** 创建OCRONNX实例，**Then** 模型应使用Polygraphy懒加载初始化，并支持CUDA和CPU providers
2. **Given** 用户提供一个颜色分类模型路径，**When** 创建ColorLayerONNX实例，**Then** 模型应延迟加载直到首次推理调用
3. **Given** 模型文件不存在，**When** 初始化任一推理器，**Then** 系统应提供清晰的错误信息，而不是崩溃

---

### User Story 2 - 标准化的推理接口 (Priority: P1)

作为开发者，我需要使用统一的`__call__()`方法进行推理，而不是每个模型都有不同的`infer()`方法，这样可以在代码中以一致的方式使用所有模型类型。

**Why this priority**: 统一的接口是可互换性的基础，使得代码更容易理解和维护，减少集成错误。

**Independent Test**: 可以编写测试用例，使用相同的调用模式测试所有推理器（detector、OCR、color classifier），验证它们都能正确响应标准的`model(image)`调用。

**Acceptance Scenarios**:

1. **Given** 已初始化的OCRONNX实例，**When** 使用`ocr_model(image)`调用，**Then** 应返回OCR识别结果和原始图像形状
2. **Given** 已初始化的ColorLayerONNX实例，**When** 使用`color_model(image)`调用，**Then** 应返回颜色和层级分类结果
3. **Given** 使用旧的`infer()`方法调用，**When** 重构完成后，**Then** 应提供清晰的弃用警告或向后兼容支持

---

### User Story 3 - TensorRT引擎比较能力 (Priority: P2)

作为性能优化人员，我需要能够对OCR和颜色分类模型执行ONNX vs TensorRT引擎的精度比较，就像对检测模型一样，以便验证TensorRT优化后的精度损失是否在可接受范围内。

**Why this priority**: 这是高级功能，支持性能优化和质量保证工作流，但不影响基本功能。

**Independent Test**: 可以独立测试engine比较功能，通过提供测试数据集验证ONNX和TensorRT输出的一致性，无需依赖实际的车牌检测流程。

**Acceptance Scenarios**:

1. **Given** OCRONNX实例和对应的TensorRT引擎文件，**When** 调用`compare_engine()`方法，**Then** 应返回精度比较报告，显示输出差异是否在容差范围内
2. **Given** ColorLayerONNX实例，**When** 创建引擎数据加载器，**Then** 应生成适合分类模型的预处理数据
3. **Given** 引擎比较失败（精度差异过大），**When** 查看比较结果，**Then** 应提供详细的差异统计信息和失败原因

---

### Edge Cases

- **当OCR或颜色模型的输入形状与检测模型不同时**，系统应正确处理不同的预处理需求（如OCR的[1,3,48,168]和颜色的[1,3,224,224]）
- **当模型有多个输入或输出时**，系统应正确适配BaseOnnx的单输入假设或扩展支持多输入
- **当调用不存在的配置文件时**（如class_names），系统应优雅返回空字典而不是崩溃
- **当OCR模型输出与预期格式不一致时**（如字典输出vs列表输出），系统应提供清晰的错误信息
- **当使用FP16或INT8量化的TensorRT引擎时**，比较容差应相应调整以反映预期的精度损失

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: ColorLayerONNX必须继承自BaseOnnx基类，而不是独立实现会话管理
- **FR-002**: OCRONNX必须继承自BaseOnnx基类，并复用Polygraphy懒加载机制
- **FR-003**: 两个类必须实现`_postprocess()`抽象方法，定义各自的后处理逻辑
- **FR-017**: OCRONNX必须将utils/ocr_post_processing.py中的`decode()`和`get_ignored_tokens()`函数整合为类的静态方法（如`_decode_static()`, `_get_ignored_tokens()`），并在`_postprocess()`中调用
- **FR-004**: 系统必须将原有的`infer()`方法重构为标准的`__call__()`接口
- **FR-005**: 系统必须保持OCR识别的输出格式和准确性不变（向后兼容）
- **FR-006**: 系统必须保持颜色和层级分类的输出格式和准确性不变
- **FR-007**: 重构后的类必须支持自定义input_shape参数，适配不同的模型尺寸（OCR: 48x168, Color: 224x224）
- **FR-008**: 系统必须正确处理OCR模型的多输入情况（如果存在），通过BaseOnnx的输入名称机制
- **FR-009**: 系统必须移除重复的provider选择逻辑，统一使用BaseOnnx的provider管理
- **FR-014**: OCRONNX必须将utils/ocr_image_processing.py中的预处理函数（`process_plate_image`, `resize_norm_img`）整合为类的私有静态方法（如`_process_plate_image_static()`, `_resize_norm_img_static()`），并通过实例方法`_preprocess()`调用
- **FR-019**: OCRONNX必须将复杂的双层车牌处理逻辑拆分为多个私有静态辅助方法（如`_detect_skew_angle()`, `_correct_skew()`, `_find_optimal_split_line()`），由`_process_plate_image_static()`主方法协调调用，保持单一职责原则
- **FR-015**: ColorLayerONNX必须将utils/ocr_image_processing.py中的`image_pretreatment`函数整合为类的私有静态方法（如`_image_pretreatment_static()`）
- **FR-016**: 整合完成后，系统必须删除utils/ocr_image_processing.py和utils/ocr_post_processing.py文件，所有功能迁移到类内部
- **FR-018**: 系统必须同步修改所有调用utils/ocr_image_processing.py和utils/ocr_post_processing.py的代码（如utils/pipeline.py），改为直接调用OCRONNX和ColorLayerONNX的类方法或静态方法
- **FR-010**: ColorLayerONNX必须正确分离和返回color_logits和layer_logits两个输出
- **FR-011**: OCRONNX必须支持多输入名称的feed_dict构建（兼容现有实现）
- **FR-012**: 重构后的类必须能够使用`create_engine_dataloader()`方法创建TensorRT比较数据加载器
- **FR-013**: 系统必须支持`compare_engine()`方法对OCR和颜色分类模型进行精度验证

### Key Entities *(include if feature involves data)*

- **ColorLayerONNX**: 车牌颜色和层级分类推理器，继承自BaseOnnx，包含私有静态预处理方法`_image_pretreatment_static()`，输出颜色类别和单/双层分类结果
- **OCRONNX**: 车牌字符识别推理器，继承自BaseOnnx，包含多个私有静态方法用于预处理（`_process_plate_image_static()`, `_resize_norm_img_static()`, `_detect_skew_angle()`, `_correct_skew()`, `_find_optimal_split_line()`）和后处理（`_decode_static()`, `_get_ignored_tokens()`），输出解码后的文本和置信度
- **BaseOnnx**: 基础ONNX推理抽象类，提供Polygraphy懒加载、统一的会话管理和通用工具方法（特别是`_preprocess()`实例方法调用`_preprocess_static()`静态方法的混合模式）
- **PreprocessResult**: 预处理结果元组，包含(input_tensor, scale, original_shape, ratio_pad)
- **InferenceSession**: Polygraphy管理的ONNX Runtime会话，支持懒加载和多provider
- **EngineDataLoader**: TensorRT引擎比较的数据加载器，根据模型类型生成适配的输入数据

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 重构完成后，所有使用OCR和颜色分类的现有测试用例应100%通过，无功能回归
- **SC-002**: 代码重复度降低至少40%（通过移除重复的会话管理和provider选择代码）
- **SC-003**: OCR和颜色分类模型的首次推理时间（包括懒加载）应保持在200ms以内
- **SC-004**: 重构后的类应支持与其他检测器相同的TensorRT引擎比较工作流，精度容差在1e-3以内
- **SC-005**: 新的实现应减少内存占用，通过懒加载机制使未使用的模型不占用GPU内存
- **SC-006**: 所有公共API调用（如`model(image)`）的响应时间应与重构前保持一致或更快（误差±5%）

## Assumptions *(optional)*

### Technical Assumptions

1. **OCR和颜色模型具有固定输入形状**: 假设OCR模型输入为[1,3,48,168]，颜色分类为[1,3,224,224]，这些形状可以从模型元数据中读取
2. **Polygraphy兼容性**: 假设OCR和颜色分类模型与Polygraphy的SessionFromOnnx完全兼容，无特殊算子需求
3. **单batch推理**: 假设OCR和颜色分类主要用于单样本推理（batch_size=1），不需要复杂的批处理逻辑
4. **输出格式稳定**: 假设OCR模型输出固定为字符概率分布（如[1, seq_len, num_classes]），颜色模型输出为[color_logits, layer_logits]

### Business Assumptions

1. **向后兼容性优先**: 假设现有的utils/pipeline.py和其他调用代码在短期内不会大规模重构，因此需要保持接口兼容性
2. **性能要求不变**: 假设重构不应降低推理性能，用户期望的实时性能指标保持不变
3. **测试覆盖充分**: 假设现有的测试用例已充分覆盖OCR和颜色分类的主要功能，可以作为回归测试基准

## Dependencies *(optional)*

### Internal Dependencies

- **infer_onnx/base_onnx.py**: 必须先理解BaseOnnx的抽象方法和初始化流程（特别是`_preprocess()`和`_preprocess_static()`的混合模式）
- **utils/ocr_image_processing.py**: 将被删除，其函数（`process_plate_image`, `resize_norm_img`, `image_pretreatment`）需迁移到OCRONNX和ColorLayerONNX类内部
- **utils/ocr_post_processing.py**: 将被删除，其`decode()`和`get_ignored_tokens()`函数需迁移到OCRONNX类内部
- **configs/plate.yaml**: OCR字典和颜色映射配置应与重构后的类兼容

### External Dependencies

- **Polygraphy**: 确保版本>=0.49.26，支持所需的懒加载和会话管理功能
- **onnxruntime-gpu**: 确保版本兼容，支持OCR和分类模型的所有算子

## Out of Scope *(optional)*

明确不包含在此次重构中的功能：

1. **后处理逻辑优化**: 不修改OCR解码算法或颜色分类的softmax逻辑，仅重构推理部分
2. **模型格式转换**: 不涉及重新导出ONNX模型或修改模型结构
3. **批处理支持**: 不增加OCR和颜色分类的批量推理能力（保持batch_size=1）
4. **新增可视化功能**: 不添加OCR或颜色分类的新可视化方法
5. **性能极限优化**: 不进行CUDA kernel优化或自定义算子实现
6. **配置文件重构**: 不修改plate.yaml或det_config.yaml的格式和内容
7. **渐进式迁移**: 不提供弃用警告或临时兼容层，直接删除utils/ocr_*.py文件并同步修改调用者

---

*准备进入下一阶段：功能设计和实施计划 (`/speckit.plan`)*
