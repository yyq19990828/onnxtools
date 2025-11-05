# Feature Specification: BaseORT结果包装类

**Feature Branch**: `001-baseort-result-third`
**Created**: 2025-11-05
**Status**: Draft
**Input**: User description: "BaseORT子类后处理后得到的结果包装成一个result类, 可以便捷地提供更多处理方法, 参考third_party/ultralytics/engine/results.py yolo官方仓库的实现, 由于我们只是onnx后处理结果的包装, 所以不需要像他这么复杂, 要做精简."

## Clarifications

### Session 2025-11-05

- Q: BaseORT.__call__()返回值的向后兼容性策略？ → A: 立即强制返回Result对象，但提供临时的.to_dict()方法供旧代码迁移（推荐1-2个迭代后废弃）
- Q: to_dict()方法的废弃时间线？ → A: 第1个迭代标记废弃（添加DeprecationWarning），第2个迭代移除
- Q: 空检测结果的初始化行为？ → A: 允许None初始化，但属性访问时自动转换为空数组（boxes返回shape为(0,4)的空数组，scores和class_ids返回shape为(0,)的空数组）
- Q: verbose()方法的实现优先级？ → A: 不纳入MVP，作为Post-MVP增强功能（P3阶段实现）
- Q: Result对象的不可变性设计？ → A: 浅层不可变（属性只读，但内部数组可修改），filter()和索引操作共享底层数组（通过numpy视图）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 基础检测结果访问和操作 (Priority: P1)

开发人员需要在执行目标检测推理后，以更直观和面向对象的方式访问和操作检测结果（边界框、置信度、类别等），而不是直接处理原始的字典或数组结构。

**Why this priority**: 这是结果包装类的核心价值，直接影响所有使用检测器的代码。提供清晰的API能大幅提升代码可读性和开发效率。

**Independent Test**: 可以独立测试，方式是创建检测结果，然后通过Result对象的属性访问boxes、scores、class_ids等数据，验证数据正确性和属性访问的便捷性。

**Acceptance Scenarios**:

1. **Given** 一个检测器完成推理返回原始结果字典，**When** 开发人员使用Result类包装该结果，**Then** 可以通过result.boxes、result.scores、result.class_ids等属性直接访问数据
2. **Given** 一个包含10个检测目标的Result对象，**When** 开发人员调用len(result)，**Then** 返回检测目标数量10
3. **Given** 一个Result对象，**When** 开发人员使用索引访问result[0]，**Then** 返回一个新的Result对象，仅包含第一个检测目标的数据
4. **Given** 一个Result对象包含原始图像信息，**When** 开发人员访问result.orig_img和result.orig_shape，**Then** 获取原始图像数组和形状元组

---

### User Story 2 - 结果可视化和保存 (Priority: P2)

开发人员需要快速可视化检测结果并保存标注后的图像，而无需手动编写绘制代码或调用外部可视化工具。

**Why this priority**: 这是高频使用场景，特别是在调试和演示时。简化可视化流程能显著提升开发体验，但不影响核心推理功能。

**Independent Test**: 可以独立测试，方式是创建Result对象，调用plot()方法绘制结果，调用show()显示图像，调用save()保存到文件，验证输出图像的正确性。

**Acceptance Scenarios**:

1. **Given** 一个包含检测结果的Result对象，**When** 开发人员调用result.plot()，**Then** 返回一个在原图上绘制了边界框、标签和置信度的numpy数组图像
2. **Given** 一个Result对象，**When** 开发人员调用result.show()，**Then** 在窗口中显示标注后的检测结果图像
3. **Given** 一个Result对象和输出路径，**When** 开发人员调用result.save(output_path)，**Then** 将标注后的图像保存到指定路径
4. **Given** 一个Result对象，**When** 开发人员调用result.plot()并指定自定义annotator_preset参数，**Then** 使用指定的可视化预设风格绘制结果

---

### User Story 3 - 结果过滤和转换 (Priority: P3)

开发人员需要根据条件过滤检测结果（如置信度阈值、特定类别）并将结果转换为其他数据格式（如Supervision格式、pandas DataFrame等），以便后续处理或与其他工具集成。

**Why this priority**: 这是增强功能，提供更灵活的数据处理能力，但不是立即必需的。可以在基础功能稳定后逐步添加。

**Independent Test**: 可以独立测试，方式是创建Result对象，应用各种过滤条件（置信度、类别ID），验证过滤后返回的新Result对象仅包含符合条件的检测目标。

**Acceptance Scenarios**:

1. **Given** 一个包含混合置信度检测的Result对象，**When** 开发人员调用result.filter(conf_threshold=0.7)，**Then** 返回一个新的Result对象，仅包含置信度>=0.7的检测
2. **Given** 一个包含多类别检测的Result对象，**When** 开发人员调用result.filter(classes=[0, 1])，**Then** 返回一个新的Result对象，仅包含类别ID为0或1的检测
3. **Given** 一个Result对象，**When** 开发人员调用result.to_supervision()，**Then** 返回一个Supervision Detections对象
4. **Given** 一个Result对象，**When** 开发人员调用result.summary()，**Then** 返回一个包含检测统计信息的字典（如总数、各类别数量、平均置信度）

---

### Edge Cases

- 当检测结果为空（无目标检测到）时，Result对象如何表示？len()返回0，索引访问抛出IndexError，属性访问返回形状正确的空数组
- 当原始图像为None时，plot()和show()方法如何处理？应抛出ValueError并提供明确错误消息
- 当对Result对象进行切片操作（如result[1:3]）时，应返回包含索引1和2的新Result对象
- 当boxes、scores或class_ids为None时初始化Result对象，属性访问时自动转换为对应形状的空numpy数组（无需显式错误，保持API一致性）
- 当调用filter()后结果为空时，应返回一个空的Result对象而非None

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Result类必须能够接收BaseORT子类后处理输出的字典（包含boxes、scores、class_ids等键）并存储为对象属性
- **FR-002**: Result类必须提供只读属性访问方式获取boxes（边界框）、scores（置信度）、class_ids（类别ID）、orig_img（原始图像）、orig_shape（原始形状）、names（类别名称字典），禁止通过赋值修改这些属性（浅层不可变设计），但允许修改内部numpy数组元素
- **FR-003**: Result类必须支持len()函数，返回检测目标的数量（基于boxes的第一维长度）
- **FR-004**: Result类必须支持索引访问（__getitem__），允许通过result[i]获取第i个检测目标，返回新的Result对象（底层数组使用numpy视图，避免不必要的拷贝）
- **FR-005**: Result类必须支持切片访问（如result[1:3]），返回包含指定范围检测目标的新Result对象（底层数组使用numpy视图，避免不必要的拷贝）
- **FR-006**: Result类必须提供plot()方法，在原始图像上绘制检测结果（边界框、标签、置信度），返回标注后的numpy图像数组
- **FR-007**: Result类必须提供show()方法，在窗口中显示标注后的检测结果
- **FR-008**: Result类必须提供save()方法，将标注后的图像保存到指定文件路径
- **FR-009**: Result类必须提供filter()方法，支持按置信度阈值和类别ID列表过滤检测结果，返回新的Result对象（底层数组使用numpy视图，避免不必要的拷贝）
- **FR-010**: Result类必须提供to_supervision()方法，将检测结果转换为Supervision Detections对象格式
- **FR-011**: Result类必须提供summary()方法，返回包含检测统计信息的字典（如检测总数、各类别数量、平均置信度）
- **FR-012**: Result类必须能够处理空检测结果，允许boxes/scores/class_ids在初始化时为None或空数组，len()返回0，索引访问抛出IndexError，但属性访问时自动转换为形状正确的空numpy数组（boxes返回shape (0,4)，scores和class_ids返回shape (0,)）
- **FR-013**: BaseORT子类的__call__()方法必须强制返回Result对象而非字典，同时Result类必须提供to_dict()方法用于临时向后兼容（第1个迭代标记为废弃，第2个迭代移除）
- **FR-014**: Result类的plot()方法必须支持自定义annotator_preset参数，允许用户选择不同的可视化风格
- **FR-015**: Result类必须提供numpy()方法，确保所有内部数据为numpy数组格式（而非torch.Tensor）
- **FR-016**: Result类必须提供to_dict()方法，返回包含boxes、scores、class_ids等键的字典格式（用于向后兼容旧代码），该方法在第1个迭代即添加DeprecationWarning，第2个迭代完全移除
- **FR-017**: Result类的属性访问必须使用@property装饰器实现只读保护，尝试赋值（如result.boxes = new_boxes）时抛出AttributeError

### Key Entities

- **Result**: 检测结果包装类，核心属性包括：
  - boxes (np.ndarray): [N, 4]边界框坐标（xyxy格式）
  - scores (np.ndarray): [N]置信度分数
  - class_ids (np.ndarray): [N]类别ID
  - orig_img (np.ndarray): 原始输入图像
  - orig_shape (tuple): 原始图像形状(height, width)
  - names (dict): 类别ID到类别名称的映射
  - path (str): 图像文件路径（可选）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有BaseORT子类（YoloORT、RtdetrORT、RfdetrORT等）能够返回Result对象，开发人员通过result.boxes访问数据的代码量比直接访问字典减少30%
- **SC-002**: Result对象的plot()和show()方法能够在1秒内完成可视化（包含20个检测目标的640x640图像）
- **SC-003**: Result类的单元测试覆盖率达到90%以上，包括属性访问、索引操作、过滤、可视化等所有公共方法
- **SC-004**: Result类的API文档和使用示例完整，开发人员无需查看源码即可理解如何使用所有主要方法
- **SC-005**: Result对象的内存占用不超过原始字典结构的120%（精简设计目标）
- **SC-006**: 从字典结果创建Result对象的开销小于5毫秒（640x640图像，20个检测目标）
- **SC-007**: 90%的常见检测结果处理场景（访问数据、过滤、可视化）能够通过Result类方法完成，无需调用外部工具函数

## Assumptions *(mandatory)*

- 假设BaseORT子类返回的字典结构保持一致，包含'boxes'、'scores'、'class_ids'键，值为numpy数组
- 假设boxes坐标格式为xyxy（左上角和右下角坐标），与现有BaseORT子类输出一致
- 假设可视化功能复用现有的Supervision集成和AnnotatorFactory（onnxtools/utils/），无需重新实现绘制逻辑
- 假设Result类不处理torch.Tensor，所有输入数据已转换为numpy数组（由BaseORT子类负责）
- 假设Result类不涉及多GPU或分布式场景，仅处理单机单卡的推理结果
- 假设Result类的可视化方法（plot/show/save）调用时原始图像已可用，否则抛出明确错误
- 假设Result类不实现Ultralytics Results中的分割掩码、关键点、旋转框等功能（当前项目不涉及）

## Dependencies *(optional)*

- **内部依赖**:
  - `onnxtools.infer_onnx.onnx_base.BaseORT`: Result类需要与BaseORT子类集成
  - `onnxtools.utils.supervision_converter`: to_supervision()方法需要调用现有的转换函数
  - `onnxtools.utils.drawing`: plot()方法需要调用draw_detections_supervision()
  - `onnxtools.utils.annotator_factory`: 支持自定义annotator_preset参数

- **外部依赖**:
  - numpy: 核心数据结构
  - opencv-python: show()和save()方法
  - supervision: to_supervision()转换

## Non-Goals *(optional)*

- **不支持**分割掩码（masks）、关键点（keypoints）、旋转边界框（obb）等高级功能（Ultralytics Results有这些，但当前项目不需要）
- **不提供**to_json()、to_csv()、to_df()等数据导出功能（可在后续迭代添加，当前仅提供summary()）
- **不实现**cpu()、cuda()、to()等设备转换方法（Ultralytics的BaseTensor有这些，但ONNX推理结果已在CPU/numpy）
- **不支持**批量推理结果的批处理（如results[0]获取batch中的第一张图像结果），当前仅处理单张图像的结果
- **不提供**save_txt()、save_crop()等Ultralytics Results中的高级保存功能（当前项目暂不需要）
- **不修改**现有BaseORT子类的_postprocess()方法签名（保持向后兼容），Result类作为可选的包装层
- **不在MVP中实现**verbose()方法（人类可读的字符串输出），该功能推迟到Post-MVP阶段（P3）

## Open Questions

- 是否需要支持Result对象之间的合并操作（如result1 + result2），用于组合多个推理结果？
  - 建议：P4优先级，当前场景较少，可在实际需求出现后添加
- filter()方法是否应该支持更复杂的过滤条件（如lambda函数、面积范围、宽高比等）？
  - 建议：初期仅支持conf_threshold和classes参数，覆盖80%场景；高级过滤可在后续迭代添加
