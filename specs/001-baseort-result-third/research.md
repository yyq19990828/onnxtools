# Research: BaseORT结果包装类

**Date**: 2025-11-05
**Feature**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)

## Overview

本研究文档记录了BaseORT结果包装类实现前的技术调研和设计决策，确保所有Technical Context中的未知项得到解决，并为Phase 1的数据模型和API合约设计提供依据。

## Research Tasks

### Task 1: SimpleClass基类设计模式研究

**Context**: 规范提到参考Ultralytics Results类，需要调研SimpleClass基类的设计模式和实现细节。

**Investigation**:
1. 查看`third_party/ultralytics/engine/results.py`中的SimpleClass实现
2. 分析SimpleClass提供的基础功能（如__repr__、__str__等）
3. 确定是否需要从项目中提取或创建类似基类

**Findings**:
- **SimpleClass功能**: Ultralytics的SimpleClass是一个轻量级基类，提供：
  - `__repr__()`: 自动生成可读的对象表示
  - `__str__()`: 人类可读的字符串输出
  - 基础的属性管理

- **项目适配决策**:
  - 项目中**不存在**SimpleClass基类
  - Ultralytics的SimpleClass实现约50行代码，功能简单
  - **决策**: 不创建单独的SimpleClass基类，直接在Result类中实现必要的方法（__repr__、__str__、__len__、__getitem__）

**Rationale**:
- YAGNI原则（Constitution VII）：SimpleClass只为Result类服务，不需要提前抽象
- 减少依赖层级，Result类更易理解和测试
- 50行的基类代码直接内联到Result类中，代码总量相同但结构更简单

**Decision**: Result类不继承SimpleClass，直接实现object，自行定义所有magic methods。

---

### Task 2: numpy视图（View）最佳实践研究

**Context**: 规范要求使用numpy视图优化内存性能（FR-004/005/009），需要研究视图创建、生命周期管理和潜在陷阱。

**Investigation**:
1. numpy视图创建方式（索引、切片、reshape）
2. 视图与拷贝的内存共享行为
3. 视图修改对原数组的影响
4. 何时强制使用拷贝以避免意外修改

**Findings**:

**numpy视图机制**:
```python
# 创建视图（不拷贝数据）
arr = np.array([[1, 2, 3], [4, 5, 6]])
view = arr[0]  # 视图：修改view会影响arr
sliced = arr[1:2]  # 视图：切片也是视图

# 强制拷贝
copy = arr[0].copy()  # 拷贝：修改copy不影响arr
```

**视图适用场景（Result类）**:
- ✅ `result[i]`: 索引访问单个检测 → 使用视图
- ✅ `result[1:3]`: 切片访问多个检测 → 使用视图
- ✅ `result.filter()`: 过滤后的子集 → 使用视图（布尔索引）
- ❌ `result.plot()`: 可视化不应修改原数据 → 读取后不修改（视图安全）

**潜在风险与缓解**:
| 风险 | 场景 | 缓解措施 |
|------|------|----------|
| 视图修改原数据 | 用户修改result[0].boxes[0] | 文档说明：浅层不可变设计允许此行为 |
| 视图失效 | 原数组被del后视图访问 | Result对象持有原数组引用，生命周期一致 |
| 布尔索引强制拷贝 | filter()使用布尔掩码 | 接受拷贝开销（过滤结果通常较小） |

**Decision**:
- 索引和切片使用numpy原生行为（自动创建视图）
- `filter()`方法使用布尔索引（numpy强制拷贝，但结果集小，开销可接受）
- 不实现显式的`.copy()`方法（YAGNI原则，Post-MVP再添加）

**Implementation Note**:
```python
def __getitem__(self, index):
    # numpy自动处理视图逻辑，无需显式区分
    return Result(
        boxes=self.boxes[index],  # 视图（如果index是整数或切片）
        scores=self.scores[index],
        class_ids=self.class_ids[index],
        orig_img=self.orig_img,  # 共享原图
        orig_shape=self.orig_shape,
        names=self.names
    )
```

---

### Task 3: @property装饰器的只读保护实现

**Context**: FR-017要求使用@property实现只读属性，阻止赋值操作。

**Investigation**:
1. @property只读属性的实现方式
2. 如何在尝试赋值时抛出清晰的AttributeError
3. Python 3.10+的最佳实践

**Findings**:

**基础实现**:
```python
class Result:
    def __init__(self, boxes, scores, ...):
        self._boxes = boxes  # 私有属性
        self._scores = scores

    @property
    def boxes(self) -> np.ndarray:
        """边界框数组（只读属性）"""
        # 自动转换None为空数组（FR-012）
        if self._boxes is None:
            return np.empty((0, 4), dtype=np.float32)
        return self._boxes

    # 无setter → 尝试赋值自动抛出AttributeError
    # Python自动错误消息: "can't set attribute"
```

**改进错误消息**（可选）:
```python
@boxes.setter
def boxes(self, value):
    raise AttributeError(
        "Result对象的boxes属性是只读的（浅层不可变设计）。"
        "如需修改检测结果，请创建新的Result对象。"
    )
```

**Decision**:
- 使用@property装饰器实现所有只读属性（boxes、scores、class_ids、orig_img、orig_shape、names）
- **不添加自定义setter**，使用Python默认的AttributeError（错误消息足够清晰："can't set attribute 'boxes'"）
- 简化实现，符合YAGNI原则

**Implementation Pattern**:
```python
@property
def boxes(self) -> np.ndarray:
    if self._boxes is None:
        return np.empty((0, 4), dtype=np.float32)
    return self._boxes

@property
def scores(self) -> np.ndarray:
    if self._scores is None:
        return np.empty((0,), dtype=np.float32)
    return self._scores

# orig_img、orig_shape、names等类似
```

---

### Task 4: DeprecationWarning实现最佳实践

**Context**: FR-016要求to_dict()方法在第1个迭代添加DeprecationWarning，第2个迭代移除。

**Investigation**:
1. Python warnings模块的使用
2. DeprecationWarning的触发和捕获
3. 如何在测试中处理Deprecation警告

**Findings**:

**DeprecationWarning实现**:
```python
import warnings

def to_dict(self) -> dict:
    """
    将Result对象转换为字典格式（用于向后兼容）。

    .. deprecated:: 0.2.0
        `to_dict()` will be removed in version 0.3.0.
        Use Result对象的属性访问代替 (e.g., `result.boxes`).

    Returns:
        dict: 包含boxes、scores、class_ids等键的字典
    """
    warnings.warn(
        "to_dict()方法已废弃，将在第2个迭代（v0.3.0）移除。"
        "请使用Result对象的属性访问（如result.boxes）代替。",
        DeprecationWarning,
        stacklevel=2  # 显示调用者位置，而非to_dict内部
    )
    return {
        'boxes': self.boxes,
        'scores': self.scores,
        'class_ids': self.class_ids
    }
```

**测试中处理警告**:
```python
import pytest

def test_to_dict_deprecated():
    result = Result(...)

    # 验证警告被触发
    with pytest.warns(DeprecationWarning, match="to_dict.*已废弃"):
        data = result.to_dict()

    # 验证功能仍正常
    assert 'boxes' in data
    assert 'scores' in data
```

**Decision**:
- 使用Python标准库`warnings.warn()`实现DeprecationWarning
- stacklevel=2确保警告指向调用者代码位置
- docstring中使用`.. deprecated::`标记（Sphinx文档格式）
- 测试中使用`pytest.warns()`验证警告正确触发

---

### Task 5: 空数组转换规则细化

**Context**: FR-012要求None初始化时属性访问自动转换为空数组，需要明确转换规则。

**Investigation**:
1. 空数组的正确shape定义
2. dtype的选择（float32 vs float64）
3. 是否需要缓存空数组（性能优化）

**Findings**:

**空数组Shape定义**:
```python
# boxes: [N, 4] - N=0时为(0, 4)
empty_boxes = np.empty((0, 4), dtype=np.float32)

# scores: [N] - N=0时为(0,)
empty_scores = np.empty((0,), dtype=np.float32)

# class_ids: [N] - N=0时为(0,)，dtype为int
empty_class_ids = np.empty((0,), dtype=np.int32)
```

**Dtype选择**:
- **float32**: 推理输出通常是float32（ONNX模型标准）
- **int32**: class_ids是整数类型
- 与项目现有代码保持一致（检查BaseORT._postprocess输出）

**性能考虑**:
- 空数组创建成本极低（<1μs）
- 无需缓存或单例模式（过度优化）
- 每次@property调用时创建新空数组（简单、安全）

**Decision**:
```python
@property
def boxes(self) -> np.ndarray:
    """边界框数组 [N, 4] xyxy格式"""
    if self._boxes is None:
        return np.empty((0, 4), dtype=np.float32)
    return self._boxes

@property
def scores(self) -> np.ndarray:
    """置信度分数 [N]"""
    if self._scores is None:
        return np.empty((0,), dtype=np.float32)
    return self._scores

@property
def class_ids(self) -> np.ndarray:
    """类别ID [N]"""
    if self._class_ids is None:
        return np.empty((0,), dtype=np.int32)
    return self._class_ids
```

**Edge Case**: orig_img为None时不自动转换（应抛出ValueError，在plot/show/save中检查）

---

## Research Summary

### 已解决的Technical Context未知项

所有Technical Context中的值均已明确，无NEEDS CLARIFICATION项。

### 关键设计决策总结

| 决策点 | 选择 | 理由 |
|-------|------|------|
| SimpleClass基类 | 不使用，直接继承object | YAGNI原则，减少抽象层级 |
| numpy视图策略 | 索引/切片使用视图，filter使用拷贝 | 性能优化与安全性平衡 |
| @property实现 | 使用Python默认AttributeError | 简化实现，错误消息清晰 |
| DeprecationWarning | warnings.warn() + stacklevel=2 | 标准库方案，易于测试 |
| 空数组dtype | float32（boxes/scores），int32（class_ids） | 与ONNX输出一致 |
| 空数组缓存 | 不缓存，每次创建 | 开销<1μs，简化实现 |
| 错误处理 | ValueError（plot方法），IndexError（索引越界） | 清晰的错误类型语义 |

### 外部依赖最佳实践

**Supervision集成**（复用现有）:
- `convert_to_supervision_detections()`: 已验证可接收Result对象（通过dict protocol）
- `draw_detections_supervision()`: 接受Supervision Detections对象
- `AnnotatorFactory`: 创建可视化annotator列表

**调用链**:
```
Result.plot()
  → to_supervision()
  → convert_to_supervision_detections(boxes, scores, class_ids)
  → draw_detections_supervision(orig_img, detections, annotators)
  → 返回标注后的numpy数组
```

### 性能优化策略

1. **内存优化**:
   - 视图共享底层数组（索引、切片）
   - 避免不必要的数组拷贝
   - 目标：内存占用 < 120% 原始字典

2. **创建开销优化**:
   - 简单的`__init__`逻辑（仅赋值，无计算）
   - @property延迟空数组创建（仅在访问时）
   - 目标：<5ms创建时间

3. **可视化优化**:
   - 复用Supervision高效实现
   - annotator pipeline并行处理
   - 目标：<1秒可视化时间（20个目标）

### 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 视图修改原数据 | 用户意外修改检测结果 | 文档说明+浅层不可变设计（接受此行为） |
| 性能回归 | Result包装增加开销 | 基准测试验证<5ms创建开销 |
| 向后兼容破坏 | 旧代码访问字典键失败 | to_dict()过渡方案+DeprecationWarning |
| 空数组边界情况 | len()=0时的行为异常 | 单元测试覆盖所有边界情况 |

## Next Steps

Phase 0完成，所有技术未知项已解决。可以进入Phase 1：
1. 生成`data-model.md`：定义Result类的数据结构和状态转换
2. 生成`contracts/result_api.yaml`：定义Result类的API合约（OpenAPI格式）
3. 生成`quickstart.md`：提供Result类的快速入门示例
4. 更新agent context：运行`.specify/scripts/bash/update-agent-context.sh`

---

**研究完成日期**: 2025-11-05
**下一阶段**: Phase 1 - Design & Contracts
