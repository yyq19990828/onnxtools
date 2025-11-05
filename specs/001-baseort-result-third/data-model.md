# Data Model: BaseORT结果包装类

**Date**: 2025-11-05
**Feature**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)
**Research**: [research.md](./research.md)

## Overview

本文档定义Result类的数据模型，包括实体结构、属性类型、状态转换和验证规则。Result类是BaseORT子类推理结果的面向对象包装，提供统一的访问接口和便捷的操作方法。

---

## Core Entity: Result

### Entity Definition

**Result**: 目标检测推理结果的包装类，封装检测框、置信度、类别等信息，提供属性访问、索引操作、可视化和数据转换功能。

### Attributes

| 属性名 | Python类型 | Numpy Shape | 描述 | 可选性 | 默认值 |
|--------|-----------|-------------|------|--------|--------|
| `_boxes` | `np.ndarray \| None` | `[N, 4]` | 边界框坐标（xyxy格式） | 可选 | `None` |
| `_scores` | `np.ndarray \| None` | `[N]` | 检测置信度分数 | 可选 | `None` |
| `_class_ids` | `np.ndarray \| None` | `[N]` | 类别ID整数 | 可选 | `None` |
| `_orig_img` | `np.ndarray \| None` | `[H, W, 3]` | 原始输入图像（BGR格式） | 可选 | `None` |
| `_orig_shape` | `tuple[int, int]` | N/A | 原始图像形状(height, width) | 必需 | N/A |
| `_names` | `dict[int, str]` | N/A | 类别ID到类别名称的映射 | 可选 | `{}` |
| `_path` | `str \| None` | N/A | 图像文件路径 | 可选 | `None` |

**注意**:
- 下划线前缀（`_`）表示私有属性，仅通过@property访问
- N = 检测目标数量，可为0（空结果）
- H, W = 原始图像的高度和宽度

### Read-Only Properties

所有属性通过`@property`装饰器暴露为只读接口：

| Property | 返回类型 | 描述 | 空值处理 |
|----------|---------|------|----------|
| `boxes` | `np.ndarray` | 边界框数组 [N, 4] | None → `np.empty((0, 4), dtype=np.float32)` |
| `scores` | `np.ndarray` | 置信度数组 [N] | None → `np.empty((0,), dtype=np.float32)` |
| `class_ids` | `np.ndarray` | 类别ID数组 [N] | None → `np.empty((0,), dtype=np.int32)` |
| `orig_img` | `np.ndarray \| None` | 原始图像 [H, W, 3] | 返回None（不自动转换） |
| `orig_shape` | `tuple[int, int]` | 原始形状(H, W) | 必需，不可为None |
| `names` | `dict[int, str]` | 类别名称映射 | 返回空字典 `{}` |
| `path` | `str \| None` | 图像路径 | 返回None |

**空值转换规则（FR-012）**:
- `boxes`、`scores`、`class_ids`: None → 形状正确的空numpy数组
- `orig_img`、`path`: 保持None（不转换）
- `orig_shape`: 必需，初始化时必须提供
- `names`: None → 空字典 `{}`

---

## Initialization Contract

### Constructor Signature

```python
def __init__(
    self,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    orig_img: np.ndarray | None = None,
    orig_shape: tuple[int, int] = None,
    names: dict[int, str] | None = None,
    path: str | None = None
) -> None:
    ...
```

### Validation Rules

| 规则 | 验证条件 | 错误类型 | 错误消息 |
|------|---------|---------|----------|
| V1 | `orig_shape` 不能为None | `TypeError` | "orig_shape is required and cannot be None" |
| V2 | `orig_shape` 必须是长度为2的tuple | `ValueError` | "orig_shape must be a tuple of (height, width)" |
| V3 | 如果提供boxes，shape必须为(N, 4) | `ValueError` | "boxes must have shape (N, 4), got {shape}" |
| V4 | 如果提供scores，shape必须为(N,) | `ValueError` | "scores must have shape (N,), got {shape}" |
| V5 | 如果提供class_ids，shape必须为(N,) | `ValueError` | "class_ids must have shape (N,), got {shape}" |
| V6 | boxes/scores/class_ids的N必须一致 | `ValueError` | "boxes, scores, and class_ids must have the same length" |

**初始化逻辑**:
```python
def __init__(self, ...):
    # V1-V2: orig_shape验证
    if orig_shape is None:
        raise TypeError("orig_shape is required and cannot be None")
    if not (isinstance(orig_shape, tuple) and len(orig_shape) == 2):
        raise ValueError("orig_shape must be a tuple of (height, width)")

    # V3-V6: 数组shape验证（仅在非None时）
    if boxes is not None and boxes.shape[1:] != (4,):
        raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}")
    # ... 其他验证

    # 存储私有属性
    self._boxes = boxes
    self._scores = scores
    self._class_ids = class_ids
    self._orig_img = orig_img
    self._orig_shape = orig_shape
    self._names = names if names is not None else {}
    self._path = path
```

---

## State Transitions

Result对象是**不可变的**（Immutable），一旦创建后属性不可修改。状态转换通过创建新Result对象实现。

### State Diagram

```
┌─────────────────┐
│   初始化         │
│   __init__()    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   Ready State           │
│   (属性可读，不可写)      │
└─────────┬───────────────┘
          │
          ├─────────► 索引访问: result[i] ──► 新Result对象（单个检测）
          │
          ├─────────► 切片访问: result[1:3] ──► 新Result对象（子集）
          │
          ├─────────► 过滤: result.filter(...) ──► 新Result对象（过滤后）
          │
          ├─────────► 转换: result.to_supervision() ──► Supervision Detections
          │
          ├─────────► 可视化: result.plot() ──► numpy数组（标注图像）
          │
          └─────────► 废弃: result.to_dict() ──► dict（触发DeprecationWarning）
```

### Immutability Enforcement

**浅层不可变（Shallow Immutability）**:
- ❌ 禁止: `result.boxes = new_boxes`（AttributeError）
- ❌ 禁止: `result.scores = new_scores`（AttributeError）
- ✅ 允许: `result.boxes[0] = [10, 20, 30, 40]`（修改内部数组元素）
- ✅ 允许: `result.orig_img[0, 0] = [255, 0, 0]`（修改图像像素）

**Rationale**: 性能优先（Constitution III），避免深拷贝开销。用户需要显式创建新Result对象以表达"状态变更"意图。

---

## Derived State

### Length (`len()`)

```python
def __len__(self) -> int:
    """返回检测目标数量"""
    return len(self.boxes)  # boxes.shape[0]
```

**行为**:
- 空结果: `len(result) == 0`
- 有N个检测: `len(result) == N`

### Indexing (`__getitem__`)

**单个索引** (int):
```python
result[0]  # 返回新Result对象，仅包含第0个检测
```

**切片** (slice):
```python
result[1:3]  # 返回新Result对象，包含索引1和2的检测
result[:5]   # 前5个检测
result[-1]   # 最后一个检测
```

**返回值**:
- 新Result对象，共享`orig_img`、`orig_shape`、`names`（不变）
- 底层数组使用numpy视图（避免拷贝）

**越界行为**:
- 索引超出范围: 抛出`IndexError`
- 空结果索引: `result[0]` 抛出`IndexError`（numpy标准行为）

---

## Relationships

### Input Relationships

```
BaseORT._postprocess() → dict
    ↓
Result.__init__(**dict)
    ↓
Result对象
```

**数据流**:
1. BaseORT子类的`_postprocess()`方法返回字典：`{'boxes': ..., 'scores': ..., 'class_ids': ...}`
2. BaseORT.__call__()创建Result对象：`Result(**post_result, orig_img=img, orig_shape=shape, names=self.class_names)`
3. 返回Result对象给调用者

### Output Relationships

```
Result对象
    ├──► to_supervision() → supervision.Detections
    ├──► to_dict() → dict (deprecated)
    ├──► plot() → np.ndarray (annotated image)
    ├──► summary() → dict (statistics)
    └──► filter() → Result (new object)
```

---

## Validation Matrix

### Runtime Validation Points

| 操作 | 验证项 | 错误类型 | 处理策略 |
|------|-------|---------|----------|
| `__init__` | orig_shape非None | TypeError | 抛出异常 |
| `__init__` | boxes shape为(N, 4) | ValueError | 抛出异常 |
| `__init__` | 数组长度一致性 | ValueError | 抛出异常 |
| `plot()` | orig_img非None | ValueError | 抛出异常 |
| `show()` | orig_img非None | ValueError | 抛出异常 |
| `save()` | orig_img非None | ValueError | 抛出异常 |
| `__getitem__` | 索引范围有效 | IndexError | 抛出异常（numpy） |
| `filter()` | conf_threshold合法（0-1） | ValueError | 抛出异常 |

---

## Type Annotations

### Complete Type Signature

```python
from typing import Optional
import numpy as np
import numpy.typing as npt

class Result:
    def __init__(
        self,
        boxes: Optional[npt.NDArray[np.float32]] = None,
        scores: Optional[npt.NDArray[np.float32]] = None,
        class_ids: Optional[npt.NDArray[np.int32]] = None,
        orig_img: Optional[npt.NDArray[np.uint8]] = None,
        orig_shape: tuple[int, int] = None,
        names: Optional[dict[int, str]] = None,
        path: Optional[str] = None
    ) -> None: ...

    @property
    def boxes(self) -> npt.NDArray[np.float32]: ...

    @property
    def scores(self) -> npt.NDArray[np.float32]: ...

    @property
    def class_ids(self) -> npt.NDArray[np.int32]: ...

    @property
    def orig_img(self) -> Optional[npt.NDArray[np.uint8]]: ...

    @property
    def orig_shape(self) -> tuple[int, int]: ...

    @property
    def names(self) -> dict[int, str]: ...

    @property
    def path(self) -> Optional[str]: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int | slice) -> "Result": ...

    def plot(
        self,
        annotator_preset: Optional[str] = None
    ) -> npt.NDArray[np.uint8]: ...

    def show(self, window_name: str = "Result") -> None: ...

    def save(self, output_path: str) -> None: ...

    def filter(
        self,
        conf_threshold: Optional[float] = None,
        classes: Optional[list[int]] = None
    ) -> "Result": ...

    def to_supervision(self) -> "supervision.Detections": ...

    def summary(self) -> dict[str, any]: ...

    def to_dict(self) -> dict[str, np.ndarray]: ...  # Deprecated

    def numpy(self) -> "Result": ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...
```

---

## Data Model Summary

### Key Characteristics

1. **Immutable Container**: 属性只读，状态转换创建新对象
2. **Graceful Degradation**: None输入自动转换为空数组（boxes/scores/class_ids）
3. **Memory Efficient**: 索引和切片使用numpy视图，避免拷贝
4. **Type Safe**: 完整的类型提示，mypy兼容
5. **Validation First**: 初始化时验证数据一致性，运行时验证前提条件

### Performance Targets (from spec)

- 创建开销: <5ms（20个检测目标）
- 内存占用: <120%原始字典
- 可视化: <1秒（20个目标，640x640图像）

### Integration Points

- **Input**: BaseORT._postprocess()输出字典
- **Output**: Supervision Detections（可视化）、dict（向后兼容）、numpy数组（plot）

---

**数据模型版本**: 1.0.0
**最后更新**: 2025-11-05
**下一步**: 生成API合约（contracts/result_api.yaml）
