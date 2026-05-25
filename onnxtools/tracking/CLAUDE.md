[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **tracking**

# 2D 多目标跟踪模块 (tracking)

## 模块职责

提供基于运动模型的 2D 多目标跟踪 (MOT) 算法,统一以 `supervision.Detections` 作为输入输出,可在 `InferencePipeline` 中即插即用。当前包含三个后端:

| 算法 | 来源 | 主要特性 |
|------|------|----------|
| `bytetrack` | supervision 内置封装 (`SupervisionByteTrack`) | 零额外依赖,默认值,向后兼容路径 |
| `bytetrack_native` | 手写向量化 numpy 实现 (`ByteTrackNative`) | 三阶段关联 + lost-buffer,严格对齐官方 byte_tracker 行为 |
| `ocsort` | 手写向量化 numpy 实现 (`OCSORT`) | 含 OCM(观测中心动量) / OCR(观测中心恢复) / ORU(观测中心重更新) |

所有 native 后端共享 `kalman.py` 与 `matching.py` 中的批量化基元(`KalmanFilterXYAH`/`XYSR`、`box_iou_batch`、`linear_assignment`),并在 `lap` 不可用时透明回落到 `scipy.optimize.linear_sum_assignment`。

## 入口和启动

```python
from onnxtools.tracking import create_tracker

tracker = create_tracker("bytetrack")          # supervision (默认)
tracker = create_tracker("bytetrack_native")   # 原生向量化 ByteTrack
tracker = create_tracker("ocsort")             # 原生 OC-SORT

tracked = tracker.update(detections, frame)    # tracker_id 已写入
tracker.reset()                                # 重置后 ID 从 1 重新发号
```

`update(detections, frame)` 接受任意长度(包括空)的 `sv.Detections`,返回 `tracker_id` 已填充的 `sv.Detections`。`frame` 对纯运动模型 tracker 是占位参数,未来 ReID 类后端会用到。

### 与 InferencePipeline 集成

```python
from onnxtools import InferencePipeline

pipeline = InferencePipeline(
    model_type="rtdetr", model_path="models/rtdetr.onnx",
    enable_tracking=True,
    tracker_algo="bytetrack_native",      # 任选三种
    tracker_extra_kwargs={"track_buffer": 60, "frame_rate": 30},
)
result_img, output_data = pipeline(frame)
# output_data[i]["tracker_id"] 已就位 (回填顺序由 pipeline._align_tracker_ids 处理)
```

## 外部接口

### 1. `BaseTracker` 抽象基类

```python
class BaseTracker(ABC):
    name: str = "base"

    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections: ...

    @abstractmethod
    def reset(self) -> None: ...
```

实现契约:
* 即使在零检测帧上也必须 tick(批量预测、老化 lost-buffer)。
* 返回的 `sv.Detections.tracker_id` 必须为长度 N 的整数数组(N 可能 < 输入长度,因为低分检测可能被丢弃)。
* 输出 `xyxy` 建议使用 **原始检测框**(非 Kalman 平滑值),以便 `InferencePipeline._align_tracker_ids` 通过 xyxy argmin 精确回填到 `output_data`。

### 2. `create_tracker(algo, **kwargs)` 工厂

| 参数(标准名) | 别名(supervision 风格) | bytetrack_native | ocsort |
|----------------|------------------------|-------------------|--------|
| `track_high_thresh` | `track_activation_threshold` | 高分检测切分阈值 | `det_thresh` |
| `track_buffer` | `lost_track_buffer` | lost 状态最大帧数 | `max_age` |
| `match_thresh` | `minimum_matching_threshold` | 一阶段 IoU 关联代价上限 | `iou_threshold = 1 - x` |
| `frame_rate` | `frame_rate` | 用于换算 buffer 帧数 | 忽略 |
| `class_aware` | — | 是否按类别隔离匹配 | 不支持 |
| `delta_t` / `inertia` | — | — | OC-SORT 专属 |

所有 backend 用 `**_` 吸收未知 kwargs,因此可以传一份共享 kwargs 字典给三个后端。

### 3. 共享原语 (`kalman.py` / `matching.py`)

```python
from onnxtools.tracking.kalman import KalmanFilterXYAH, KalmanFilterXYSR
from onnxtools.tracking.matching import (
    box_iou_batch,    # [N,4] x [M,4] -> [N,M] IoU, 纯 numpy 广播
    box_giou_batch,
    iou_distance,     # 1 - IoU 代价矩阵
    fuse_score,       # ByteTrack 一阶段分数融合: cost = 1 - (1-cost) * score
    linear_assignment,  # 优先 lap.lapjv, 退化 scipy.linear_sum_assignment
)
```

`KalmanFilterXYAH` 提供 `multi_predict(means, covs)` 一次性批量预测 N 个 track,关键热点。

## 模块结构

```
onnxtools/tracking/
├── __init__.py         # BaseTracker + SupervisionByteTrack + create_tracker 工厂
├── base.py             # TrackState 枚举 + TrackRecord 轻量描述
├── kalman.py           # KalmanFilterXYAH (8D) / KalmanFilterXYSR (7D)
├── matching.py         # IoU/GIoU 向量化 + 分数融合 + lap/scipy 分配器
├── bytetrack.py        # STrack 状态机 + ByteTrackNative (3 阶段关联)
├── ocsort.py           # KalmanBoxTracker + OCSORT (OCM/OCR/ORU)
└── _archive/           # 历史 BoxMOT 适配器 (PyTorch 依赖被归档)
```

## 算法行为要点

### ByteTrackNative 三阶段关联

1. 按 `track_high_thresh` 切分 `det_high` / `det_low`。
2. `multi_predict(tracked + lost)` 批量 KF 预测。
3. **一阶段**:`det_high × (tracked+lost)`,`fuse_score` 加权 IoU,thresh=`match_thresh` (默认 0.8)。
4. **二阶段**:`det_low × 未匹配的 tracked`,纯 IoU,thresh=0.5;失败 → `Lost`。
5. **三阶段**:未匹配高分 det × `unconfirmed`(只激活了一帧的新 track),thresh=0.7;失败 unconfirmed → `Removed`。
6. 剩余高分且 `score > new_track_thresh` 的 det 创建 `STrack(state=New, is_activated=False)`,**需下一帧再被命中才正式 emit**。
7. `lost.frame_id - frame_id > buffer_size` → `Removed`。

`class_aware=True` 时在 IoU 代价矩阵中将不同类别的位置置为大值,实现类别隔离匹配。

### OCSORT 的三个观测中心改进

| 缩写 | 含义 | 实现位置 |
|------|------|----------|
| **OCM** | Observation-centric Momentum — 匹配代价 = `-(IoU + inertia × cos(velocity_track, velocity_det))` | `ocsort._angle_cost` |
| **OCR** | Observation-centric Recovery — 一阶段失败的 track 用 `last_observation`(而非 KF 预测)再做一次 IoU 关联 | `OCSORT.update` 第二阶段 |
| **ORU** | Observation-centric Re-Update — track 重新匹配时根据最近真实观测重算 velocity 单位向量 | `KalmanBoxTracker.update` |

## 性能预算

在 x86 + 200 持续目标 + 1080p 场景下(50 帧均值):

| 后端 | scipy fallback | lap.lapjv |
|------|----------------|-----------|
| `bytetrack_native` | ~8 ms/帧 | ~7 ms/帧 |
| `ocsort` | ~10 ms/帧 | ~8 ms/帧 |
| `bytetrack` (supervision) | ~10 ms/帧 | n/a |

对应 100+ FPS,远高于实时阈值。Jetson 上若超 15ms 建议安装 `lap`。

```bash
uv pip install -e ".[tracking-fast]"   # 安装 lap (可选)
```

## 关键依赖

* `numpy>=2.2.6` — 核心数据
* `supervision==0.26.1` — 检测/跟踪结果容器
* `scipy>=1.10.0` — Hungarian 分配求解 (linear_sum_assignment)
* `lap>=0.5.0` *(可选, `[tracking-fast]` extras)* — lapjv 加速分配

## 测试和质量

| 测试文件 | 覆盖 |
|----------|------|
| [`tests/unit/test_tracking_kalman.py`](../../tests/unit/test_tracking_kalman.py) | initiate 形状、multi_predict ≡ predict loop、update 后协方差正定、门控距离非负、空输入 |
| [`tests/unit/test_tracking_matching.py`](../../tests/unit/test_tracking_matching.py) | IoU 数值正确性、广播 shape、空输入、GIoU ≤ IoU、分配器三种边界、fuse_score 单调性 |
| [`tests/unit/test_tracking_bytetrack_native.py`](../../tests/unit/test_tracking_bytetrack_native.py) | ID 稳定性、buffer 内 ID 保留、buffer 过期换新 ID、reset、kwargs 别名 |
| [`tests/unit/test_tracking_ocsort.py`](../../tests/unit/test_tracking_ocsort.py) | min_hits 延迟 emit、max_age 过期、velocity 单位向量、reset、kwargs 别名 |
| [`tests/unit/test_tracking_factory.py`](../../tests/unit/test_tracking_factory.py) | 三种算法注册、未知算法 ValueError、kwargs 透传 |
| [`tests/unit/test_tracking.py`](../../tests/unit/test_tracking.py) | `pipeline._align_tracker_ids` 回填契约、`supervision_labels` ID 前缀 |
| [`tests/performance/test_tracking_benchmark.py`](../../tests/performance/test_tracking_benchmark.py) | 200 检测/帧 < 16ms 上限 |

总计 **56 个测试**,全部通过。

## 常见问题 (FAQ)

### Q: 选哪个后端?
* **稳定可用、零依赖、与历史代码完全兼容** → `bytetrack` (supervision)。
* **想要严格的 ByteTrack 论文行为、可调三阶段阈值、类别隔离** → `bytetrack_native`。
* **目标在快速运动或长时间遮挡场景多** → `ocsort`(运动方向一致性 + 观测重更新)。

### Q: 为何 OCSORT 输出的 ID 从 1 开始而不是 0?
内部 `KalmanBoxTracker.id` 从 0 开始计数,emit 时统一 `+1`,以保持与 supervision/ByteTrack 的 1-based 约定一致(便于 `_align_tracker_ids` 的回填和前端显示)。

### Q: 在多个 tracker 实例之间 ID 会冲突吗?
不会。`ByteTrackNative.__init__` 和 `OCSORT.__init__` 都会重置类级计数器,**新建实例 = ID 从 1 重新发号**。但跨实例 *并行* 使用会重叠 — 当前为单 stream 场景设计。

### Q: 如何自定义新算法?
1. 在 `onnxtools/tracking/` 新增 `mytracker.py`,继承 `BaseTracker`,实现 `update` 与 `reset`。
2. 在 `__init__.py` 的 `create_tracker` 工厂中加 `if algo == "mytracker": from .mytracker import MyTracker; return MyTracker(**kwargs)`(延迟导入,避免无关依赖)。
3. 把 `"mytracker"` 加入 `SUPPORTED_TRACKERS`。
4. 复用 `kalman.py` / `matching.py` 的批量化基元。

### Q: 边缘设备性能不达标怎么办?
1. `uv pip install -e ".[tracking-fast]"` 安装 `lap`。
2. 减少 `max_age` / `lost_track_buffer`,缩小 pool 规模。
3. 在上游用 NMS 或 conf 阈值削减进入跟踪的检测数。

## 相关文件列表

### 实现
* [`__init__.py`](__init__.py) — BaseTracker / SupervisionByteTrack / 工厂
* [`base.py`](base.py) — TrackState / TrackRecord
* [`kalman.py`](kalman.py) — 两个 KF 类
* [`matching.py`](matching.py) — IoU/分配/fuse
* [`bytetrack.py`](bytetrack.py) — STrack + ByteTrackNative
* [`ocsort.py`](ocsort.py) — KalmanBoxTracker + OCSORT

### 集成点
* [`onnxtools/pipeline.py`](../pipeline.py) — `InferencePipeline.enable_tracking` / `_align_tracker_ids`
* [`onnxtools/utils/supervision_labels.py`](../utils/supervision_labels.py) — `tracker_id` 前缀格式化

### 文档
* [`docs/api/tracking.md`](../../docs/api/tracking.md) — 用户 API 参考(mkdocstrings 自动生成)
* [`docs/guides/tracking/`](../../docs/guides/tracking/index.md) — **2D 跟踪学习手册**(概念/卡尔曼/匈牙利/评测指标 + 传统方法 + 近五年方法,mermaid 图文并茂)

## 变更日志 (Changelog)

**2026-05-18** — 手写高性能跟踪后端
- ✅ 新增 `ByteTrackNative`:严格对齐官方 byte_tracker 三阶段关联 + lost-buffer + class_aware
- ✅ 新增 `OCSORT`:含 OCM/OCR/ORU 的观测中心 SORT
- ✅ 共享 `kalman.py`(XYAH 8D / XYSR 7D)与 `matching.py`(IoU/分配/fuse)向量化基元
- ✅ 可选 `lap.lapjv` 加速分配,scipy 透明回退
- ✅ kwargs 标准化映射,与 supervision/ByteTrack 别名兼容
- ✅ 56 个单元/性能测试覆盖
- ✅ `pyproject.toml` 新增 `scipy` 必需依赖、`[tracking-fast]` 可选依赖

**2025-11-13** — 初始版本:仅 `SupervisionByteTrack` 一种后端

---

*模块路径: `onnxtools/tracking/`*
*最后更新: 2026-05-18*
