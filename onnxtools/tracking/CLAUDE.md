[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **tracking**

# 2D 多目标跟踪模块 (tracking)

## 模块职责

基于运动模型的 2D 多目标跟踪 (MOT)，统一以 `supervision.Detections` 为输入输出，可在 `InferencePipeline` 中即插即用。三个后端共享 `kalman.py` / `matching.py` 的批量化基元，`lap` 不可用时透明回落到 `scipy.optimize.linear_sum_assignment`。

| 算法 | 实现 | 主要特性 |
|------|------|----------|
| `bytetrack` | supervision 封装 `SupervisionByteTrack` | 零额外依赖，默认值，向后兼容 |
| `bytetrack_native` | 向量化 numpy `ByteTrackNative` | 三阶段关联 + lost-buffer，对齐官方 byte_tracker，支持 `class_aware` |
| `ocsort` | 向量化 numpy `OCSORT` | 含 OCM / OCR / ORU 观测中心改进 |

## 入口

```python
from onnxtools.tracking import create_tracker

tracker = create_tracker("bytetrack")          # supervision (默认)
tracker = create_tracker("bytetrack_native")   # 原生 ByteTrack
tracker = create_tracker("ocsort")             # 原生 OC-SORT

tracked = tracker.update(detections, frame)    # 返回 tracker_id 已填充的 sv.Detections
tracker.reset()                                # 重置后 ID 从 1 重新发号
```

`update(detections, frame)` 接受任意长度(含空)的 `sv.Detections`。`frame` 对纯运动模型是占位参数(预留 ReID)。

### BaseTracker 契约

```python
class BaseTracker(ABC):
    name: str = "base"
    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections: ...
    @abstractmethod
    def reset(self) -> None: ...
```

* 即使零检测帧也必须 tick(批量预测、老化 lost-buffer)。
* 返回的 `tracker_id` 为长度 N 整数数组(N 可能 < 输入长度，低分检测被丢弃)。
* 输出 `xyxy` 用**原始检测框**(非 Kalman 平滑值)，便于 `_align_tracker_ids` 通过 xyxy argmin 精确回填。

### 与 InferencePipeline 集成

```python
pipeline = InferencePipeline(
    model_type="rtdetr", model_path="models/rtdetr.onnx",
    enable_tracking=True,
    tracker_algo="bytetrack_native",
    tracker_extra_kwargs={"track_buffer": 60, "frame_rate": 30},
)
result_img, output_data = pipeline(frame)
# output_data[i]["tracker_id"] 由 pipeline._align_tracker_ids 回填
```

## kwargs 标准名 / 别名映射

所有 backend 用 `**_` 吸收未知 kwargs，可共用一份 kwargs 字典。

| 标准名 | supervision 别名 | bytetrack_native | ocsort |
|--------|------------------|-------------------|--------|
| `track_high_thresh` | `track_activation_threshold` | 高分检测切分阈值 | `det_thresh` |
| `track_buffer` | `lost_track_buffer` | lost 最大帧数 | `max_age` |
| `match_thresh` | `minimum_matching_threshold` | 一阶段 IoU 代价上限 | `iou_threshold = 1 - x` |
| `frame_rate` | `frame_rate` | 换算 buffer 帧数 | 忽略 |
| `class_aware` | — | 按类别隔离匹配 | 不支持 |
| `delta_t` / `inertia` | — | — | OC-SORT 专属 |

## 共享原语 (`kalman.py` / `matching.py`)

```python
from onnxtools.tracking.kalman import KalmanFilterXYAH, KalmanFilterXYSR
from onnxtools.tracking.matching import (
    box_iou_batch, box_giou_batch, iou_distance, fuse_score, linear_assignment,
)
```

* `KalmanFilterXYAH` (8D) / `KalmanFilterXYSR` (7D)：状态滤波；`multi_predict(means, covs)` 一次性批量预测 N 个 track(关键热点)。
* `box_iou_batch` / `box_giou_batch`：`[N,4]×[M,4]→[N,M]` 纯 numpy 广播 IoU/GIoU。
* `iou_distance`：`1 - IoU` 代价矩阵。
* `fuse_score`：ByteTrack 一阶段分数融合 `cost = 1 - (1-cost)*score`。
* `linear_assignment`：优先 `lap.lapjv`，退化 `scipy.linear_sum_assignment`。

## 模块结构

```
onnxtools/tracking/
├── __init__.py    # BaseTracker + SupervisionByteTrack + create_tracker 工厂
├── base.py        # TrackState 枚举 + TrackRecord
├── kalman.py      # KalmanFilterXYAH (8D) / KalmanFilterXYSR (7D)
├── matching.py    # IoU/GIoU 向量化 + 分数融合 + lap/scipy 分配器
├── bytetrack.py   # STrack 状态机 + ByteTrackNative (3 阶段关联)
└── ocsort.py      # KalmanBoxTracker + OCSORT (OCM/OCR/ORU)
```

## 算法行为要点

### ByteTrackNative 三阶段关联

1. 按 `track_high_thresh` 切分 `det_high` / `det_low`；`multi_predict(tracked + lost)` 批量 KF 预测。
2. **一阶段**：`det_high × (tracked+lost)`，`fuse_score` 加权 IoU，thresh=`match_thresh`(默认 0.8)。
3. **二阶段**：`det_low × 未匹配 tracked`，纯 IoU，thresh=0.5；失败 → `Lost`。
4. **三阶段**：未匹配高分 det × `unconfirmed`(仅激活一帧)，thresh=0.7；失败 → `Removed`。
5. 剩余高分且 `score > new_track_thresh` → 新建 `STrack(New, is_activated=False)`，需下一帧再命中才 emit。
6. `lost.frame_id - frame_id > buffer_size` → `Removed`。
7. `class_aware=True` 时在 IoU 代价矩阵中将异类位置置大值，实现类别隔离。

### OCSORT 观测中心改进

| 缩写 | 含义 | 实现位置 |
|------|------|----------|
| **OCM** | Observation-centric Momentum，代价 = `-(IoU + inertia × cos(v_track, v_det))` | `ocsort._angle_cost` |
| **OCR** | Observation-centric Recovery，一阶段失败的 track 用 `last_observation` 再做一次 IoU | `OCSORT.update` 二阶段 |
| **ORU** | Observation-centric Re-Update，重匹配时按最近真实观测重算 velocity 单位向量 | `KalmanBoxTracker.update` |

OCSORT 内部 `id` 从 0 计数，emit 时 `+1`，与 supervision 的 1-based 约定一致。

## 性能

x86 + 200 持续目标 + 1080p 下约 7-10 ms/帧(100+ FPS)，`lap.lapjv` 比 scipy fallback 略快。边缘设备超阈值时安装 `lap`(`uv pip install -e ".[tracking-fast]"`) 或减小 `max_age` / 上游收紧 conf。

## 依赖与测试

* `numpy>=2.2.6`、`supervision==0.26.1`、`scipy>=1.10.0`(必需)、`lap>=0.5.0`(可选 `[tracking-fast]`)。
* 单元测试覆盖 `kalman` / `matching` / `bytetrack_native` / `ocsort` / `factory` / `_align_tracker_ids` 回填，性能测试 `tests/performance/test_tracking_benchmark.py`(200 检测/帧 < 16ms)。

## 相关文件

* 实现：`__init__.py` · `base.py` · `kalman.py` · `matching.py` · `bytetrack.py` · `ocsort.py`
* 集成：[`onnxtools/pipeline.py`](../pipeline.py)(`enable_tracking` / `_align_tracker_ids`)、[`utils/supervision_labels.py`](../utils/supervision_labels.py)(`tracker_id` 前缀)
* 文档：[`docs/api/tracking.md`](../../docs/api/tracking.md) · [`docs/guides/tracking/`](../../docs/guides/tracking/index.md)

## FAQ

* **选哪个后端？** 零依赖/兼容历史 → `bytetrack`；严格论文行为/可调阈值/类别隔离 → `bytetrack_native`；快速运动或长遮挡 → `ocsort`。
* **多实例 ID 会冲突吗？** 不会，新建实例即重置类级计数器，ID 从 1 发号；但跨实例**并行**会重叠(仅为单 stream 设计)。
* **如何加新算法？** 在本目录新增继承 `BaseTracker` 的类，在 `create_tracker` 中延迟导入注册，并加入 `SUPPORTED_TRACKERS`，复用 `kalman.py` / `matching.py` 基元。
