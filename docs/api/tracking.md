# Tracking

2D 多目标跟踪 (MOT)。所有 tracker 以 `supervision.Detections` 为输入输出,可通过 [`create_tracker`](#onnxtools.tracking.create_tracker) 工厂统一创建,亦可在 [`InferencePipeline`](pipeline.md) 中开启 `enable_tracking=True` 即插即用。

## 选择哪个算法?

| 算法名 | 实现 | 适用场景 |
|--------|------|----------|
| `bytetrack` | supervision 内置封装 (`SupervisionByteTrack`) | 默认,零额外依赖,完全向后兼容 |
| `bytetrack_native` | 手写向量化 numpy (`ByteTrackNative`) | 需要严格的官方 ByteTrack 三阶段行为、可调阈值、类别隔离 |
| `ocsort` | 手写向量化 numpy (`OCSORT`) | 快速运动 / 长时间遮挡场景,含 OCM/OCR/ORU 观测中心改进 |

性能预算(200 持续目标 / 1080p / x86):约 7-10 ms/帧(>100 FPS)。安装可选依赖 `pip install onnxtools[tracking-fast]` 启用 `lap.lapjv` 加速分配。

## 快速开始

```python
from onnxtools.tracking import create_tracker
import supervision as sv

tracker = create_tracker("bytetrack_native", track_buffer=60, frame_rate=30)

for frame, raw_dets in stream:
    tracked = tracker.update(raw_dets, frame)
    # tracked.tracker_id 已就位 — 同一目标的连续帧 ID 保持一致
```

切换视频 / 摄像头时调用 `tracker.reset()`,ID 从 1 重新发号。

## 与 InferencePipeline 集成

```python
from onnxtools import InferencePipeline

pipeline = InferencePipeline(
    model_type="rtdetr",
    model_path="models/rtdetr.onnx",
    enable_tracking=True,
    tracker_algo="ocsort",
    tracker_extra_kwargs={"min_hits": 3, "max_age": 30, "inertia": 0.2},
)
result_img, output_data = pipeline(frame)
# output_data[i]["tracker_id"] 已经回填到每条检测/车牌记录
```

## kwargs 标准化

为了让一份共享 kwargs 字典适用于三种后端,所有 native tracker 都接受 supervision 风格的别名:

| 标准参数 | supervision 别名 | bytetrack_native 含义 | ocsort 含义 |
|----------|-------------------|----------------------|--------------|
| `track_high_thresh` / `det_thresh` | `track_activation_threshold` | 高/低分检测切分阈值 | 检测置信度下限 |
| `track_buffer` / `max_age` | `lost_track_buffer` | lost 状态最大帧数 | track 失踪最大帧数 |
| `match_thresh` / `iou_threshold` | `minimum_matching_threshold` | 一阶段代价上限 (cost = 1 - IoU) | iou_threshold = 1 − x |
| `frame_rate` | `frame_rate` | buffer 帧数换算 | 忽略 |

未知 kwargs 用 `**_` 吸收,不会抛错。

## 工厂函数

::: onnxtools.tracking.create_tracker
    options:
      heading_level: 3

## BaseTracker

::: onnxtools.tracking.BaseTracker
    options:
      heading_level: 3
      show_root_heading: true
      members_order: source

## SupervisionByteTrack (默认)

::: onnxtools.tracking.SupervisionByteTrack
    options:
      heading_level: 3
      members_order: source

## ByteTrackNative

手写向量化 ByteTrack。严格对齐官方 [byte_tracker.py](https://github.com/ifzhang/ByteTrack) 的 MOT17 SOTA 配置:三阶段关联(高分 IoU+fuse_score / 低分 IoU / unconfirmed)+ lost-buffer + new-track 双帧确认。

::: onnxtools.tracking.bytetrack.ByteTrackNative
    options:
      heading_level: 3
      show_root_heading: true
      members_order: source

### STrack 状态机

::: onnxtools.tracking.bytetrack.STrack
    options:
      heading_level: 4
      members_order: source
      members:
        - activate
        - re_activate
        - update
        - multi_predict
        - mark_lost
        - mark_removed
        - tlbr

## OCSORT

手写向量化 OC-SORT,参考 [noahcao/OC_SORT](https://github.com/noahcao/OC_SORT)。在经典 SORT 之上新增三个观测中心改进:

* **OCM** — 一阶段代价加入速度方向余弦一致性项(`inertia` 控制权重)。
* **OCR** — 一阶段未匹配的 track 用 `last_observation`(而非 KF 预测)再做 IoU 关联。
* **ORU** — track 重新匹配时基于最近真实观测重算 velocity 单位向量。

::: onnxtools.tracking.ocsort.OCSORT
    options:
      heading_level: 3
      show_root_heading: true
      members_order: source

### KalmanBoxTracker

::: onnxtools.tracking.ocsort.KalmanBoxTracker
    options:
      heading_level: 4
      members_order: source
      members:
        - predict
        - update
        - get_state

## 共享基元

### Kalman 滤波

::: onnxtools.tracking.kalman.KalmanFilterXYAH
    options:
      heading_level: 4
      members_order: source

::: onnxtools.tracking.kalman.KalmanFilterXYSR
    options:
      heading_level: 4
      members_order: source

### 关联 / 分配

::: onnxtools.tracking.matching.box_iou_batch
    options:
      heading_level: 4

::: onnxtools.tracking.matching.box_giou_batch
    options:
      heading_level: 4

::: onnxtools.tracking.matching.iou_distance
    options:
      heading_level: 4

::: onnxtools.tracking.matching.fuse_score
    options:
      heading_level: 4

::: onnxtools.tracking.matching.linear_assignment
    options:
      heading_level: 4

## 数据类型

::: onnxtools.tracking.base.TrackState
    options:
      heading_level: 4

::: onnxtools.tracking.base.TrackRecord
    options:
      heading_level: 4

## 性能建议

* **x86**:scipy fallback 已足够(~8 ms/帧 @ 200 检测)。
* **边缘设备(Jetson)**:`uv pip install -e ".[tracking-fast]"` 启用 `lap.lapjv`,Hungarian 求解快 ~3-5×。
* **缩减 pool**:降低 `lost_track_buffer` / `max_age`,或在上游用 NMS 削减进入跟踪的检测数。
* **类别隔离**:`bytetrack_native(class_aware=True)` 用 cost mask 屏蔽跨类匹配,适合多类共存场景。

更多内部约定与扩展指引参见仓库内 `onnxtools/tracking/CLAUDE.md`(GitHub 浏览:
<https://github.com/yyq19990828/onnxtools/blob/main/onnxtools/tracking/CLAUDE.md>)。
