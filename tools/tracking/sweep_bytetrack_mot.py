"""ByteTrack 超参数搜索：MOT 指标(有标注) + 3Hz 无标注代理指标，找 top5。

设计要点
========
* **检测模型**：``vehicle_det_detr_batch4``(RT-DETR)。检测后**只保留行人 + 核心非机动车**
  ``{9 pedestrian, 5 bicycle, 6 cyclist, 7 tricycle}``，其余(机动车/cone/plate…)在喂 tracker
  前丢弃，并把保留类**重映射成单一 VRU 类**——这样 ``bicycle↔cyclist`` 的类别抖动天然消解，
  GT 的 ``{1 行人, 2 非机动车}`` 也池化为 VRU(类无关 HOTA 匹配)。
* **两类信号**：
    - **MOT 指标**(有真值、5Hz)：``MOTEvaluator`` 出 HOTA/MOTA/IDF1 —— 当**优化目标**。
    - **3Hz 代理**(无真值)：``short_id_ratio`` 等碎片化信号 —— 当**部署帧率下的护栏**。
* **目标函数 (门控 + HOTA)**：先用代理 ``short_id_ratio``(及 ``num_ids`` 合理性)做门，
  筛掉"会碎"的参数；幸存者纯按 HOTA 排序取 top5(幸存不足 5 个自动放宽门限)。
* **检测缓存**：检测贵、tracker 廉价 —— 每序列/每视频只检测一次，48 组参数复用同一缓存。

产出(默认落 ``runs/bytetrack_sweep/``)
* ``result.json`` —— 全部组指标 + 门控 + 排名。
* ``REPORT.md``   —— 搜索设置 / 空间 / top5 对比表 / 逐序列分解 / 推荐。
* ``viz/<seq>/rank<k>_<params>.mp4`` —— top5 × MOT 5 序列 = 25 个带轨迹+ID 的可视化。

用法::

    .venv/bin/python tools/tracking/sweep_bytetrack_mot.py
    .venv/bin/python tools/tracking/sweep_bytetrack_mot.py --skip-viz --max-mot-frames 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from configparser import ConfigParser
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from onnxtools import create_detector, setup_logger  # noqa: E402
from onnxtools.eval import MOTEvaluator  # noqa: E402
from onnxtools.eval.mot_data import MOTSequence  # noqa: E402
from onnxtools.tracking import create_tracker  # noqa: E402
from onnxtools.tracking.class_voting import ClassVotingTracker  # noqa: E402

logger = logging.getLogger("sweep_bytetrack")

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
MODEL = "models/rfdetr-medium_20260629_d_unified.onnx"
MODEL_TYPE = "rfdetr_unified"
MOT_ROOT = "data/track/MOT_dataset"
PROXY_DIR = "data/track/proxy_videos"

DET_CONF_FLOOR = 0.4  # <0.4 是噪声，直接丢
KEEP_CLASSES = (9, 5, 6, 7)  # pedestrian, bicycle, cyclist, tricycle
VRU_CLASS = 0  # 全部重映射为单一 VRU 类

MOT_FRAME_RATE = 5  # MOT 数据集原生帧率
PROXY_FPS = 3.0  # 工程部署帧率

# 搜索空间。新检测模型(rfdetr-medium_20260629_d)误检跳变显著降低，
# 因此较旧模型扩大三个轴的范围：
#   - high/new 向下加 0.5(检测可信→可放更多高质量框)，向上加 0.8(更严格 stage-1)
#   - match_thresh 向下加 0.6、向上加 0.95(更紧/更松两端都试)
#   - track_buffer 向上加 30/60(检测更稳→可容忍更长真实遮挡而非靠短 buffer 早断)
TRACK_LOW_THRESH = 0.4  # 固定
HIGH_NEW_PAIRS = (
    (0.5, 0.5),
    (0.5, 0.6),
    (0.6, 0.6),
    (0.6, 0.7),
    (0.7, 0.7),
    (0.7, 0.8),
    (0.8, 0.8),
)
MATCH_THRESH = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
TRACK_BUFFER = (3, 9, 15, 30, 60)  # 30fps 名义值，按各源 fps 缩放; 60≈12s@5Hz

# 门控
GATE_SHORT_ID = 0.30  # short_id_ratio 上限
GATE_NUMID_MULT = 3.0  # num_ids 不超过最小值的倍数


# ----------------------------------------------------------------------------
# 检测 + 过滤
# ----------------------------------------------------------------------------
def detect_filtered(detector, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跑检测，保留目标类。返回 ``(xyxy[N,4], conf[N], cls[N])``。

    保留 detector 的原始类别 id（不池化为 VRU），这样 class_aware 关联和
    ClassVotingTracker 在 sweep 中是有效的；评估时统一调用 ``classes=None``
    使 HOTA 类无关，因此 GT 的类体系与预测的类体系无须对齐。
    """
    r = detector(image)
    boxes = np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(r.scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(r.class_ids, dtype=int).reshape(-1)
    keep = np.isin(cls, KEEP_CLASSES)
    return boxes[keep], scores[keep], cls[keep]


def make_dets(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> sv.Detections:
    if len(xyxy) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=xyxy.astype(float),
        confidence=conf.astype(float),
        class_id=cls.astype(int),
    )


def make_tracker(params: dict, frame_rate: int, *, class_aware: bool, class_voting: bool):
    """构建 sweep 所需的 tracker：bytetrack_native + 可选 ClassVotingTracker 包装。"""
    inner = create_tracker(
        "bytetrack_native",
        **tracker_kwargs(params, frame_rate),
        class_aware=class_aware,
    )
    if class_voting:
        return ClassVotingTracker(inner, decay=1.0)
    return inner


# ----------------------------------------------------------------------------
# 检测缓存
# ----------------------------------------------------------------------------
def read_seqmap(mot_root: Path) -> list[str]:
    seqmap = mot_root / "seqmap.txt"
    names = []
    for line in seqmap.read_text().splitlines():
        s = line.strip()
        if s and s.lower() != "name":
            names.append(s)
    return names


def cache_mot_detections(detector, mot_root: Path, max_frames: int | None) -> dict[str, dict]:
    """``{seq: {"len": L, "frames": {frame(1-based): (xyxy, conf)}}}``。"""
    cache: dict[str, dict] = {}
    seq_names = read_seqmap(mot_root)
    logger.info(f"MOT 检测缓存: {len(seq_names)} 个序列")
    total_dets = 0
    t0 = time.time()
    for si, name in enumerate(seq_names, 1):
        ini = ConfigParser()
        ini.read(mot_root / name / "seqinfo.ini")
        n = int(ini["Sequence"]["seqLength"])
        im_dir = mot_root / name / ini["Sequence"].get("imDir", "img1")
        im_ext = ini["Sequence"].get("imExt", ".jpg")
        limit = min(n, max_frames) if max_frames else n
        frames: dict[int, tuple] = {}
        seq_dets = 0
        ts = time.time()
        for f in range(1, limit + 1):
            img = cv2.imread(str(im_dir / f"{f:06d}{im_ext}"))
            if img is None:
                continue
            xyxy, conf, cls = detect_filtered(detector, img)
            frames[f] = (xyxy, conf, cls)
            seq_dets += len(xyxy)
        cache[name] = {"len": limit, "frames": frames}
        total_dets += seq_dets
        elapsed = time.time() - ts
        logger.info(
            f"  [{si}/{len(seq_names)}] {name}: {len(frames)} 帧, "
            f"{seq_dets} 个 VRU 检测, {elapsed:.1f}s ({len(frames) / max(elapsed, 1e-6):.1f} fps)"
        )
    logger.info(f"MOT 缓存完成: {total_dets} 个 VRU 检测, 总耗时 {time.time() - t0:.1f}s")
    return cache


def select_proxy_videos(proxy_dir: Path) -> list[Path]:
    vids = sorted(p for p in proxy_dir.glob("*.mp4") if "heizhima" not in p.name.lower())
    return vids


def cache_proxy_detections(detector, videos: list[Path]) -> dict[str, list]:
    """``{video_name: [(xyxy, conf), ...]}``，按 ~3Hz 采样。"""
    cache: dict[str, list] = {}
    logger.info(f"3Hz 代理视频检测缓存: {len(videos)} 个视频")
    total_dets = 0
    t0 = time.time()
    for vi, v in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / PROXY_FPS))
        dets, idx, vdets = [], 0, 0
        ts = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                xyxy, conf, cls = detect_filtered(detector, frame)
                dets.append((xyxy, conf, cls))
                vdets += len(xyxy)
            idx += 1
        cap.release()
        cache[v.name] = dets
        total_dets += vdets
        elapsed = time.time() - ts
        logger.info(
            f"  [{vi}/{len(videos)}] {v.name[:50]}: " f"{len(dets)} 采样帧, {vdets} 检测, step={step}, {elapsed:.1f}s"
        )
    logger.info(f"代理缓存完成: {total_dets} 个 VRU 检测, 总耗时 {time.time() - t0:.1f}s")
    return cache


# ----------------------------------------------------------------------------
# 在缓存上跑 tracker
# ----------------------------------------------------------------------------
def tracker_kwargs(params: dict, frame_rate: int) -> dict:
    high, new = params["track_high_thresh"], params["new_track_thresh"]
    return dict(
        track_high_thresh=high,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=new,
        match_thresh=params["match_thresh"],
        track_buffer=params["track_buffer"],
        frame_rate=frame_rate,
    )


def track_mot(mot_cache: dict, params: dict, *, class_aware: bool, class_voting: bool) -> dict[str, MOTSequence]:
    """在缓存的 MOT 检测上跑 ByteTrack(+ 可选 class_aware / 投票)，产出 ``{seq: MOTSequence}``。

    评估侧的预测类列统一写为 VRU_CLASS——评估调用 ``classes=None`` 走类无关 HOTA，
    类列只是占位；class_aware/voting 只影响 tracker 内部的关联决策。
    """
    preds: dict[str, MOTSequence] = {}
    for name, seq in mot_cache.items():
        tracker = make_tracker(params, MOT_FRAME_RATE, class_aware=class_aware, class_voting=class_voting)
        frames_out: dict[int, np.ndarray] = {}
        for f in range(1, seq["len"] + 1):
            if f not in seq["frames"]:
                continue
            xyxy, conf, cls = seq["frames"][f]
            tracked = tracker.update(make_dets(xyxy, conf, cls), None)
            if tracked.tracker_id is None or len(tracked) == 0:
                continue
            txyxy = tracked.xyxy
            rows = np.column_stack(
                [
                    tracked.tracker_id.astype(float),
                    txyxy[:, 0],
                    txyxy[:, 1],
                    txyxy[:, 2] - txyxy[:, 0],
                    txyxy[:, 3] - txyxy[:, 1],
                    np.ones(len(tracked)),
                    np.full(len(tracked), VRU_CLASS, dtype=float),
                ]
            )
            frames_out[f] = rows
        preds[name] = MOTSequence(name=name, frames=frames_out)
    return preds


def proxy_metrics(proxy_cache: dict, params: dict, *, class_aware: bool, class_voting: bool) -> dict:
    """3Hz 无监督代理指标，跨视频平均。"""
    per_video = []
    for dets_list in proxy_cache.values():
        tracker = make_tracker(params, int(PROXY_FPS), class_aware=class_aware, class_voting=class_voting)
        id_frames: dict[int, int] = defaultdict(int)
        for xyxy, conf, cls in dets_list:
            tracked = tracker.update(make_dets(xyxy, conf, cls), None)
            ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
            for tid in ids:
                id_frames[int(tid)] += 1
        lifetimes = list(id_frames.values())
        if not lifetimes:
            per_video.append((0, 0.0, 0.0))
            continue
        short_ratio = sum(1 for v in lifetimes if v <= 2) / len(lifetimes)
        per_video.append((len(lifetimes), float(np.mean(lifetimes)), short_ratio))
    num_ids = float(np.mean([x[0] for x in per_video]))
    mean_life = float(np.mean([x[1] for x in per_video]))
    short_id_ratio = float(np.mean([x[2] for x in per_video]))
    return {
        "proxy_num_ids": num_ids,
        "proxy_mean_life": mean_life,
        "proxy_short_id_ratio": short_id_ratio,
    }


# ----------------------------------------------------------------------------
# 网格 / 排名
# ----------------------------------------------------------------------------
def build_grid() -> list[dict]:
    grid = []
    for (high, new), mt, tb in product(HIGH_NEW_PAIRS, MATCH_THRESH, TRACK_BUFFER):
        grid.append(
            {
                "track_high_thresh": high,
                "new_track_thresh": new,
                "match_thresh": mt,
                "track_buffer": tb,
            }
        )
    return grid


def rank_with_gate(rows: list[dict], top_k: int) -> tuple[list[dict], float]:
    """门控 + HOTA 排名。返回 ``(ranked_rows, used_gate_short_id)``。"""
    min_numid = min(r["proxy_num_ids"] for r in rows) or 1.0
    gate = GATE_SHORT_ID
    # 自动放宽，确保至少 top_k 个幸存
    while True:
        for r in rows:
            r["pass_gate"] = r["proxy_short_id_ratio"] <= gate and r["proxy_num_ids"] <= GATE_NUMID_MULT * min_numid
        survivors = [r for r in rows if r["pass_gate"]]
        if len(survivors) >= top_k or gate >= 1.0:
            break
        gate = round(gate + 0.05, 2)
    survivors = sorted((r for r in rows if r["pass_gate"]), key=lambda r: r["HOTA"], reverse=True)
    rest = sorted((r for r in rows if not r["pass_gate"]), key=lambda r: r["HOTA"], reverse=True)
    ranked = survivors + rest
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked, gate


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def render_seq(
    mot_root: Path,
    mot_cache: dict,
    seq: str,
    params: dict,
    out_path: Path,
    *,
    class_aware: bool,
    class_voting: bool,
) -> None:
    """用给定参数重跑 tracker，并把带轨迹+ID 的标注写成 mp4。"""
    ini = ConfigParser()
    ini.read(mot_root / seq / "seqinfo.ini")
    sec = ini["Sequence"]
    w, h = int(sec["imWidth"]), int(sec["imHeight"])
    im_dir = mot_root / seq / sec.get("imDir", "img1")
    im_ext = sec.get("imExt", ".jpg")
    seqc = mot_cache[seq]

    box_a = sv.BoxAnnotator(thickness=2)
    label_a = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_a = sv.TraceAnnotator(trace_length=30, thickness=2)

    tracker = make_tracker(params, MOT_FRAME_RATE, class_aware=class_aware, class_voting=class_voting)
    info = sv.VideoInfo(width=w, height=h, fps=MOT_FRAME_RATE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    empty_cls = np.empty((0,), dtype=int)
    with sv.VideoSink(str(out_path), info) as sink:
        for f in range(1, seqc["len"] + 1):
            img = cv2.imread(str(im_dir / f"{f:06d}{im_ext}"))
            if img is None:
                continue
            xyxy, conf, cls = seqc["frames"].get(f, (np.empty((0, 4)), np.empty((0,)), empty_cls))
            tracked = tracker.update(make_dets(xyxy, conf, cls), img)
            frame = img.copy()
            if tracked.tracker_id is not None and len(tracked):
                labels = [f"#{int(t)}" for t in tracked.tracker_id]
                frame = trace_a.annotate(frame, tracked)
                frame = box_a.annotate(frame, tracked)
                frame = label_a.annotate(frame, tracked, labels=labels)
            sink.write_frame(frame)


def param_tag(p: dict) -> str:
    return f"h{p['track_high_thresh']}_n{p['new_track_thresh']}_m{p['match_thresh']}_b{p['track_buffer']}"


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def write_report(out_dir: Path, top5: list[dict], gate: float, ctx: dict) -> None:
    lines = []
    lines.append("# ByteTrack 超参数搜索报告\n")
    lines.append("## 1. 搜索设置\n")
    lines.append(f"- 检测模型：`{MODEL}`(conf floor={DET_CONF_FLOOR})")
    lines.append("- 目标类(检测→VRU)：pedestrian(9) + bicycle(5)/cyclist(6)/tricycle(7)，其余过滤")
    lines.append(f"- MOT 评估：`{MOT_ROOT}`，{ctx['n_seq']} 序列，原生 {MOT_FRAME_RATE}Hz，池化为单一 VRU 类")
    lines.append(f"- 3Hz 代理护栏：{ctx['n_proxy']} 个视频(每机位 5 个)")
    lines.append(f"- 网格组合数：{ctx['n_grid']}")
    lines.append("")
    lines.append("## 2. 搜索空间\n")
    lines.append(f"- `track_low_thresh` = {TRACK_LOW_THRESH}(固定，低质量框下限)")
    lines.append(f"- `(track_high_thresh, new_track_thresh)` ∈ {list(HIGH_NEW_PAIRS)}(new≥high)")
    lines.append(f"- `match_thresh` ∈ {list(MATCH_THRESH)}")
    lines.append(f"- `track_buffer` ∈ {list(TRACK_BUFFER)}(30fps 名义值，按源 fps 缩放)")
    lines.append("")
    lines.append("## 3. 目标函数(门控 + HOTA)\n")
    lines.append(
        f"先用 3Hz 代理 `short_id_ratio ≤ {gate}` 且 `num_ids ≤ {GATE_NUMID_MULT}×min` 做门，"
        "幸存者按 MOT **HOTA** 排序取 top5(门限自动放宽以保证 ≥5 个幸存)。\n"
    )
    lines.append("## 4. 指标说明\n")
    lines.append("### MOT 指标（有真值，越接近 GT 越高）\n")
    lines.append(
        "- **HOTA**（Higher Order Tracking Accuracy，0–100，越高越好，**主排序轴**）："
        "几何平均 `√(DetA · AssA)`，同时反映“检测准不准”(DetA) 与“ID 关联对不对”(AssA)，"
        "对检测与关联**等权**，是当前 MOT 公认主指标。本搜索改善的主要是 AssA 部分"
        "（检测精度已被模型质量封顶）。"
    )
    lines.append(
        "- **MOTA**（Multiple Object Tracking Accuracy，≤100，越高越好，**可为负**）："
        "`1 − (FN + FP + IDSW) / GT`，把漏检、误检、ID 切换三类错误一并扣分，**对检测质量极敏感**——"
        "误检多时直接为负。负值不代表跟踪坏，多半是检测器在该序列 FP 偏高。"
    )
    lines.append(
        "- **IDF1**（ID F1-score，0–100，越高越好）："
        "以“轨迹 ID 全程匹配正确”为核心的 F1（`2·IDTP / (2·IDTP + IDFP + IDFN)`），"
        "比 MOTA 更看重 **ID 在时间上的持久一致**——同一物体 ID 不变、不串号则高。"
    )
    lines.append("")
    lines.append("### 3Hz 无标注代理指标（部署帧率下的“别碎”护栏，跨所有代理视频平均）\n")
    lines.append(
        "- **proxy_short_id_ratio**（0–1，**越低越好**，**门控主信号**）："
        "生命 ≤2 帧就消失的轨迹占比 = **碎片化率**。高 = ID 频繁断裂重生（3Hz 的典型失败模式）。"
    )
    lines.append(
        "- **proxy_num_ids**（越少越好，相对比较）："
        "平均每个视频发出的唯一 ID 总数。偏高通常意味同一物体被拆成多个 ID（**ID 膨胀**）。"
    )
    lines.append("- **proxy_mean_life**（越高越好）：ID 平均存活帧数。越长 = 轨迹越稳定持久。")
    lines.append("")
    lines.append("## 5. Top5 参数\n")
    lines.append(
        "| 排名 | high | new | match | buffer | HOTA | MOTA | IDF1 | "
        "proxy_short_id | proxy_num_ids | proxy_mean_life | 过门 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in top5:
        lines.append(
            f"| {r['rank']} | {r['track_high_thresh']} | {r['new_track_thresh']} | "
            f"{r['match_thresh']} | {r['track_buffer']} | {r['HOTA']:.2f} | {r['MOTA']:.2f} | "
            f"{r['IDF1']:.2f} | {r['proxy_short_id_ratio']:.3f} | {r['proxy_num_ids']:.1f} | "
            f"{r['proxy_mean_life']:.1f} | {'✅' if r['pass_gate'] else '⚠️'} |"
        )
    lines.append("")
    lines.append("## 6. 推荐参数(rank 1)逐序列指标\n")
    best = top5[0]
    lines.append("| 序列 | HOTA | MOTA | IDF1 |")
    lines.append("|---|---|---|---|")
    for seq, mm in best.get("per_sequence", {}).items():
        lines.append(
            f"| {seq} | {mm.get('HOTA', float('nan')):.2f} | "
            f"{mm.get('MOTA', float('nan')):.2f} | {mm.get('IDF1', float('nan')):.2f} |"
        )
    lines.append("")
    lines.append("## 7. 结论\n")
    bp = {k: best[k] for k in ("track_high_thresh", "new_track_thresh", "match_thresh", "track_buffer")}
    lines.append(
        f"推荐参数 `{bp}`(+ `track_low_thresh={TRACK_LOW_THRESH}`)，在 5Hz MOT 上 "
        f"HOTA={best['HOTA']:.2f}、IDF1={best['IDF1']:.2f}，且在 3Hz 代理视频上 "
        f"short_id_ratio={best['proxy_short_id_ratio']:.3f}(不碎)。可视化见 `viz/`。"
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ByteTrack 超参搜索 (MOT + 3Hz 代理)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-type", default=MODEL_TYPE, choices=["rtdetr", "yolo", "rfdetr", "rfdetr_unified"])
    ap.add_argument("--mot-root", default=MOT_ROOT)
    ap.add_argument("--proxy-dir", default=PROXY_DIR)
    ap.add_argument("--out-dir", default="runs/bytetrack_sweep")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-mot-frames", type=int, default=None, help="每序列最多检测帧数(调试用)")
    ap.add_argument("--max-proxy", type=int, default=None, help="最多用几个代理视频(调试用)")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-every", type=int, default=1, help="网格扫描时每 N 组打一行日志 (默认每组都打)")
    ap.add_argument(
        "--no-class-aware", dest="class_aware", action="store_false", help="禁用 ByteTrack 类别隔离匹配 (默认开启)"
    )
    ap.add_argument(
        "--no-class-voting",
        dest="class_voting",
        action="store_false",
        help="禁用 ClassVotingTracker 类别投票(含反向写回) (默认开启)",
    )
    ap.set_defaults(class_aware=True, class_voting=True)
    args = ap.parse_args()

    setup_logger(args.log_level)
    t_total = time.time()

    mot_root = Path(args.mot_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    logger.info("=" * 70)
    logger.info("ByteTrack 超参搜索")
    logger.info("=" * 70)
    logger.info(f"检测模型: {args.model} (type={args.model_type}, conf_floor={DET_CONF_FLOOR})")
    logger.info(f"MOT 根目录: {args.mot_root} (帧率 {MOT_FRAME_RATE}Hz)")
    logger.info(f"代理视频目录: {args.proxy_dir} (采样到 {PROXY_FPS}Hz)")
    logger.info(
        f"搜索网格: {len(HIGH_NEW_PAIRS)} (high,new) × "
        f"{len(MATCH_THRESH)} match × {len(TRACK_BUFFER)} buffer = {len(grid)} 组"
    )
    logger.info(f"  (high, new) ∈ {list(HIGH_NEW_PAIRS)}")
    logger.info(f"  match_thresh ∈ {list(MATCH_THRESH)}")
    logger.info(f"  track_buffer ∈ {list(TRACK_BUFFER)}")
    logger.info(f"门控: short_id_ratio≤{GATE_SHORT_ID}, num_ids≤{GATE_NUMID_MULT}×min")
    logger.info(
        f"Tracker 配置: class_aware={args.class_aware}, class_voting={args.class_voting} "
        f"(投票含反向写回到 STrack.class_id)"
    )
    logger.info(f"输出目录: {out_dir}")

    logger.info("=" * 70)
    logger.info("阶段 1/4 — 构建检测器并缓存检测")
    logger.info("=" * 70)
    t0 = time.time()
    detector = create_detector(args.model_type, args.model, conf_thres=DET_CONF_FLOOR, iou_thres=0.5)
    logger.info(f"检测器就绪 ({time.time() - t0:.1f}s)")
    mot_cache = cache_mot_detections(detector, mot_root, args.max_mot_frames)
    proxy_videos = select_proxy_videos(Path(args.proxy_dir))
    if args.max_proxy:
        proxy_videos = proxy_videos[: args.max_proxy]
        logger.info(f"代理视频限制为前 {args.max_proxy} 个")
    proxy_cache = cache_proxy_detections(detector, proxy_videos)
    logger.info(f"阶段 1 完成, 累计 {time.time() - t_total:.1f}s")

    logger.info("=" * 70)
    logger.info(f"阶段 2/4 — 扫参数网格 ({len(grid)} 组)")
    logger.info("=" * 70)
    evaluator = MOTEvaluator(mot_root)
    rows: list[dict] = []
    t_sweep = time.time()
    for i, p in enumerate(grid, 1):
        t_combo = time.time()
        mot_preds = track_mot(mot_cache, p, class_aware=args.class_aware, class_voting=args.class_voting)
        res = evaluator.evaluate(mot_preds, iou_threshold=0.5, metrics=("clear", "identity", "hota"), classes=None)
        row = dict(p)
        row["HOTA"] = float(res.overall.get("HOTA", float("nan")))
        row["MOTA"] = float(res.overall.get("MOTA", float("nan")))
        row["IDF1"] = float(res.overall.get("IDF1", float("nan")))
        row["per_sequence"] = {s: dict(m) for s, m in res.per_sequence.items()}
        row.update(proxy_metrics(proxy_cache, p, class_aware=args.class_aware, class_voting=args.class_voting))
        rows.append(row)
        combo_s = time.time() - t_combo
        sweep_s = time.time() - t_sweep
        eta = sweep_s / i * (len(grid) - i)
        if i % max(1, args.log_every) == 0 or i == len(grid):
            logger.info(
                f"[{i:>3}/{len(grid)}] h{p['track_high_thresh']} n{p['new_track_thresh']} "
                f"m{p['match_thresh']:<4} b{p['track_buffer']:>2} -> "
                f"HOTA={row['HOTA']:5.2f} MOTA={row['MOTA']:6.2f} IDF1={row['IDF1']:5.2f} "
                f"short_id={row['proxy_short_id_ratio']:.3f} num_ids={row['proxy_num_ids']:5.1f} "
                f"({combo_s:.1f}s, ETA {eta / 60:.1f}min)"
            )
    logger.info(f"阶段 2 完成 ({time.time() - t_sweep:.1f}s), 累计 {time.time() - t_total:.1f}s")

    logger.info("=" * 70)
    logger.info("阶段 3/4 — 门控 + 排名")
    logger.info("=" * 70)
    ranked, gate = rank_with_gate(rows, args.top_k)
    top5 = ranked[: args.top_k]
    n_pass = sum(1 for r in ranked if r["pass_gate"])
    if gate != GATE_SHORT_ID:
        logger.warning(f"门限自动放宽: short_id_ratio gate {GATE_SHORT_ID} -> {gate} 才凑齐 top{args.top_k}")
    logger.info(f"过门组数: {n_pass}/{len(ranked)} (门限 short_id≤{gate})")
    ctx = {"n_seq": len(mot_cache), "n_proxy": len(proxy_cache), "n_grid": len(grid)}

    result = {
        "config": {
            "model": args.model,
            "model_type": args.model_type,
            "keep_classes": list(KEEP_CLASSES),
            "det_conf_floor": DET_CONF_FLOOR,
            "mot_frame_rate": MOT_FRAME_RATE,
            "proxy_fps": PROXY_FPS,
            "gate_short_id": gate,
            "gate_numid_mult": GATE_NUMID_MULT,
            "class_aware": args.class_aware,
            "class_voting": args.class_voting,
            **ctx,
        },
        "top5": [{k: v for k, v in r.items() if k != "per_sequence"} for r in top5],
        "all_ranked": [{k: v for k, v in r.items() if k != "per_sequence"} for r in ranked],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, top5, gate, ctx)
    logger.info(f"已写 {out_dir / 'result.json'} 和 {out_dir / 'REPORT.md'}")
    logger.info("Top5:")
    for r in top5:
        logger.info(
            f"  rank{r['rank']}: h{r['track_high_thresh']} n{r['new_track_thresh']} "
            f"m{r['match_thresh']} b{r['track_buffer']} | HOTA={r['HOTA']:.2f} "
            f"MOTA={r['MOTA']:.2f} IDF1={r['IDF1']:.2f} "
            f"short_id={r['proxy_short_id_ratio']:.3f} "
            f"{'PASS' if r['pass_gate'] else 'relax'}"
        )

    if args.skip_viz:
        logger.info("阶段 4/4 — 跳过可视化")
        logger.info(f"全流程总耗时 {time.time() - t_total:.1f}s")
        return
    logger.info("=" * 70)
    logger.info(f"阶段 4/4 — 渲染 top{args.top_k} × {len(mot_cache)} 序列可视化 = {args.top_k * len(mot_cache)} 个 mp4")
    logger.info("=" * 70)
    viz_dir = out_dir / "viz"
    t_viz = time.time()
    total_viz = len(top5) * len(mot_cache)
    done = 0
    for r in top5:
        for seq in mot_cache:
            done += 1
            out_path = viz_dir / seq / f"rank{r['rank']}_{param_tag(r)}.mp4"
            tv = time.time()
            render_seq(
                mot_root,
                mot_cache,
                seq,
                r,
                out_path,
                class_aware=args.class_aware,
                class_voting=args.class_voting,
            )
            logger.info(f"  [{done}/{total_viz}] {out_path} ({time.time() - tv:.1f}s)")
    logger.info(f"可视化完成 ({time.time() - t_viz:.1f}s), {total_viz} 个 mp4 于 {viz_dir}")
    logger.info(f"全流程总耗时 {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
