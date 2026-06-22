"""MOTChallenge 格式多目标跟踪评估器。

公共入口 :class:`MOTEvaluator` 接收 ground truth 目录与预测（result 文件目录或内存
``dict``），逐帧做 IoU 匹配，并调用两套现成评估库出指标：

* **motmetrics** (`py-motmetrics`) —— CLEAR-MOT（MOTA / MOTP）与 Identity（IDF1/IDP/IDR）
  及 IDsw / FP / FN / Frag / MT / ML 等计数指标。
* **TrackEval** —— HOTA 及其子指标 DetA / AssA / LocA（当前 MOT 主流综合指标）。

两库均**惰性导入、缺失时优雅降级**：只装其中之一也能跑出对应的指标子集。安装：
``uv pip install -e ".[mot]"``。

便捷函数 :func:`run_tracker_on_gt` 把 GT 框当作"理想检测"喂给
:func:`onnxtools.tracking.create_tracker` 的任一后端，直接对比各跟踪器的**关联质量**
（不依赖检测器）。

Examples:
    >>> from onnxtools.eval import MOTEvaluator, run_tracker_on_gt
    >>> ev = MOTEvaluator("data/track/MOT_dataset")
    >>> preds = run_tracker_on_gt("data/track/MOT_dataset", "bytetrack_native")
    >>> result = ev.evaluate(preds)
    >>> print(result.summary_table())
    >>> result.overall["HOTA"]            # 标量
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mot_data import (
    COL_CLASS,
    COL_CONF,
    COL_H,
    COL_ID,
    COL_X,
    MOTSequence,
    load_gt,
    load_predictions,
)

logger = logging.getLogger(__name__)

# 表格中展示的指标顺序（百分比型 + 计数型）。
_SUMMARY_ORDER = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "Rcll",
    "Prcn",
    "IDsw",
    "FP",
    "FN",
    "Frag",
    "MT",
    "ML",
    "GT_IDs",
)
_COUNT_METRICS = frozenset({"IDsw", "FP", "FN", "Frag", "MT", "ML", "GT_IDs"})


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------


def iou_matrix_xywh(gt_xywh: np.ndarray, pred_xywh: np.ndarray) -> np.ndarray:
    """计算两组 ``xywh`` 框的 IoU 矩阵。

    Args:
        gt_xywh: 形状 ``(G, 4)``，列为左上角 ``x, y`` 与宽高 ``w, h``。
        pred_xywh: 形状 ``(P, 4)``，同上。

    Returns:
        形状 ``(G, P)`` 的 IoU 矩阵，取值 ``[0, 1]``。任一组为空时返回对应形状的空数组。
    """
    g, p = len(gt_xywh), len(pred_xywh)
    if g == 0 or p == 0:
        return np.zeros((g, p), dtype=float)

    gt = np.asarray(gt_xywh, dtype=float)
    pr = np.asarray(pred_xywh, dtype=float)
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    px1, py1 = pr[:, 0], pr[:, 1]
    px2, py2 = pr[:, 0] + pr[:, 2], pr[:, 1] + pr[:, 3]

    iw = np.maximum(0.0, np.minimum(gx2[:, None], px2[None]) - np.maximum(gx1[:, None], px1[None]))
    ih = np.maximum(0.0, np.minimum(gy2[:, None], py2[None]) - np.maximum(gy1[:, None], py1[None]))
    inter = iw * ih
    area_g = (gt[:, 2] * gt[:, 3])[:, None]
    area_p = (pr[:, 2] * pr[:, 3])[None]
    union = area_g + area_p - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class MOTResult:
    """评估结果：每序列指标 + 汇总（OVERALL）行。

    百分比型指标（HOTA/MOTA/IDF1/...）以 ``[0, 100]`` 的数值存放；计数型指标
    （IDsw/FP/FN/...）为整数。缺失的评估库对应的指标键不会出现。

    Attributes:
        per_sequence: ``{seq_name: {metric: value}}``。
        overall: 汇总行 ``{metric: value}``（HOTA 跨序列重组、CLEAR/Identity 取全局聚合）。
        metric_names: 实际计算出的指标名（保持展示顺序）。
    """

    per_sequence: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    metric_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """转为可 JSON 序列化的嵌套 ``dict``（含 ``OVERALL`` 行）。"""
        return {**self.per_sequence, "OVERALL": self.overall}

    def summary_table(self, float_fmt: str = "{:.2f}") -> str:
        """渲染为等宽对齐的文本表格（每行一个序列，末行 OVERALL）。"""
        cols = [m for m in _SUMMARY_ORDER if m in self.metric_names]
        header = ["sequence", *cols]
        rows: list[list[str]] = []
        for name in [*self.per_sequence, "OVERALL"]:
            src = self.overall if name == "OVERALL" else self.per_sequence[name]
            cells = [name]
            for m in cols:
                v = src.get(m)
                if v is None:
                    cells.append("-")
                elif m in _COUNT_METRICS:
                    cells.append(str(int(round(v))))
                else:
                    cells.append(float_fmt.format(v))
            rows.append(cells)

        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        sep = "  "

        def _fmt(cells: list[str]) -> str:
            return sep.join(c.rjust(widths[i]) for i, c in enumerate(cells))

        lines = [_fmt(header), _fmt(["-" * w for w in widths])]
        lines += [_fmt(r) for r in rows[:-1]]
        lines.append(_fmt(["-" * w for w in widths]))
        lines.append(_fmt(rows[-1]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _filter_frames(
    frames: dict[int, np.ndarray],
    classes: list[int] | None,
    conf_threshold: float,
) -> dict[int, np.ndarray]:
    """按类别集合与置信度阈值过滤每帧框数组。返回新字典（不修改入参）。"""
    if classes is None and conf_threshold <= 0:
        return frames
    cls_set = set(classes) if classes is not None else None
    out: dict[int, np.ndarray] = {}
    for f, arr in frames.items():
        keep = np.ones(len(arr), dtype=bool)
        if cls_set is not None:
            keep &= np.isin(arr[:, COL_CLASS].astype(int), list(cls_set))
        if conf_threshold > 0:
            keep &= arr[:, COL_CONF] >= conf_threshold
        sel = arr[keep]
        if len(sel):
            out[f] = sel
    return out


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------


class MOTEvaluator:
    """MOTChallenge 格式跟踪评估器。

    Args:
        gt_root: GT 数据集根目录（含 ``<SEQ>/gt/gt.txt`` 等，见
            :func:`onnxtools.eval.mot_data.load_gt`）。
        seqmap: 限定评估的序列名列表；``None`` 时读 ``seqmap.txt`` 或自动发现。

    Raises:
        FileNotFoundError: GT 目录无效或不含任何序列。
    """

    def __init__(self, gt_root: str | Path, seqmap: list[str] | None = None):
        self.gt_root = Path(gt_root)
        self.gt: dict[str, MOTSequence] = load_gt(gt_root, seqmap)
        self.seqmap: list[str] = list(self.gt)

    def evaluate(
        self,
        predictions: str | Path | dict[str, MOTSequence | dict[int, np.ndarray]],
        *,
        iou_threshold: float = 0.5,
        metrics: tuple[str, ...] = ("clear", "identity", "hota"),
        classes: list[int] | None = None,
        conf_threshold: float = 0.0,
    ) -> MOTResult:
        """评估预测相对 GT 的跟踪指标。

        Args:
            predictions: 预测来源——result 文件目录、``{seq: MOTSequence}`` 或
                ``{seq: {frame: ndarray}}``（见
                :func:`onnxtools.eval.mot_data.load_predictions`）。
            iou_threshold: CLEAR/Identity 匹配的最小 IoU（MOTChallenge 默认 0.5）。
                HOTA 不受此参数影响（其在内部对一组 alpha 阈值积分）。
            metrics: 要计算的指标族子集，取值 ``"clear"`` / ``"identity"`` / ``"hota"``。
                ``clear``、``identity`` 需 motmetrics；``hota`` 需 trackeval。
            classes: 仅评估这些类别 id（对 GT 与预测同时过滤）；``None`` 表示
                **池化所有类别**（标准单类 MOT 评估，默认）。
            conf_threshold: 过滤掉置信度低于该值的预测框；``0`` 不过滤。

        Returns:
            :class:`MOTResult`。

        Raises:
            ValueError: ``metrics`` 含未知项。
        """
        metrics = tuple(m.lower() for m in metrics)
        unknown = set(metrics) - {"clear", "identity", "hota"}
        if unknown:
            raise ValueError(f"未知指标族: {sorted(unknown)}，可选 clear/identity/hota")

        pred_seqs = load_predictions(predictions, seqmap=self.seqmap)

        # 逐序列预处理：过滤 + 帧时间线 + 每帧 IoU。
        per_seq_frames: dict[str, dict] = {}
        for name in self.seqmap:
            gt_seq = self.gt[name]
            pred_seq = pred_seqs.get(name)
            gt_frames = _filter_frames(gt_seq.frames, classes, 0.0)
            pred_frames = _filter_frames(pred_seq.frames, classes, conf_threshold) if pred_seq is not None else {}
            if pred_seq is None:
                logger.warning("序列 %s 无预测，按全漏检处理", name)
            n = gt_seq.info.seq_length if gt_seq.info else 0
            timeline = range(1, n + 1) if n > 0 else sorted(set(gt_frames) | set(pred_frames))
            per_seq_frames[name] = {
                "gt": gt_frames,
                "pred": pred_frames,
                "timeline": list(timeline),
            }

        result = MOTResult()
        want_mm = ("clear" in metrics) or ("identity" in metrics)
        if want_mm:
            self._compute_motmetrics(per_seq_frames, iou_threshold, metrics, result)
        if "hota" in metrics:
            self._compute_hota(per_seq_frames, result)

        # 固定展示顺序。
        result.metric_names = [m for m in _SUMMARY_ORDER if m in result.metric_names]
        return result

    # -- motmetrics (CLEAR + Identity) --------------------------------------

    def _compute_motmetrics(
        self,
        per_seq_frames: dict[str, dict],
        iou_threshold: float,
        metrics: tuple[str, ...],
        result: MOTResult,
    ) -> None:
        try:
            import motmetrics as mm
        except ImportError:
            logger.warning('未安装 motmetrics，跳过 CLEAR/Identity 指标。安装: uv pip install -e ".[mot]"')
            return

        accs: list = []
        names: list[str] = []
        for name in self.seqmap:
            data = per_seq_frames[name]
            gt_frames, pred_frames = data["gt"], data["pred"]
            acc = mm.MOTAccumulator(auto_id=True)
            for frame in data["timeline"]:
                g = gt_frames.get(frame)
                p = pred_frames.get(frame)
                gids = g[:, COL_ID].astype(int).tolist() if g is not None else []
                pids = p[:, COL_ID].astype(int).tolist() if p is not None else []
                if g is not None and p is not None:
                    iou = iou_matrix_xywh(g[:, COL_X : COL_H + 1], p[:, COL_X : COL_H + 1])
                    dist = 1.0 - iou
                    dist[iou < iou_threshold] = np.nan  # 门限：IoU 不足视为不可匹配
                else:
                    dist = np.empty((len(gids), len(pids)))
                acc.update(gids, pids, dist)
            accs.append(acc)
            names.append(name)

        mm_metrics = [
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_fragmentations",
            "mostly_tracked",
            "mostly_lost",
            "num_unique_objects",
            "precision",
            "recall",
        ]
        mh = mm.metrics.create()
        # motmetrics 用模块级 logging.info 打计时噪声（"partials: x seconds"）到 root
        # logger；评估期间临时抬高 root 级别屏蔽，结束后还原。
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(max(prev_level, logging.WARNING))
        try:
            summary = mh.compute_many(accs, metrics=mm_metrics, names=names, generate_overall=True)
        finally:
            root_logger.setLevel(prev_level)

        want_clear = "clear" in metrics
        want_identity = "identity" in metrics
        for row_name in summary.index:
            row = summary.loc[row_name]
            out: dict[str, float] = {}
            if want_clear:
                out["MOTA"] = float(row["mota"]) * 100.0
                out["MOTP"] = (1.0 - float(row["motp"])) * 100.0  # 转为平均 IoU 重叠%
                out["FP"] = float(row["num_false_positives"])
                out["FN"] = float(row["num_misses"])
                out["IDsw"] = float(row["num_switches"])
                out["Frag"] = float(row["num_fragmentations"])
                out["MT"] = float(row["mostly_tracked"])
                out["ML"] = float(row["mostly_lost"])
                out["Rcll"] = float(row["recall"]) * 100.0
                out["Prcn"] = float(row["precision"]) * 100.0
                out["GT_IDs"] = float(row["num_unique_objects"])
            if want_identity:
                out["IDF1"] = float(row["idf1"]) * 100.0
                out["IDP"] = float(row["idp"]) * 100.0
                out["IDR"] = float(row["idr"]) * 100.0

            target = result.overall if row_name == "OVERALL" else result.per_sequence.setdefault(row_name, {})
            target.update(out)
            for k in out:
                if k not in result.metric_names:
                    result.metric_names.append(k)

    # -- TrackEval (HOTA) ---------------------------------------------------

    def _compute_hota(self, per_seq_frames: dict[str, dict], result: MOTResult) -> None:
        try:
            from trackeval.metrics import HOTA
        except ImportError:
            logger.warning('未安装 trackeval，跳过 HOTA 指标。安装: uv pip install -e ".[mot]"')
            return

        hota = HOTA()
        per_seq_res: dict[str, dict] = {}
        for name in self.seqmap:
            data = per_seq_frames[name]
            te_data = self._build_trackeval_data(data["gt"], data["pred"], data["timeline"])
            per_seq_res[name] = hota.eval_sequence(te_data)

        combined = hota.combine_sequences(per_seq_res)

        def _summ(res: dict) -> dict[str, float]:
            return {
                "HOTA": float(np.mean(res["HOTA"])) * 100.0,
                "DetA": float(np.mean(res["DetA"])) * 100.0,
                "AssA": float(np.mean(res["AssA"])) * 100.0,
                "LocA": float(np.mean(res["LocA"])) * 100.0,
            }

        for name, res in per_seq_res.items():
            result.per_sequence.setdefault(name, {}).update(_summ(res))
        result.overall.update(_summ(combined))
        for k in ("HOTA", "DetA", "AssA", "LocA"):
            if k not in result.metric_names:
                result.metric_names.append(k)

    @staticmethod
    def _build_trackeval_data(
        gt_frames: dict[int, np.ndarray],
        pred_frames: dict[int, np.ndarray],
        timeline: list[int],
    ) -> dict:
        """构造 TrackEval ``eval_sequence`` 所需的 data 字典。

        TrackEval 要求 gt/tracker id 为**连续 0 基**索引，故先建立全序列重映射，再逐帧
        填充 id 数组与原始 IoU 相似度矩阵（HOTA 在内部对一组 alpha 阈值积分，故此处不做门限）。
        """
        gt_id_map: dict[int, int] = {}
        pr_id_map: dict[int, int] = {}
        for arr in gt_frames.values():
            for tid in arr[:, COL_ID].astype(int):
                gt_id_map.setdefault(int(tid), len(gt_id_map))
        for arr in pred_frames.values():
            for tid in arr[:, COL_ID].astype(int):
                pr_id_map.setdefault(int(tid), len(pr_id_map))

        gt_ids_per_t: list[np.ndarray] = []
        pr_ids_per_t: list[np.ndarray] = []
        sims_per_t: list[np.ndarray] = []
        n_gt_dets = n_pr_dets = 0
        for frame in timeline:
            g = gt_frames.get(frame)
            p = pred_frames.get(frame)
            gids = (
                np.array([gt_id_map[int(t)] for t in g[:, COL_ID].astype(int)], dtype=int)
                if g is not None
                else np.empty(0, dtype=int)
            )
            pids = (
                np.array([pr_id_map[int(t)] for t in p[:, COL_ID].astype(int)], dtype=int)
                if p is not None
                else np.empty(0, dtype=int)
            )
            if g is not None and p is not None:
                sims = iou_matrix_xywh(g[:, COL_X : COL_H + 1], p[:, COL_X : COL_H + 1])
            else:
                sims = np.zeros((len(gids), len(pids)), dtype=float)
            gt_ids_per_t.append(gids)
            pr_ids_per_t.append(pids)
            sims_per_t.append(sims)
            n_gt_dets += len(gids)
            n_pr_dets += len(pids)

        return {
            "num_gt_ids": len(gt_id_map),
            "num_tracker_ids": len(pr_id_map),
            "num_gt_dets": n_gt_dets,
            "num_tracker_dets": n_pr_dets,
            "gt_ids": gt_ids_per_t,
            "tracker_ids": pr_ids_per_t,
            "similarity_scores": sims_per_t,
        }


# ---------------------------------------------------------------------------
# 便捷：用 GT 框作为理想检测跑跟踪器
# ---------------------------------------------------------------------------


def run_tracker_on_gt(
    gt_root: str | Path,
    tracker_algo: str = "bytetrack",
    *,
    seqmap: list[str] | None = None,
    confidence: float = 1.0,
    **tracker_kwargs,
) -> dict[str, MOTSequence]:
    """把 GT 框当作"理想检测"逐帧喂给跟踪器，产出可直接评估的预测。

    这隔离掉检测误差，单独度量各跟踪后端的**关联质量**。每个序列用独立的 tracker
    实例（ID 从 1 重新发号）。

    Args:
        gt_root: GT 数据集根目录。
        tracker_algo: :func:`onnxtools.tracking.create_tracker` 支持的算法名。
        seqmap: 限定序列；``None`` 时同 :func:`onnxtools.eval.mot_data.load_gt`。
        confidence: 赋给每个 GT 框的检测置信度（默认 1.0，确保通过跟踪器高分阈值）。
        **tracker_kwargs: 透传给 ``create_tracker`` 的参数（如 ``track_buffer``、
            ``frame_rate``）。

    Returns:
        ``{seq_name: MOTSequence}``，可直接传给 :meth:`MOTEvaluator.evaluate`。
    """
    import supervision as sv

    from ..tracking import create_tracker

    gt = load_gt(gt_root, seqmap)
    preds: dict[str, MOTSequence] = {}
    for name, seq in gt.items():
        tracker = create_tracker(tracker_algo, **tracker_kwargs)
        frames_out: dict[int, np.ndarray] = {}
        n = seq.info.seq_length if seq.info else (max(seq.frames) if seq.frames else 0)
        for frame in range(1, n + 1):
            g = seq.get(frame)
            if len(g):
                xywh = g[:, COL_X : COL_H + 1]
                xyxy = np.column_stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]])
                dets = sv.Detections(
                    xyxy=xyxy.astype(float),
                    confidence=np.full(len(g), confidence, dtype=float),
                    class_id=g[:, COL_CLASS].astype(int),
                )
            else:
                dets = sv.Detections.empty()
            tracked = tracker.update(dets, None)
            if tracked.tracker_id is None or len(tracked) == 0:
                continue
            txyxy = tracked.xyxy
            tw = txyxy[:, 2] - txyxy[:, 0]
            th = txyxy[:, 3] - txyxy[:, 1]
            tcls = tracked.class_id if tracked.class_id is not None else np.full(len(tracked), -1)
            rows = np.column_stack(
                [
                    tracked.tracker_id.astype(float),
                    txyxy[:, 0],
                    txyxy[:, 1],
                    tw,
                    th,
                    np.full(len(tracked), confidence),
                    np.asarray(tcls, dtype=float),
                ]
            )
            frames_out[frame] = rows
        preds[name] = MOTSequence(name=name, frames=frames_out)
    return preds
