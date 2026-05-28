"""Association primitives for 2D tracking — IoU/GIoU, fuse_score, and a
``lap``/``scipy`` linear-assignment dispatcher.

Hot paths are pure numpy broadcasting; the assignment solver prefers
``lap.lapjv`` (with ``cost_limit`` for sparsity) and falls back to
``scipy.optimize.linear_sum_assignment`` when ``lap`` is not installed
(common on edge / Jetson environments).
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised in both branches via CI matrix
    import lap  # type: ignore

    _HAS_LAP = True
except ImportError:
    _HAS_LAP = False

from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Box geometry (vectorised)
# ---------------------------------------------------------------------------


def box_iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy box sets.

    Args:
        a: ``[N, 4]`` xyxy.
        b: ``[M, 4]`` xyxy.

    Returns:
        ``[N, M]`` IoU matrix in [0, 1].
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    # Intersect.
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def box_giou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise GIoU between two xyxy box sets. Returns ``[N, M]`` in [-1, 1]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Enclosing box.
    etl = np.minimum(a[:, None, :2], b[None, :, :2])
    ebr = np.maximum(a[:, None, 2:], b[None, :, 2:])
    ewh = (ebr - etl).clip(min=0)
    enc = ewh[..., 0] * ewh[..., 1]
    giou = iou - np.where(enc > 0, (enc - union) / enc, 0.0)
    return giou.astype(np.float32)


def iou_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``1 - IoU`` cost matrix."""
    return (1.0 - box_iou_batch(a, b)).astype(np.float32)


# ---------------------------------------------------------------------------
# Score fusion (ByteTrack first-stage)
# ---------------------------------------------------------------------------


def fuse_score(cost: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """ByteTrack score fusion: ``cost = 1 - (1 - cost) * det_score``.

    Lower cost when a high-confidence detection overlaps an existing track.
    """
    if cost.size == 0:
        return cost
    iou_sim = 1.0 - cost
    fuse_sim = iou_sim * det_scores[None, :]
    return (1.0 - fuse_sim).astype(np.float32)


# ---------------------------------------------------------------------------
# Linear assignment dispatch
# ---------------------------------------------------------------------------


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a rectangular linear assignment problem with a cost cutoff.

    Args:
        cost: ``[N, M]`` cost matrix (lower is better).
        thresh: Costs strictly greater than ``thresh`` are forbidden.

    Returns:
        Tuple ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is a
        ``[K, 2]`` int array of ``(row, col)`` index pairs.
    """
    if cost.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(cost.shape[0], dtype=np.int64),
            np.arange(cost.shape[1], dtype=np.int64),
        )

    n, m = cost.shape

    if _HAS_LAP:
        # lap returns y[i] = col matched to row i (or -1)
        _cost = cost.astype(np.float64, copy=False)
        _, x, y = lap.lapjv(_cost, extend_cost=True, cost_limit=float(thresh))
        matched_rows = np.where(x >= 0)[0]
        if matched_rows.size == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.arange(n, dtype=np.int64),
                np.arange(m, dtype=np.int64),
            )
        matches = np.stack([matched_rows, x[matched_rows]], axis=1).astype(np.int64)
        unmatched_a = np.where(x < 0)[0].astype(np.int64)
        unmatched_b = np.where(y < 0)[0].astype(np.int64)
        return matches, unmatched_a, unmatched_b

    # scipy fallback — cannot take inf; replace forbidden cells then filter.
    big = float(thresh) + 1e3
    cost_padded = np.where(cost > thresh, big, cost).astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_padded)

    matches = []
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    matches_arr = np.asarray(matches, dtype=np.int64).reshape(-1, 2) if matches else np.empty((0, 2), dtype=np.int64)
    unmatched_a = np.array([i for i in range(n) if i not in matched_rows], dtype=np.int64)
    unmatched_b = np.array([j for j in range(m) if j not in matched_cols], dtype=np.int64)
    return matches_arr, unmatched_a, unmatched_b


__all__ = [
    "box_iou_batch",
    "box_giou_batch",
    "iou_distance",
    "fuse_score",
    "linear_assignment",
]
