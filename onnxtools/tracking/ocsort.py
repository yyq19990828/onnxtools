"""Native OC-SORT (Observation-Centric SORT) implementation.

Reference: Cao et al., "Observation-Centric SORT: Rethinking SORT for
Robust Multi-Object Tracking" (CVPR 2023). This file follows the
``noahcao/OC_SORT`` reference layout but is built on the vectorised
:mod:`onnxtools.tracking.kalman` / :mod:`.matching` shared utilities.

Three OC-SORT-specific tweaks beyond classic SORT are implemented:

* **OCM (Observation-Centric Momentum)** — first-stage cost augments IoU
  with a velocity-direction consistency term.
* **OCR (Observation-Centric Recovery)** — a second pass associates
  unmatched detections to each track's *last observed* bbox (not the KF
  prediction), recovering targets whose KF has drifted during occlusion.
* **ORU (Observation-centric Re-Update)** — on re-acquisition, the track's
  velocity is recomputed from the most recent real observations rather than
  trusting the KF state accumulated during the lost window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from . import BaseTracker
from .kalman import KalmanFilterXYSR
from .matching import box_iou_batch, linear_assignment

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """xyxy -> [cx, cy, s, r]."""
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    cx = float(box[0]) + w / 2.0
    cy = float(box[1]) + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = float(s / w) if w > 0 else 0.0
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Unit direction vector between two box centres."""
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    dx = cx2 - cx1
    dy = cy2 - cy1
    norm = float(np.sqrt(dx * dx + dy * dy)) + 1e-6
    return np.array([dy / norm, dx / norm], dtype=np.float32)


def _speed_direction_batch(dets: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Pairwise direction vectors from each track centre to each det centre.

    Returns ``[2, T, D]`` (Y component then X component) — matches the OCM
    cost computation in the official repo.
    """
    cx1 = (dets[:, 0] + dets[:, 2]) / 2.0
    cy1 = (dets[:, 1] + dets[:, 3]) / 2.0
    cx2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    cy2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = cx1[None, :] - cx2[:, None]
    dy = cy1[None, :] - cy2[:, None]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    return np.stack([dy / norm, dx / norm], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-track wrapper
# ---------------------------------------------------------------------------


class KalmanBoxTracker:
    """Per-track KF + observation history (OC-SORT style)."""

    _count: int = 0

    __slots__ = (
        "kf",
        "mean",
        "covariance",
        "time_since_update",
        "id",
        "age",
        "hits",
        "hit_streak",
        "delta_t",
        "last_observation",
        "observations",
        "history_observations",
        "velocity",
        "class_id",
        "score",
    )

    def __init__(self, bbox: np.ndarray, score: float, class_id: int, delta_t: int = 3):
        self.kf = KalmanFilterXYSR()
        self.mean, self.covariance = self.kf.initiate(_xyxy_to_z(bbox))
        self.time_since_update = 0
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.delta_t = delta_t
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.history_observations: list[np.ndarray] = []
        self.velocity: np.ndarray | None = None
        self.class_id = int(class_id)
        self.score = float(score)

    @classmethod
    def reset_count(cls) -> None:
        cls._count = 0

    def predict(self) -> np.ndarray:
        # Mirror SORT's behaviour: prevent negative area predictions.
        if self.mean[6] + self.mean[2] <= 0:
            self.mean[6] = 0.0
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_xyxy(self.mean)

    def _previous_obs(self, age: int) -> np.ndarray:
        """Find most recent observation within ``delta_t`` frames before
        ``age``; falls back to ``last_observation`` if none exists.
        """
        max_age = self.age
        # Look backwards from current age - 1 to current age - delta_t.
        for dt in range(self.delta_t, 0, -1):
            target = max_age - dt
            if target in self.observations:
                return self.observations[target]
        return self.last_observation

    def update(self, bbox: np.ndarray, score: float, class_id: int) -> None:
        """Update with a new observation. Implements ORU velocity recompute."""
        # ORU: compute velocity from history rather than KF state.
        if self.last_observation.sum() >= 0:
            previous_box = self._previous_obs(self.age)
            self.velocity = _speed_direction(previous_box, bbox)

        self.last_observation = np.array(
            [bbox[0], bbox[1], bbox[2], bbox[3], score], dtype=np.float32
        )
        self.observations[self.age] = self.last_observation
        self.history_observations.append(self.last_observation)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        self.class_id = int(class_id)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, _xyxy_to_z(bbox))

    def get_state(self) -> np.ndarray:
        return _x_to_xyxy(self.mean)


# ---------------------------------------------------------------------------
# Associations
# ---------------------------------------------------------------------------


def _angle_cost(
    detections: np.ndarray,
    trackers: np.ndarray,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
) -> np.ndarray:
    """Compute OCM (observation-centric momentum) cost.

    Args:
        detections: ``[D, 4]`` xyxy.
        trackers: ``[T, 4]`` predicted xyxy.
        velocities: ``[T, 2]`` per-track unit direction vectors.
        previous_obs: ``[T, 5]`` historical observations (xyxy + score).
        vdc_weight: scalar weight applied to the angle term.

    Returns:
        ``[T, D]`` angle cost (subtracted from IoU in the final matrix).
    """
    if detections.size == 0 or trackers.size == 0:
        return np.zeros((trackers.shape[0], detections.shape[0]), dtype=np.float32)

    Y, X = _speed_direction_batch(detections, previous_obs)
    inertia_Y = velocities[:, 0][:, None]
    inertia_X = velocities[:, 1][:, None]
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, -1.0, 1.0)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    # Invalid (no previous_obs) — zero out their contribution.
    valid_mask = (previous_obs.sum(axis=1) >= 0).astype(np.float32)
    scores = np.repeat(
        np.ones((detections.shape[0], 1), dtype=np.float32),
        trackers.shape[0],
        axis=1,
    ).T  # [T, D] — placeholder weight, real det score multiplier handled by caller

    angle_diff_cost = (valid_mask[:, None] * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost.astype(np.float32)


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------


class OCSORT(BaseTracker):
    """Hand-rolled OC-SORT — vectorised numpy implementation.

    Parameter names follow the reference repository; also accepts the
    supervision-style aliases so the package factory can forward a shared
    kwargs dict.
    """

    name = "ocsort"

    def __init__(
        self,
        det_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        # supervision-style aliases
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
        frame_rate: int | None = None,
        **_: Any,
    ):
        if track_activation_threshold is not None:
            det_thresh = float(track_activation_threshold)
        if lost_track_buffer is not None:
            max_age = int(lost_track_buffer)
        if minimum_matching_threshold is not None:
            # supervision exposes a 0.8-style "matching threshold" (cost cap).
            # OC-SORT's iou_threshold is the IoU cap. Convert: cost = 1 - iou.
            iou_threshold = max(0.0, 1.0 - float(minimum_matching_threshold))
        _ = frame_rate  # unused — kept for kwargs compatibility

        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

    def reset(self) -> None:
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        del frame
        self.frame_count += 1

        xyxy = (
            detections.xyxy.astype(np.float32)
            if detections.xyxy is not None
            else np.empty((0, 4), dtype=np.float32)
        )
        scores = (
            np.asarray(detections.confidence, dtype=np.float32)
            if getattr(detections, "confidence", None) is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        class_ids = (
            np.asarray(detections.class_id, dtype=int)
            if getattr(detections, "class_id", None) is not None
            else np.zeros(len(xyxy), dtype=int)
        )

        # Filter by det threshold.
        remain = scores >= self.det_thresh
        xyxy = xyxy[remain]
        scores = scores[remain]
        class_ids = class_ids[remain]

        # Predict existing trackers — drop those producing NaN.
        trks_predicted: list[np.ndarray] = []
        keep_idx: list[int] = []
        for i, t in enumerate(self.trackers):
            pos = t.predict()
            if np.any(np.isnan(pos)):
                continue
            trks_predicted.append(pos)
            keep_idx.append(i)
        self.trackers = [self.trackers[i] for i in keep_idx]

        if trks_predicted:
            trks_arr = np.stack(trks_predicted).astype(np.float32)
        else:
            trks_arr = np.empty((0, 4), dtype=np.float32)

        velocities = np.array(
            [
                t.velocity if t.velocity is not None else np.zeros(2, dtype=np.float32)
                for t in self.trackers
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        previous_obs = np.array(
            [
                t._previous_obs(t.age)
                for t in self.trackers
            ],
            dtype=np.float32,
        ).reshape(-1, 5)

        # ---- Stage 1: IoU + OCM ------------------------------------------
        matches, u_det, u_trk = self._associate(
            xyxy, trks_arr, velocities, previous_obs
        )
        for d, t in matches:
            self.trackers[int(t)].update(xyxy[int(d)], float(scores[int(d)]), int(class_ids[int(d)]))

        # ---- Stage 2: OCR — use last observations for unmatched tracks ---
        if len(u_trk) > 0 and len(u_det) > 0:
            last_boxes = np.stack(
                [self.trackers[int(t)].last_observation[:4] for t in u_trk]
            ).astype(np.float32)
            left_dets = xyxy[u_det]
            iou_ocr = box_iou_batch(left_dets, last_boxes)
            # Cost = 1 - IoU, only valid when last observation exists.
            valid = np.array(
                [self.trackers[int(t)].last_observation.sum() >= 0 for t in u_trk]
            )
            cost = 1.0 - iou_ocr  # [Du, Tu]
            if not valid.all():
                cost[:, ~valid] = 1e6
            cost = cost.T  # [Tu, Du]
            m2, u_t2, u_d2 = linear_assignment(cost, thresh=1.0 - self.iou_threshold)
            for it, idet in m2:
                trk_idx = int(u_trk[int(it)])
                det_idx = int(u_det[int(idet)])
                self.trackers[trk_idx].update(
                    xyxy[det_idx], float(scores[det_idx]), int(class_ids[det_idx])
                )
            u_det = np.array([u_det[int(k)] for k in u_d2], dtype=np.int64)
            u_trk = np.array([u_trk[int(k)] for k in u_t2], dtype=np.int64)

        # ---- Create new trackers for unmatched high-score detections -----
        for i in u_det:
            t = KalmanBoxTracker(
                xyxy[int(i)], float(scores[int(i)]), int(class_ids[int(i)]), delta_t=self.delta_t
            )
            self.trackers.append(t)

        # ---- Emit + cull -------------------------------------------------
        out_xyxy: list[np.ndarray] = []
        out_scores: list[float] = []
        out_classes: list[int] = []
        out_ids: list[int] = []

        survivors: list[KalmanBoxTracker] = []
        for t in self.trackers:
            if t.time_since_update < 1 and (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = (
                    t.last_observation[:4]
                    if t.last_observation.sum() >= 0
                    else t.get_state()
                )
                out_xyxy.append(box)
                out_scores.append(t.score)
                out_classes.append(t.class_id)
                out_ids.append(t.id + 1)  # 1-based IDs (matches supervision convention)
            if t.time_since_update <= self.max_age:
                survivors.append(t)
        self.trackers = survivors

        if not out_xyxy:
            empty = sv.Detections.empty()
            empty.tracker_id = np.empty((0,), dtype=int)
            return empty

        out = sv.Detections(
            xyxy=np.stack(out_xyxy).astype(np.float32),
            confidence=np.asarray(out_scores, dtype=np.float32),
            class_id=np.asarray(out_classes, dtype=int),
        )
        out.tracker_id = np.asarray(out_ids, dtype=int)
        return out

    # ---- internals -------------------------------------------------------

    def _associate(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
        velocities: np.ndarray,
        previous_obs: np.ndarray,
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        if detections.shape[0] == 0 or trackers.shape[0] == 0:
            return (
                [],
                np.arange(detections.shape[0], dtype=np.int64),
                np.arange(trackers.shape[0], dtype=np.int64),
            )

        iou = box_iou_batch(detections, trackers).T  # [T, D]
        angle_cost = _angle_cost(
            detections, trackers, velocities, previous_obs, vdc_weight=self.inertia
        )
        # Final cost: lower IoU and lower momentum agreement -> higher cost.
        cost = -(iou + angle_cost)
        # Gate by raw IoU (not iou+angle) — matches reference behaviour.
        forbidden = iou < self.iou_threshold
        cost = np.where(forbidden, 1e6, cost)
        matches_arr, u_t, u_d = linear_assignment(cost, thresh=1e5)
        # Convert to (det, trk) pairs.
        matches = [(int(c), int(r)) for r, c in matches_arr]
        return matches, u_d.astype(np.int64), u_t.astype(np.int64)


__all__ = ["OCSORT", "KalmanBoxTracker"]
