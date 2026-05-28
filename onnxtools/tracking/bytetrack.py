"""Native ByteTrack implementation.

Mirrors the official ``byte_tracker.py`` behaviour from the ByteTrack paper
(MOT17 SOTA config) but built on top of vectorised numpy primitives and the
shared :mod:`onnxtools.tracking.kalman` / :mod:`.matching` modules. Drop-in
replacement for :class:`SupervisionByteTrack` via the package factory.

The three association stages, the lost-buffer expiry policy, and the
unconfirmed-track confirmation rule are intentionally identical to the
reference implementation so behaviour is comparable across MOT benchmarks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from . import BaseTracker
from .base import TrackState
from .kalman import KalmanFilterXYAH
from .matching import fuse_score, iou_distance, linear_assignment


def _xyxy_to_xyah(boxes: np.ndarray) -> np.ndarray:
    """``[N, 4]`` xyxy -> ``[N, 4]`` (cx, cy, aspect, height)."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.astype(np.float32, copy=False)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + w * 0.5
    cy = boxes[:, 1] + h * 0.5
    a = np.where(h > 0, w / np.maximum(h, 1e-6), 0.0)
    return np.stack([cx, cy, a, h], axis=1).astype(np.float32)


def _xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[0], xyah[1], xyah[2], xyah[3]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class STrack:
    """Single-track state used internally by :class:`ByteTrackNative`."""

    shared_kalman: KalmanFilterXYAH = KalmanFilterXYAH()
    _count: int = 0

    __slots__ = (
        "xyah",
        "score",
        "class_id",
        "kalman_filter",
        "mean",
        "covariance",
        "track_id",
        "state",
        "is_activated",
        "tracklet_len",
        "frame_id",
        "start_frame",
        "last_xyxy",
    )

    def __init__(self, xyah: np.ndarray, score: float, class_id: int):
        self.xyah = xyah.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)

        self.kalman_filter: KalmanFilterXYAH | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        self.last_xyxy = _xyah_to_xyxy(self.xyah)

    # ---- ID issuance -----------------------------------------------------

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls) -> None:
        cls._count = 0

    # ---- batch predict ---------------------------------------------------

    @staticmethod
    def multi_predict(tracks: list[STrack]) -> None:
        if not tracks:
            return
        means = np.stack([t.mean for t in tracks])
        covs = np.stack([t.covariance for t in tracks])
        # If a track is Lost, zero its h-velocity component so the box does
        # not drift uncontrollably (matches reference behaviour).
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                means[i, 7] = 0.0
        new_means, new_covs = STrack.shared_kalman.multi_predict(means, covs)
        for i, t in enumerate(tracks):
            t.mean = new_means[i]
            t.covariance = new_covs[i]

    # ---- state transitions ----------------------------------------------

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = STrack.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.xyah)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1  # first frame can activate immediately
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        assert self.kalman_filter is not None and self.mean is not None
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xyah)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = STrack.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_xyxy = new_track.last_xyxy

    def update(self, new_track: STrack, frame_id: int) -> None:
        assert self.kalman_filter is not None and self.mean is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xyah)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_xyxy = new_track.last_xyxy

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @property
    def tlbr(self) -> np.ndarray:
        """xyxy from current state estimate."""
        if self.mean is None:
            return self.last_xyxy
        return _xyah_to_xyxy(self.mean[:4])


def _joint_tracks(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _sub_tracks(a: list[STrack], b: list[STrack]) -> list[STrack]:
    ids_b = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids_b]


def _remove_duplicate_tracks(a: list[STrack], b: list[STrack]) -> tuple[list[STrack], list[STrack]]:
    """Resolve duplicates between two pools by keeping the older track."""
    if not a or not b:
        return a, b
    pdist = iou_distance(np.stack([t.tlbr for t in a]), np.stack([t.tlbr for t in b]))
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = set(), set()
    for ai, bi in zip(*pairs):
        time_a = a[ai].frame_id - a[ai].start_frame
        time_b = b[bi].frame_id - b[bi].start_frame
        if time_a > time_b:
            dup_b.add(int(bi))
        else:
            dup_a.add(int(ai))
    new_a = [t for i, t in enumerate(a) if i not in dup_a]
    new_b = [t for i, t in enumerate(b) if i not in dup_b]
    return new_a, new_b


class ByteTrackNative(BaseTracker):
    """Hand-rolled ByteTrack — vectorised numpy implementation.

    Parameters mirror the official byte_tracker.py options. Also accepts the
    supervision-style alias names so the package factory can forward a single
    shared kwargs dict.
    """

    name = "bytetrack_native"

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        class_aware: bool = False,
        # supervision-style aliases for factory kwargs forwarding
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
        **_: Any,
    ):
        if track_activation_threshold is not None:
            track_high_thresh = float(track_activation_threshold)
            # supervision uses a single threshold; derive new_track_thresh
            new_track_thresh = max(new_track_thresh, track_high_thresh + 0.1)
        if lost_track_buffer is not None:
            track_buffer = int(lost_track_buffer)
        if minimum_matching_threshold is not None:
            match_thresh = float(minimum_matching_threshold)

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.class_aware = class_aware

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYAH()
        # Fresh tracker instance -> IDs restart at 1 (matches supervision).
        STrack.reset_id()

    # ---- public API ------------------------------------------------------

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_id()

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,  # frame unused
    ) -> sv.Detections:
        del frame  # frame data not consumed by motion-only ByteTrack
        self.frame_id += 1

        activated: list[STrack] = []
        refind: list[STrack] = []
        lost: list[STrack] = []
        removed: list[STrack] = []

        xyxy = detections.xyxy.astype(np.float32) if detections.xyxy is not None else np.empty((0, 4), dtype=np.float32)
        scores = (
            detections.confidence
            if getattr(detections, "confidence", None) is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        class_ids = (
            detections.class_id if getattr(detections, "class_id", None) is not None else np.zeros(len(xyxy), dtype=int)
        )
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=int)

        # Split high / low detections
        remain_inds = scores >= self.track_high_thresh
        inds_low = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)

        dets_high_xyxy = xyxy[remain_inds]
        dets_high_scores = scores[remain_inds]
        dets_high_classes = class_ids[remain_inds]

        dets_low_xyxy = xyxy[inds_low]
        dets_low_scores = scores[inds_low]
        dets_low_classes = class_ids[inds_low]

        detections_high = self._build_stracks(dets_high_xyxy, dets_high_scores, dets_high_classes)
        detections_low = self._build_stracks(dets_low_xyxy, dets_low_scores, dets_low_classes)

        # Split tracked into unconfirmed and confirmed.
        unconfirmed: list[STrack] = []
        tracked_stracks: list[STrack] = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked_stracks.append(t)

        # ---- Stage 1: first association with high-score detections -------
        strack_pool = _joint_tracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = self._iou_dist(strack_pool, detections_high)
        if dists.size:
            dists = fuse_score(dists, dets_high_scores)
        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)

        for it, idet in matches:
            track = strack_pool[int(it)]
            det = detections_high[int(idet)]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # ---- Stage 2: low-score detections vs unmatched Tracked tracks ---
        r_tracked = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == TrackState.Tracked]
        dists2 = self._iou_dist(r_tracked, detections_low)
        matches2, u_track2, _ = linear_assignment(dists2, thresh=0.5)

        for it, idet in matches2:
            track = r_tracked[int(it)]
            det = detections_low[int(idet)]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        for it in u_track2:
            track = r_tracked[int(it)]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # ---- Stage 3: unconfirmed vs remaining high-score detections -----
        detections_remaining = [detections_high[int(i)] for i in u_det]
        dists3 = self._iou_dist(unconfirmed, detections_remaining)
        if dists3.size:
            dists3 = fuse_score(dists3, np.array([d.score for d in detections_remaining], dtype=np.float32))
        matches3, u_unc, u_det3 = linear_assignment(dists3, thresh=0.7)

        for it, idet in matches3:
            unconfirmed[int(it)].update(detections_remaining[int(idet)], self.frame_id)
            activated.append(unconfirmed[int(it)])

        for it in u_unc:
            track = unconfirmed[int(it)]
            track.mark_removed()
            removed.append(track)

        # ---- New tracks from leftover high-score detections --------------
        for inew in u_det3:
            track = detections_remaining[int(inew)]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # ---- Update lost-buffer expiry -----------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # ---- Maintain pools ----------------------------------------------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint_tracks(self.tracked_stracks, activated)
        self.tracked_stracks = _joint_tracks(self.tracked_stracks, refind)
        self.lost_stracks = _sub_tracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub_tracks(self.lost_stracks, removed)
        self.removed_stracks.extend(removed)

        self.tracked_stracks, self.lost_stracks = _remove_duplicate_tracks(self.tracked_stracks, self.lost_stracks)

        # ---- Emit activated tracks ---------------------------------------
        output_tracks = [t for t in self.tracked_stracks if t.is_activated]
        if not output_tracks:
            empty = sv.Detections.empty()
            empty.tracker_id = np.empty((0,), dtype=int)
            return empty

        out_xyxy = np.stack([t.last_xyxy for t in output_tracks]).astype(np.float32)
        out_scores = np.array([t.score for t in output_tracks], dtype=np.float32)
        out_classes = np.array([t.class_id for t in output_tracks], dtype=int)
        out_ids = np.array([t.track_id for t in output_tracks], dtype=int)

        out = sv.Detections(xyxy=out_xyxy, confidence=out_scores, class_id=out_classes)
        out.tracker_id = out_ids
        return out

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _build_stracks(xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[STrack]:
        if xyxy.size == 0:
            return []
        xyah = _xyxy_to_xyah(xyxy)
        out: list[STrack] = []
        for i in range(len(xyxy)):
            t = STrack(xyah[i], float(scores[i]), int(classes[i]))
            t.last_xyxy = xyxy[i].astype(np.float32)
            out.append(t)
        return out

    def _iou_dist(self, tracks: list[STrack], dets: list[STrack]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        a = np.stack([t.tlbr for t in tracks])
        b = np.stack([d.last_xyxy for d in dets])
        cost = iou_distance(a, b)
        if self.class_aware:
            ca = np.array([t.class_id for t in tracks])
            cb = np.array([d.class_id for d in dets])
            mask = ca[:, None] != cb[None, :]
            # Use a value strictly above any plausible thresh.
            cost = np.where(mask, 1e6, cost)
        return cost


__all__ = ["ByteTrackNative", "STrack"]
