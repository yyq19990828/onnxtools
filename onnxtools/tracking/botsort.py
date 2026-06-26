"""Native BoT-SORT implementation.

BoT-SORT keeps ByteTrack's high/low-score association policy and adds three
practical extensions:

* XYWH Kalman state, tracking width and height directly.
* Optional camera motion compensation (CMC) through a lazily imported OpenCV
  ORB/RANSAC global affine estimator.
* Optional ReID association through caller-provided embeddings. Embeddings can
  be passed in ``detections.data["features"]`` or produced by a
  ``reid_encoder(frame, xyxy)`` callable.

No PyTorch/FastReID dependency is pulled into the tracker core. Production
deployments can plug in any external ReID model that emits L2-normalisable
feature vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import supervision as sv

from . import BaseTracker
from .base import TrackState
from .matching import fuse_score, iou_distance, linear_assignment

ReIDEncoder = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """``[N, 4]`` xyxy -> ``[N, 4]`` (cx, cy, width, height)."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.astype(np.float32, copy=False)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + w * 0.5
    cy = boxes[:, 1] + h * 0.5
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    cx, cy = float(xywh[0]), float(xywh[1])
    w = max(float(xywh[2]), 0.0)
    h = max(float(xywh[3]), 0.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _normalize_features(features: np.ndarray | None) -> np.ndarray | None:
    if features is None:
        return None
    out = np.asarray(features, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, np.maximum(norms, 1e-12)).astype(np.float32)


def _warp_xyxy(boxes: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to xyxy boxes and return enclosing xyxy."""
    if boxes.size == 0:
        return boxes.astype(np.float32, copy=False)
    m = np.asarray(affine, dtype=np.float32)
    if m.shape != (2, 3):
        raise ValueError(f"affine must have shape (2, 3), got {m.shape}")

    tl = boxes[:, [0, 1]]
    tr = boxes[:, [2, 1]]
    br = boxes[:, [2, 3]]
    bl = boxes[:, [0, 3]]
    corners = np.stack([tl, tr, br, bl], axis=1)
    ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
    hom = np.concatenate([corners.astype(np.float32), ones], axis=2)
    warped = hom @ m.T
    xy_min = warped.min(axis=1)
    xy_max = warped.max(axis=1)
    return np.concatenate([xy_min, xy_max], axis=1).astype(np.float32)


class KalmanFilterXYWH:
    """Kalman filter using BoT-SORT's ``[cx, cy, w, h]`` measurement state."""

    ndim = 4

    def __init__(self) -> None:
        dt = 1.0
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def _scale(self, xywh: np.ndarray) -> tuple[float, float]:
        w = max(float(abs(xywh[2])), 1.0)
        h = max(float(abs(xywh[3])), 1.0)
        return w, h

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialise state from one ``[cx, cy, w, h]`` measurement."""
        m = np.asarray(measurement, dtype=np.float32)
        mean = np.concatenate([m, np.zeros_like(m)])
        w, h = self._scale(m)
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * w,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * w,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        return mean.astype(np.float32), np.diag(std * std).astype(np.float32)

    def _motion_cov(self, xywh: np.ndarray) -> np.ndarray:
        w, h = self._scale(xywh)
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                self._std_weight_position * w,
                self._std_weight_position * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * w,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        return np.diag(std * std).astype(np.float32)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run one constant-velocity prediction step."""
        motion_cov = self._motion_cov(mean[:4])
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised prediction over ``N`` tracks."""
        if means.shape[0] == 0:
            return means, covariances

        f = self._motion_mat
        new_means = means @ f.T

        ws = np.maximum(np.abs(means[:, 2]), 1.0)
        hs = np.maximum(np.abs(means[:, 3]), 1.0)
        n = means.shape[0]
        motion_cov = np.zeros((n, 8, 8), dtype=np.float32)
        diag = np.empty((n, 8), dtype=np.float32)
        diag[:, 0] = self._std_weight_position * hs
        diag[:, 1] = self._std_weight_position * hs
        diag[:, 2] = self._std_weight_position * ws
        diag[:, 3] = self._std_weight_position * hs
        diag[:, 4] = self._std_weight_velocity * hs
        diag[:, 5] = self._std_weight_velocity * hs
        diag[:, 6] = self._std_weight_velocity * ws
        diag[:, 7] = self._std_weight_velocity * hs
        diag *= diag
        idx = np.arange(8)
        motion_cov[:, idx, idx] = diag

        new_covs = np.einsum("ij,njk,lk->nil", f, covariances, f) + motion_cov
        return new_means.astype(np.float32), new_covs.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        w, h = self._scale(mean[:4])
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                self._std_weight_position * w,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(std * std).astype(np.float32)
        h_mat = self._update_mat
        mean_proj = h_mat @ mean
        cov_proj = h_mat @ covariance @ h_mat.T + innovation_cov
        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct state with one ``[cx, cy, w, h]`` measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)
        h_mat = self._update_mat
        cov_ht = covariance @ h_mat.T
        try:
            chol = np.linalg.cholesky(projected_cov)
            tmp = np.linalg.solve(chol, cov_ht.T)
            kalman_gain = np.linalg.solve(chol.T, tmp).T
        except np.linalg.LinAlgError:
            damped = projected_cov + 1e-6 * np.eye(self.ndim, dtype=np.float32)
            kalman_gain = np.linalg.solve(damped.T, cov_ht.T).T

        innovation = np.asarray(measurement, dtype=np.float32) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ h_mat @ covariance
        return new_mean.astype(np.float32), new_cov.astype(np.float32)


class CameraMotionCompensator:
    """Sparse-feature global affine estimator for optional camera motion.

    The OpenCV dependency is imported lazily and only when CMC is enabled.
    Currently ``method="orb"`` is implemented because it is available from the
    normal OpenCV wheels and does not require contrib-only VideoStab bindings.
    """

    def __init__(
        self,
        method: str = "orb",
        max_features: int = 1000,
        min_matches: int = 12,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        if method != "orb":
            raise ValueError("Only cmc_method='orb' is currently supported")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "camera_motion=True requires OpenCV. Install the inference extra "
                "or opencv-python/opencv-contrib-python."
            ) from exc

        self.method = method
        self.max_features = int(max_features)
        self.min_matches = int(min_matches)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)
        self._cv2 = cv2
        self._previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous frame."""
        self._previous_gray = None

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate affine transform mapping previous-frame pixels to current."""
        gray = self._to_gray(frame)
        if self._previous_gray is None:
            self._previous_gray = gray
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        affine = self._estimate_orb(self._previous_gray, gray)
        self._previous_gray = gray
        return affine

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.uint8, copy=False)
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)

    def _estimate_orb(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        cv2 = self._cv2
        orb = cv2.ORB_create(nfeatures=self.max_features)
        kp_prev, desc_prev = orb.detectAndCompute(previous, None)
        kp_curr, desc_curr = orb.detectAndCompute(current, None)
        if desc_prev is None or desc_curr is None or len(kp_prev) < self.min_matches or len(kp_curr) < self.min_matches:
            return identity

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(desc_prev, desc_curr), key=lambda m: m.distance)
        if len(matches) < self.min_matches:
            return identity

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        affine, _ = cv2.estimateAffinePartial2D(
            pts_prev,
            pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        if affine is None or not np.all(np.isfinite(affine)):
            return identity
        return affine.astype(np.float32)


class BOTrack:
    """Single-track state used by :class:`BoTSORT`."""

    shared_kalman: KalmanFilterXYWH = KalmanFilterXYWH()
    _count: int = 0

    __slots__ = (
        "xywh",
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
        "curr_feat",
        "smooth_feat",
        "alpha",
    )

    def __init__(self, xywh: np.ndarray, score: float, class_id: int, feature: np.ndarray | None, alpha: float):
        self.xywh = xywh.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.kalman_filter: KalmanFilterXYWH | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        self.last_xyxy = _xywh_to_xyxy(self.xywh)
        self.curr_feat: np.ndarray | None = None
        self.smooth_feat: np.ndarray | None = None
        self.alpha = float(alpha)
        self.update_features(feature)

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls) -> None:
        cls._count = 0

    @staticmethod
    def multi_predict(tracks: list[BOTrack]) -> None:
        if not tracks:
            return
        means = np.stack([t.mean for t in tracks])
        covs = np.stack([t.covariance for t in tracks])
        for i, track in enumerate(tracks):
            if track.state != TrackState.Tracked:
                means[i, 6:] = 0.0
        new_means, new_covs = BOTrack.shared_kalman.multi_predict(means, covs)
        for i, track in enumerate(tracks):
            track.mean = new_means[i]
            track.covariance = new_covs[i]

    @staticmethod
    def multi_gmc(tracks: list[BOTrack], affine: np.ndarray) -> None:
        if not tracks:
            return
        for track in tracks:
            track.apply_gmc(affine)

    def apply_gmc(self, affine: np.ndarray) -> None:
        """Warp the predicted bbox by a global camera-motion affine."""
        if self.mean is None:
            self.last_xyxy = _warp_xyxy(self.last_xyxy.reshape(1, 4), affine)[0]
            self.xywh = _xyxy_to_xywh(self.last_xyxy.reshape(1, 4))[0]
            return
        warped_xyxy = _warp_xyxy(_xywh_to_xyxy(self.mean[:4]).reshape(1, 4), affine)[0]
        warped_xywh = _xyxy_to_xywh(warped_xyxy.reshape(1, 4))[0]
        self.mean[:4] = warped_xywh

    def update_features(self, feature: np.ndarray | None) -> None:
        if feature is None:
            return
        feat = _normalize_features(np.asarray(feature, dtype=np.float32))
        if feat is None or feat.size == 0:
            return
        self.curr_feat = feat[0]
        if self.smooth_feat is None:
            self.smooth_feat = self.curr_feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * self.curr_feat
            norm = float(np.linalg.norm(self.smooth_feat))
            if norm > 1e-12:
                self.smooth_feat = (self.smooth_feat / norm).astype(np.float32)

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = BOTrack.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.xywh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        assert self.kalman_filter is not None and self.mean is not None
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = BOTrack.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_xyxy = new_track.last_xyxy
        self.update_features(new_track.curr_feat)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        assert self.kalman_filter is not None and self.mean is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_xyxy = new_track.last_xyxy
        self.update_features(new_track.curr_feat)

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @property
    def tlbr(self) -> np.ndarray:
        """xyxy from the current state estimate."""
        if self.mean is None:
            return self.last_xyxy
        return _xywh_to_xyxy(self.mean[:4])


def _joint_tracks(a: list[BOTrack], b: list[BOTrack]) -> list[BOTrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _sub_tracks(a: list[BOTrack], b: list[BOTrack]) -> list[BOTrack]:
    ids_b = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids_b]


def _remove_duplicate_tracks(a: list[BOTrack], b: list[BOTrack]) -> tuple[list[BOTrack], list[BOTrack]]:
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


class BoTSORT(BaseTracker):
    """Native BoT-SORT with optional CMC and ReID.

    Args:
        track_high_thresh: High-confidence detection threshold.
        track_low_thresh: Low-confidence rescue threshold.
        new_track_thresh: Minimum score for starting a new track.
        match_thresh: Linear-assignment cost cutoff.
        track_buffer: Number of frames a lost track is retained at 30 FPS.
        frame_rate: Stream frame rate used to scale ``track_buffer``.
        class_aware: If true, block cross-class associations.
        camera_motion: Enables ORB/RANSAC camera motion compensation.
        cmc_method: Camera motion method. Currently only ``"orb"``.
        with_reid: Force ReID on/off. ``None`` auto-enables it when features
            or ``reid_encoder`` are available.
        reid_encoder: Optional callable ``(frame, xyxy) -> [N, D]``.
        feature_key: Key in ``detections.data`` containing precomputed
            embeddings.
        appearance_thresh: Minimum cosine similarity for accepting appearance.
        proximity_thresh: Minimum IoU for allowing appearance to override IoU.
        reid_alpha: Exponential smoothing coefficient for track embeddings.
    """

    name = "botsort"

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        class_aware: bool = False,
        camera_motion: bool = False,
        cmc_method: str = "orb",
        cmc_max_features: int = 1000,
        cmc_min_matches: int = 12,
        with_reid: bool | None = None,
        reid_encoder: ReIDEncoder | None = None,
        feature_key: str = "features",
        appearance_thresh: float = 0.25,
        proximity_thresh: float = 0.5,
        reid_alpha: float = 0.9,
        # supervision-style aliases
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
        **_: Any,
    ) -> None:
        if track_activation_threshold is not None:
            track_high_thresh = float(track_activation_threshold)
            new_track_thresh = max(new_track_thresh, track_high_thresh + 0.1)
        if lost_track_buffer is not None:
            track_buffer = int(lost_track_buffer)
        if minimum_matching_threshold is not None:
            match_thresh = float(minimum_matching_threshold)

        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate = int(frame_rate)
        self.class_aware = bool(class_aware)
        self.camera_motion = bool(camera_motion)
        self.feature_key = feature_key
        self.reid_encoder = reid_encoder
        self._force_reid = with_reid
        self.with_reid = bool(reid_encoder) if with_reid is None else bool(with_reid)
        self.appearance_thresh = float(appearance_thresh)
        self.proximity_thresh = float(proximity_thresh)
        self.reid_alpha = float(reid_alpha)

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        self.tracked_stracks: list[BOTrack] = []
        self.lost_stracks: list[BOTrack] = []
        self.removed_stracks: list[BOTrack] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        self.cmc = (
            CameraMotionCompensator(
                method=cmc_method,
                max_features=cmc_max_features,
                min_matches=cmc_min_matches,
            )
            if self.camera_motion
            else None
        )
        BOTrack.reset_id()

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        if self.cmc is not None:
            self.cmc.reset()
        BOTrack.reset_id()

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        self.frame_id += 1

        xyxy = detections.xyxy.astype(np.float32) if detections.xyxy is not None else np.empty((0, 4), dtype=np.float32)
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
        features = self._get_features(detections, frame, xyxy)

        activated: list[BOTrack] = []
        refind: list[BOTrack] = []
        lost: list[BOTrack] = []
        removed: list[BOTrack] = []

        remain_inds = scores >= self.track_high_thresh
        inds_low = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)

        dets_high_xyxy = xyxy[remain_inds]
        dets_high_scores = scores[remain_inds]
        dets_high_classes = class_ids[remain_inds]
        dets_high_features = features[remain_inds] if features is not None else None

        dets_low_xyxy = xyxy[inds_low]
        dets_low_scores = scores[inds_low]
        dets_low_classes = class_ids[inds_low]
        dets_low_features = features[inds_low] if features is not None else None

        detections_high = self._build_tracks(dets_high_xyxy, dets_high_scores, dets_high_classes, dets_high_features)
        detections_low = self._build_tracks(dets_low_xyxy, dets_low_scores, dets_low_classes, dets_low_features)

        unconfirmed: list[BOTrack] = []
        tracked_stracks: list[BOTrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = _joint_tracks(tracked_stracks, self.lost_stracks)
        BOTrack.multi_predict(strack_pool)
        if self.cmc is not None:
            affine = self.cmc.estimate(frame)
            BOTrack.multi_gmc(strack_pool, affine)

        dists = self._association_cost(
            strack_pool,
            detections_high,
            det_scores=dets_high_scores,
            fuse_det_score=True,
            use_reid=features is not None,
        )
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

        r_tracked = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].state == TrackState.Tracked]
        dists2 = self._iou_cost(r_tracked, detections_low)
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

        detections_remaining = [detections_high[int(i)] for i in u_det]
        dists3 = self._association_cost(
            unconfirmed,
            detections_remaining,
            det_scores=np.array([d.score for d in detections_remaining], dtype=np.float32),
            fuse_det_score=True,
            use_reid=False,
        )
        matches3, u_unc, u_det3 = linear_assignment(dists3, thresh=0.7)

        for it, idet in matches3:
            unconfirmed[int(it)].update(detections_remaining[int(idet)], self.frame_id)
            activated.append(unconfirmed[int(it)])

        for it in u_unc:
            track = unconfirmed[int(it)]
            track.mark_removed()
            removed.append(track)

        for inew in u_det3:
            track = detections_remaining[int(inew)]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint_tracks(self.tracked_stracks, activated)
        self.tracked_stracks = _joint_tracks(self.tracked_stracks, refind)
        self.lost_stracks = _sub_tracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub_tracks(self.lost_stracks, removed)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_tracks(self.tracked_stracks, self.lost_stracks)

        output_tracks = [t for t in self.tracked_stracks if t.is_activated]
        if not output_tracks:
            empty = sv.Detections.empty()
            empty.tracker_id = np.empty((0,), dtype=int)
            return empty

        out = sv.Detections(
            xyxy=np.stack([t.last_xyxy for t in output_tracks]).astype(np.float32),
            confidence=np.array([t.score for t in output_tracks], dtype=np.float32),
            class_id=np.array([t.class_id for t in output_tracks], dtype=int),
        )
        out.tracker_id = np.array([t.track_id for t in output_tracks], dtype=int)
        return out

    def _get_features(self, detections: sv.Detections, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        if self._force_reid is False:
            return None
        if len(xyxy) == 0:
            return np.empty((0, 0), dtype=np.float32) if self.with_reid else None

        data = getattr(detections, "data", {}) or {}
        raw_features = data.get(self.feature_key)
        if raw_features is not None:
            features = _normalize_features(np.asarray(raw_features, dtype=np.float32))
            if features is None or features.shape[0] != len(xyxy):
                raise ValueError(
                    f"detections.data[{self.feature_key!r}] must have one feature per detection; "
                    f"got {None if features is None else features.shape[0]} for {len(xyxy)} detections"
                )
            return features

        if self.reid_encoder is not None:
            features = _normalize_features(self.reid_encoder(frame, xyxy))
            if features is None or features.shape[0] != len(xyxy):
                raise ValueError(
                    f"reid_encoder must return one feature per detection; "
                    f"got {None if features is None else features.shape[0]} for {len(xyxy)} detections"
                )
            return features

        if self.with_reid:
            raise ValueError(
                "BoTSORT(with_reid=True) requires precomputed detections.data"
                f"[{self.feature_key!r}] or a reid_encoder(frame, xyxy) callable"
            )
        return None

    def _association_cost(
        self,
        tracks: list[BOTrack],
        dets: list[BOTrack],
        det_scores: np.ndarray,
        fuse_det_score: bool,
        use_reid: bool,
    ) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)

        iou_cost_raw = self._iou_cost(tracks, dets, apply_class_mask=False)
        class_mask = self._class_mismatch_mask(tracks, dets)
        cost = fuse_score(iou_cost_raw, det_scores) if fuse_det_score else iou_cost_raw.copy()
        cost = np.where(class_mask, 1e6, cost)

        if not use_reid:
            return cost.astype(np.float32)

        appearance_cost = self._embedding_distance(tracks, dets)
        appearance_cost = np.where(appearance_cost > 1.0 - self.appearance_thresh, 1.0, appearance_cost)
        appearance_cost = np.where(iou_cost_raw > 1.0 - self.proximity_thresh, 1.0, appearance_cost)
        appearance_cost = np.where(class_mask, 1e6, appearance_cost)
        return np.minimum(cost, appearance_cost).astype(np.float32)

    def _iou_cost(self, tracks: list[BOTrack], dets: list[BOTrack], apply_class_mask: bool = True) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        cost = iou_distance(np.stack([t.tlbr for t in tracks]), np.stack([d.last_xyxy for d in dets]))
        if apply_class_mask:
            cost = np.where(self._class_mismatch_mask(tracks, dets), 1e6, cost)
        return cost.astype(np.float32)

    def _embedding_distance(self, tracks: list[BOTrack], dets: list[BOTrack]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        out = np.ones((len(tracks), len(dets)), dtype=np.float32)
        for i, track in enumerate(tracks):
            if track.smooth_feat is None:
                continue
            for j, det in enumerate(dets):
                if det.curr_feat is None:
                    continue
                sim = float(np.dot(track.smooth_feat, det.curr_feat))
                out[i, j] = 1.0 - np.clip(sim, -1.0, 1.0)
        return out

    def _class_mismatch_mask(self, tracks: list[BOTrack], dets: list[BOTrack]) -> np.ndarray:
        if not self.class_aware or not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=bool)
        track_classes = np.array([t.class_id for t in tracks], dtype=int)
        det_classes = np.array([d.class_id for d in dets], dtype=int)
        return track_classes[:, None] != det_classes[None, :]

    def _build_tracks(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        features: np.ndarray | None,
    ) -> list[BOTrack]:
        if xyxy.size == 0:
            return []
        xywh = _xyxy_to_xywh(xyxy)
        out: list[BOTrack] = []
        for i in range(len(xyxy)):
            feature = features[i] if features is not None and features.size else None
            track = BOTrack(xywh[i], float(scores[i]), int(classes[i]), feature=feature, alpha=self.reid_alpha)
            track.last_xyxy = xyxy[i].astype(np.float32)
            out.append(track)
        return out


__all__ = [
    "BOTrack",
    "BoTSORT",
    "CameraMotionCompensator",
    "KalmanFilterXYWH",
]
