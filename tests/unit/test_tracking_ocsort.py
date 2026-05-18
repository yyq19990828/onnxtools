"""Behavioural tests for the hand-rolled OCSORT."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from onnxtools.tracking import create_tracker
from onnxtools.tracking.ocsort import OCSORT, KalmanBoxTracker


def _dets(xyxy, scores=None, classes=None) -> sv.Detections:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.asarray(scores if scores is not None else [0.9] * n, dtype=np.float32),
        class_id=np.asarray(classes if classes is not None else [0] * n, dtype=int),
    )


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestEmitTiming:
    def test_min_hits_delays_emission(self, frame):
        tracker = OCSORT(min_hits=3, det_thresh=0.5)
        emitted = []
        for _ in range(6):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
            emitted.append(out.tracker_id.size if out.tracker_id is not None else 0)
        # Once frame_count > min_hits, must emit on subsequent frames.
        assert emitted[-1] == 1

    def test_stable_id(self, frame):
        tracker = OCSORT(min_hits=1, det_thresh=0.5)
        ids = []
        for i in range(6):
            box = [100 + i, 100 + i, 200 + i, 200 + i]
            out = tracker.update(_dets([box]), frame)
            if out.tracker_id is not None and out.tracker_id.size > 0:
                ids.append(int(out.tracker_id[0]))
        assert len(set(ids)) == 1, f"expected stable id, got {ids}"


class TestMaxAge:
    def test_track_expires_after_max_age(self, frame):
        tracker = OCSORT(min_hits=1, max_age=3, det_thresh=0.5)
        # Build a track.
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 200]]), frame)
        # Drop it for >max_age frames.
        for _ in range(10):
            tracker.update(sv.Detections.empty(), frame)
        assert len(tracker.trackers) == 0


class TestVelocity:
    def test_velocity_set_on_reacquisition(self):
        """ORU should populate velocity after the second observation."""
        kb = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float32), 0.9, 0)
        assert kb.velocity is None
        kb.predict()
        kb.update(np.array([110, 110, 210, 210], dtype=np.float32), 0.9, 0)
        # First update has a sentinel "no previous observation" — velocity may
        # be None; do a second to trigger ORU.
        kb.predict()
        kb.update(np.array([120, 120, 220, 220], dtype=np.float32), 0.9, 0)
        assert kb.velocity is not None
        # Unit vector.
        assert np.isclose(np.linalg.norm(kb.velocity), 1.0, atol=1e-3)


class TestKwargAliases:
    def test_supervision_alias_kwargs_accepted(self):
        tracker = create_tracker(
            "ocsort",
            track_activation_threshold=0.4,
            lost_track_buffer=20,
            minimum_matching_threshold=0.7,
            frame_rate=30,
            some_unknown=99,
        )
        assert isinstance(tracker, OCSORT)
        assert tracker.det_thresh == 0.4
        assert tracker.max_age == 20
        # iou_threshold = 1 - 0.7 = 0.3 (approx).
        assert abs(tracker.iou_threshold - 0.3) < 1e-6


class TestReset:
    def test_reset_clears_trackers(self, frame):
        tracker = OCSORT(min_hits=1, det_thresh=0.5)
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert len(tracker.trackers) > 0
        tracker.reset()
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
