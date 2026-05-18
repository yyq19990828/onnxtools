"""Behavioural tests for the hand-rolled ByteTrackNative."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from onnxtools.tracking import create_tracker
from onnxtools.tracking.bytetrack import ByteTrackNative


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


class TestStableID:
    def test_stable_id_across_frames(self, frame):
        tracker = ByteTrackNative()
        ids = []
        for _ in range(5):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
            if out.tracker_id is not None and len(out.tracker_id) > 0:
                ids.append(int(out.tracker_id[0]))
        assert len(ids) >= 2
        assert len(set(ids)) == 1, f"expected stable id, got {ids}"

    def test_empty_frame_ticks_state(self, frame):
        """Empty detection frames must still advance the lost-buffer clock."""
        tracker = ByteTrackNative(track_buffer=2, frame_rate=30)
        # Establish track.
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 200]]), frame)
        # Push empty frames past buffer.
        for _ in range(10):
            tracker.update(sv.Detections.empty(), frame)
        # Re-emit same box — should get a NEW id since old expired.
        ids = []
        for _ in range(5):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
            if out.tracker_id is not None and len(out.tracker_id) > 0:
                ids.append(int(out.tracker_id[0]))
        assert ids, "tracker never re-acquired target"
        assert all(i != 1 for i in ids), f"reused expired id 1: {ids}"


class TestBufferRetention:
    def test_id_retained_within_buffer(self, frame):
        tracker = ByteTrackNative(track_buffer=30, frame_rate=30)
        # Activate.
        for _ in range(3):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        first_id = int(out.tracker_id[0])

        # Short occlusion (< buffer).
        for _ in range(5):
            tracker.update(sv.Detections.empty(), frame)

        out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out.tracker_id is not None and len(out.tracker_id) == 1
        assert int(out.tracker_id[0]) == first_id


class TestReset:
    def test_reset_restarts_ids(self, frame):
        tracker = ByteTrackNative()
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 200]]), frame)
        tracker.reset()
        out = None
        for _ in range(3):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out is not None and out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1


class TestKwargAliases:
    def test_supervision_alias_kwargs_accepted(self):
        # Factory must pass through supervision-style kwargs without crashing.
        tracker = create_tracker(
            "bytetrack_native",
            track_activation_threshold=0.3,
            lost_track_buffer=15,
            minimum_matching_threshold=0.7,
            frame_rate=30,
            some_unknown=42,
        )
        assert isinstance(tracker, ByteTrackNative)
        assert tracker.track_buffer == 15
        assert tracker.match_thresh == 0.7
        assert tracker.track_high_thresh == 0.3
