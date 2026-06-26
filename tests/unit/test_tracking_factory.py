"""Unit tests for the ``onnxtools.tracking`` factory + adapters.

No ONNX models required. Currently the live package only ships the
supervision-backed ``bytetrack`` adapter; the BoxMOT adapter has been
archived (see ``onnxtools/tracking/_archive/`` for the why).
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from onnxtools.tracking import SUPPORTED_TRACKERS, BaseTracker, SupervisionByteTrack, create_tracker


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


class TestFactory:
    def test_supported_lists_only_shipped_algos(self):
        # Living source of truth — when we vendor more, this expands.
        assert SUPPORTED_TRACKERS == ("bytetrack", "bytetrack_native", "ocsort", "botsort")

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            create_tracker("nonexistent")

    def test_native_bytetrack_is_constructed(self, frame):
        from onnxtools.tracking.bytetrack import ByteTrackNative

        tracker = create_tracker("bytetrack_native")
        assert isinstance(tracker, BaseTracker)
        assert isinstance(tracker, ByteTrackNative)
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_ocsort_is_constructed(self, frame):
        from onnxtools.tracking.ocsort import OCSORT

        tracker = create_tracker("ocsort", min_hits=1)
        assert isinstance(tracker, BaseTracker)
        assert isinstance(tracker, OCSORT)
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_botsort_is_constructed(self, frame):
        from onnxtools.tracking.botsort import BoTSORT

        tracker = create_tracker("botsort")
        assert isinstance(tracker, BaseTracker)
        assert isinstance(tracker, BoTSORT)
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_bytetrack_is_supervision_adapter(self):
        tracker = create_tracker("bytetrack")
        assert isinstance(tracker, BaseTracker)
        assert isinstance(tracker, SupervisionByteTrack)

    def test_bytetrack_update_returns_detections(self, frame):
        tracker = create_tracker("bytetrack")
        # supervision activates tracks on the second consistent observation
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert isinstance(out, sv.Detections)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_bytetrack_reset_restarts_ids(self, frame):
        tracker = create_tracker("bytetrack")
        for _ in range(2):
            tracker.update(_dets([[100, 100, 200, 200]]), frame)
        tracker.reset()
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 200]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_unknown_kwargs_absorbed(self):
        # Factory must forward arbitrary kwargs without crashing — callers
        # build a shared kwargs dict that may include keys irrelevant to the
        # chosen backend.
        tracker = create_tracker(
            "bytetrack",
            frame_rate=15,
            lost_track_buffer=12,
            # algo-specific extra that supervision doesn't know:
            some_future_param=42,
        )
        assert isinstance(tracker, SupervisionByteTrack)
