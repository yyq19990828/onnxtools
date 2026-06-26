"""Behavioural tests for native BoT-SORT."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from onnxtools.tracking import create_tracker
from onnxtools.tracking.botsort import BOTrack, BoTSORT, CameraMotionCompensator, _xyxy_to_xywh


def _dets(xyxy, scores=None, classes=None, features=None) -> sv.Detections:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    n = len(xyxy)
    data = {}
    if features is not None:
        data["features"] = np.asarray(features, dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.asarray(scores if scores is not None else [0.9] * n, dtype=np.float32),
        class_id=np.asarray(classes if classes is not None else [0] * n, dtype=int),
        data=data,
    )


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestBoTSortMotion:
    def test_stable_id_without_reid_or_cmc(self, frame):
        tracker = BoTSORT()
        ids = []
        for _ in range(5):
            out = tracker.update(_dets([[100, 100, 200, 220]]), frame)
            if out.tracker_id is not None and len(out.tracker_id) > 0:
                ids.append(int(out.tracker_id[0]))
        assert len(ids) >= 2
        assert len(set(ids)) == 1

    def test_empty_frame_ticks_lost_buffer(self, frame):
        tracker = BoTSORT(track_buffer=2, frame_rate=30)
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 220]]), frame)
        for _ in range(10):
            tracker.update(sv.Detections.empty(), frame)
        assert all(t.track_id != 1 for t in tracker.tracked_stracks)

    def test_reset_restarts_ids(self, frame):
        tracker = BoTSORT()
        for _ in range(3):
            tracker.update(_dets([[100, 100, 200, 220]]), frame)
        tracker.reset()
        out = None
        for _ in range(3):
            out = tracker.update(_dets([[100, 100, 200, 220]]), frame)
        assert out is not None and out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1


class TestBoTSortReID:
    def test_reid_requires_feature_source_when_forced(self, frame):
        tracker = BoTSORT(with_reid=True)
        with pytest.raises(ValueError, match="requires precomputed"):
            tracker.update(_dets([[100, 100, 200, 220]]), frame)

    def test_reid_accepts_precomputed_features(self, frame):
        tracker = BoTSORT(with_reid=True)
        features = [[1.0, 0.0, 0.0]]
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 220]], features=features), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1
        assert tracker.tracked_stracks[0].smooth_feat is not None

    def test_reid_encoder_is_used(self, frame):
        def encoder(_frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(xyxy), 1))

        tracker = BoTSORT(with_reid=True, reid_encoder=encoder)
        for _ in range(2):
            out = tracker.update(_dets([[100, 100, 200, 220]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1

    def test_reid_can_be_forced_off_even_when_features_exist(self, frame):
        def encoder(_frame: np.ndarray, _xyxy: np.ndarray) -> np.ndarray:
            raise AssertionError("encoder should not be called")

        tracker = BoTSORT(with_reid=False, reid_encoder=encoder)
        out = tracker.update(_dets([[100, 100, 200, 220]], features=[[1, 0]]), frame)
        assert out.tracker_id is not None
        assert int(out.tracker_id[0]) == 1
        assert tracker.tracked_stracks[0].smooth_feat is None

    def test_appearance_cost_prefers_matching_features(self):
        tracker = BoTSORT(with_reid=True, proximity_thresh=0.0, appearance_thresh=0.2)
        tracks = tracker._build_tracks(
            np.asarray([[100, 100, 200, 220], [102, 100, 202, 220]], dtype=np.float32),
            np.asarray([0.9, 0.9], dtype=np.float32),
            np.asarray([0, 0], dtype=int),
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        )
        for track in tracks:
            track.activate(tracker.kalman_filter, frame_id=1)

        dets = tracker._build_tracks(
            np.asarray([[101, 100, 201, 220], [103, 100, 203, 220]], dtype=np.float32),
            np.asarray([0.9, 0.9], dtype=np.float32),
            np.asarray([0, 0], dtype=int),
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        )
        cost = tracker._association_cost(
            tracks,
            dets,
            det_scores=np.asarray([0.9, 0.9], dtype=np.float32),
            fuse_det_score=False,
            use_reid=True,
        )
        assert cost[0, 0] < cost[0, 1]
        assert cost[1, 1] < cost[1, 0]


class TestCameraMotion:
    def test_camera_motion_switch_constructs_compensator(self):
        tracker = create_tracker("botsort", camera_motion=True)
        assert isinstance(tracker, BoTSORT)
        assert isinstance(tracker.cmc, CameraMotionCompensator)

    def test_cmc_reset_and_blank_frame_identity(self, frame):
        cmc = CameraMotionCompensator()
        affine = cmc.estimate(frame)
        np.testing.assert_allclose(affine, [[1, 0, 0], [0, 1, 0]], atol=1e-6)
        cmc.reset()
        affine = cmc.estimate(frame)
        np.testing.assert_allclose(affine, [[1, 0, 0], [0, 1, 0]], atol=1e-6)


def test_bo_track_uses_xywh_state():
    track = BOTrack(_xyxy_to_xywh(np.asarray([[0, 0, 20, 40]], dtype=np.float32))[0], 0.9, 0, None, 0.9)
    assert track.xywh.tolist() == [10.0, 20.0, 20.0, 40.0]
