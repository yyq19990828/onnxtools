"""Unit tests for 2D ByteTrack integration in InferencePipeline.

These tests intentionally avoid loading real ONNX models. They exercise only
the tracker-adjacent logic:

  * ``InferencePipeline._align_tracker_ids`` — alignment from ByteTrack output
    back to the per-detection / output_data order.
  * ``sv.ByteTrack`` baseline behaviour we rely on (stable IDs across frames,
    new IDs after lost-track-buffer expires).
  * ``supervision_labels`` tracker-id prefix.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from onnxtools.pipeline import InferencePipeline
from onnxtools.utils.supervision_labels import create_confidence_labels, create_ocr_labels


def _make_detections(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> sv.Detections:
    return sv.Detections(
        xyxy=boxes.astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=class_ids.astype(int),
    )


class TestByteTrackBaseline:
    """Guardrails on the supervision API we depend on."""

    def test_stable_id_across_frames(self):
        tracker = sv.ByteTrack(track_activation_threshold=0.25, minimum_matching_threshold=0.8)
        boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        scores = np.array([0.9])
        class_ids = np.array([0])

        ids = []
        for _ in range(5):
            tracked = tracker.update_with_detections(
                _make_detections(boxes, scores, class_ids)
            )
            assert tracked.tracker_id is not None
            assert len(tracked) == 1
            ids.append(int(tracked.tracker_id[0]))

        assert len(set(ids)) == 1, f"expected stable id, got {ids}"

    def test_new_id_after_buffer_expires(self):
        tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=2,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        scores = np.array([0.9])
        class_ids = np.array([0])

        # Establish track id on frame 0.
        first = tracker.update_with_detections(
            _make_detections(boxes, scores, class_ids)
        )
        first_id = int(first.tracker_id[0])

        # Push 10 empty frames — well past lost_track_buffer=2.
        for _ in range(10):
            tracker.update_with_detections(sv.Detections.empty())

        # Same box reappears — feed several frames so ByteTrack can confirm a
        # new track. Once any id is emitted it must not be the original.
        reappeared_ids: list[int] = []
        for _ in range(10):
            out = tracker.update_with_detections(
                _make_detections(boxes, scores, class_ids)
            )
            if out.tracker_id is not None and len(out.tracker_id) > 0:
                reappeared_ids.append(int(out.tracker_id[0]))

        assert reappeared_ids, "ByteTrack never re-acquired the target"
        assert all(tid != first_id for tid in reappeared_ids), (
            f"ByteTrack reused expired id {first_id}: {reappeared_ids}"
        )


class TestAlignTrackerIds:
    """`_align_tracker_ids` is the only piece of glue we own end-to-end."""

    def test_full_match(self):
        boxes = np.array(
            [[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]]
        )
        tracked = _make_detections(boxes, np.array([0.9, 0.8]), np.array([0, 1]))
        tracked.tracker_id = np.array([7, 11])

        output_data = [
            {"box2d": [10.0, 10.0, 50.0, 50.0], "type": "vehicle"},
            {"plate_box2d": [100.0, 100.0, 200.0, 200.0], "plate_name": "ABC"},
        ]

        ids = InferencePipeline._align_tracker_ids(boxes, tracked, output_data)
        assert ids is not None
        assert list(ids) == [7, 11]
        assert output_data[0]["tracker_id"] == 7
        assert output_data[1]["tracker_id"] == 11

    def test_unmatched_detections_get_none(self):
        # boxes contains two detections; tracker only kept the first one
        boxes = np.array(
            [[10.0, 10.0, 50.0, 50.0], [300.0, 300.0, 400.0, 400.0]]
        )
        tracked = _make_detections(boxes[:1], np.array([0.9]), np.array([0]))
        tracked.tracker_id = np.array([3])

        output_data = [
            {"box2d": [10.0, 10.0, 50.0, 50.0], "type": "vehicle"},
            {"box2d": [300.0, 300.0, 400.0, 400.0], "type": "vehicle"},
        ]
        ids = InferencePipeline._align_tracker_ids(boxes, tracked, output_data)
        assert ids is not None
        assert ids[0] == 3 and ids[1] is None
        assert output_data[0]["tracker_id"] == 3
        assert output_data[1]["tracker_id"] is None

    def test_low_conf_plate_skipped_in_output(self):
        # boxes has two detections but output_data only carries the high-conf
        # entry (simulates the pipeline's plate_conf_thres skip).
        boxes = np.array(
            [[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]]
        )
        tracked = _make_detections(boxes, np.array([0.9, 0.8]), np.array([0, 1]))
        tracked.tracker_id = np.array([5, 9])

        output_data = [
            {"plate_box2d": [100.0, 100.0, 200.0, 200.0], "plate_name": "ABC"},
        ]
        ids = InferencePipeline._align_tracker_ids(boxes, tracked, output_data)
        assert ids is not None
        assert list(ids) == [5, 9]
        # output_data was shorter than boxes — only the high-conf entry exists
        # and must still receive the correct id (matched by xyxy).
        assert output_data[0]["tracker_id"] == 9

    def test_empty_tracker_output(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        tracked = sv.Detections.empty()
        output_data = [{"box2d": [10.0, 10.0, 50.0, 50.0]}]
        ids = InferencePipeline._align_tracker_ids(boxes, tracked, output_data)
        assert ids is None
        assert output_data[0]["tracker_id"] is None


class TestLabelTrackerPrefix:
    def test_confidence_labels_no_prefix_when_ids_none(self):
        scores = np.array([0.95, 0.50])
        labels = create_confidence_labels(scores)
        assert labels == ["0.95", "0.50"]

    def test_confidence_labels_with_ids(self):
        scores = np.array([0.95, 0.50])
        ids = np.array([7, None], dtype=object)
        labels = create_confidence_labels(scores, tracker_ids=ids)
        assert labels == ["#7 0.95", "0.50"]

    def test_ocr_labels_with_ids(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
        scores = np.array([0.9])
        class_ids = np.array([0])
        labels = create_ocr_labels(
            boxes, scores, class_ids,
            plate_results=[None],
            class_names={0: "vehicle"},
            tracker_ids=[42],
        )
        assert labels == ["#42 vehicle 0.90"]


@pytest.mark.parametrize("tid", [None, float("nan"), "not-an-int"])
def test_format_prefix_edge_cases(tid):
    from onnxtools.utils.supervision_labels import _format_tracker_prefix

    assert _format_tracker_prefix(tid) == ""
