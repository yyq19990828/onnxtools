"""Quantify the effect of ClassVotingTracker on per-track label flicker.

For each tracker_id we count:
  * raw_distinct_classes — how many different class_ids it ever held
  * raw_flips           — adjacent-frame class_id changes
  * voted_distinct      — same, after voting wrapper
  * voted_flips         — same, after voting wrapper (should be 0 by design)

Voting collapses raw_distinct -> 1 (constant per id). Useful metric is
"how many IDs had ANY flicker without voting" — that's the fraction of
the fleet that voting actually cleaned up.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

from onnxtools import create_detector
from onnxtools.tracking import create_tracker
from onnxtools.tracking.class_voting import ClassVotingTracker

VIDEO = "data/track/R27_Aw_CamS__000_rss-ai-prd-suz-wt-gpu23_1920X1080_2026-04-08-19-44-14-485_60.mp4"
MODEL = "models/rfdetr-small_shangdian_20260518_unified.onnx"
MODEL_TYPE = "rfdetr_unified"
STRIDE = 20
CONF = 0.4

TRACKER_KWARGS = dict(
    track_high_thresh=CONF,
    new_track_thresh=CONF,
    match_thresh=0.95,
    track_buffer=120,
    frame_rate=30,
)


def cache_detections():
    detector = create_detector(MODEL_TYPE, MODEL, conf_thres=CONF, iou_thres=0.5)
    cap = cv2.VideoCapture(VIDEO)
    cached = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE == 0:
            r = detector(frame)
            cached.append(
                (
                    frame,
                    sv.Detections(
                        xyxy=np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4),
                        confidence=np.asarray(r.scores, dtype=np.float32).reshape(-1),
                        class_id=np.asarray(r.class_ids, dtype=int).reshape(-1),
                    ),
                )
            )
        idx += 1
    cap.release()
    return cached


def run_and_collect(cached, voting: bool):
    inner = create_tracker("bytetrack_native", **TRACKER_KWARGS)
    tracker = ClassVotingTracker(inner) if voting else inner
    # tid -> ordered list of class_ids per frame seen
    series: dict[int, list[int]] = defaultdict(list)
    for frame, det in cached:
        out = tracker.update(det, frame)
        if out.tracker_id is None:
            continue
        for tid, cid in zip(out.tracker_id, out.class_id):
            series[int(tid)].append(int(cid))
    return series


def summarise(label, series):
    distinct = [len(set(s)) for s in series.values() if s]
    flips = [sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) for s in series.values() if s]
    ids_with_any_flicker = sum(1 for d in distinct if d > 1)
    print(f"\n=== {label} ===")
    print(f"  total IDs                : {len(distinct)}")
    print(f"  IDs with ANY class change: {ids_with_any_flicker} ({ids_with_any_flicker/max(len(distinct),1):.1%})")
    print(f"  mean distinct classes/id : {np.mean(distinct):.2f}")
    print(f"  max  distinct classes/id : {max(distinct) if distinct else 0}")
    print(f"  total class flips        : {sum(flips)}")


def main():
    cached = cache_detections()
    print(f"cached {len(cached)} frames")

    s_raw = run_and_collect(cached, voting=False)
    s_voted = run_and_collect(cached, voting=True)

    summarise("WITHOUT voting", s_raw)
    summarise("WITH voting   ", s_voted)


if __name__ == "__main__":
    main()
