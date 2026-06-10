"""Sweep ByteTrack (supervision + native) on the same 3Hz video.

ByteTrack uses a different KF (XYAH 8D) and a 3-stage association
(high-score → low-score → unconfirmed). Key knobs:

* ``match_thresh`` — first-stage IoU+fuse-score cost cap. ByteTrack
  uses cost = 1 - IoU·det_score, so match_thresh=0.8 means "accept if
  fused cost < 0.8" — looser than it sounds. Same direction as OC-SORT's
  iou_threshold but inverted semantics.
* ``track_buffer`` — frames a Lost track can stay before Removed.
* ``track_high_thresh`` / ``new_track_thresh`` — detection score gates.

We compare:
  1. ``bytetrack`` (supervision wrapper) at defaults
  2. ``bytetrack_native`` (our own) at defaults
  3. native, sweeping match_thresh
  4. native, sweeping track_buffer
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

from onnxtools import create_detector
from onnxtools.tracking import create_tracker

VIDEO = "data/R39_Bn_CamW__102_rss-ai-prd-suz-wt-gpu30_1920X1080_2025-05-14-07-29-59-898.mp4"
MODEL = "models/rfdetr-20250811.onnx"
STRIDE = 20
CONF = 0.4


def cache_detections():
    detector = create_detector("rfdetr", MODEL, conf_thres=CONF, iou_thres=0.5)
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
    print(f"[detect] cached {len(cached)} frames")
    return cached


def run(cached, algo: str, **kwargs):
    tracker = create_tracker(algo, **kwargs)
    id_frames: dict[int, list[int]] = defaultdict(list)
    active = []
    for fi, (frame, det) in enumerate(cached):
        tracked = tracker.update(det, frame)
        ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
        active.append(len(ids))
        for tid in ids:
            id_frames[int(tid)].append(fi)
    lifetimes = [len(v) for v in id_frames.values()]
    if not lifetimes:
        return {
            "num_ids": 0,
            "mean_life": 0.0,
            "median_life": 0.0,
            "max_life": 0,
            "short_id_ratio": 0.0,
            "mean_active": float(np.mean(active)),
        }
    return {
        "num_ids": len(lifetimes),
        "mean_life": float(np.mean(lifetimes)),
        "median_life": float(np.median(lifetimes)),
        "max_life": int(np.max(lifetimes)),
        "short_id_ratio": sum(1 for L in lifetimes if L <= 2) / len(lifetimes),
        "mean_active": float(np.mean(active)),
    }


def pp(label, m):
    print(
        f"{label:>26}  num_ids={m['num_ids']:>4d}  mean_life={m['mean_life']:>5.1f}  "
        f"median={m['median_life']:>4.1f}  max={m['max_life']:>4d}  "
        f"short<=2={m['short_id_ratio']:>6.2%}  act={m['mean_active']:>5.2f}"
    )


def main():
    cached = cache_detections()

    print("\n=== defaults ===")
    pp("bytetrack(supervision)", run(cached, "bytetrack"))
    pp("bytetrack_native", run(cached, "bytetrack_native"))
    pp("ocsort (last winner)", run(cached, "ocsort", det_thresh=CONF, max_age=10, min_hits=1, iou_threshold=0.2))

    print("\n=== bytetrack_native: match_thresh sweep (track_buffer=30) ===")
    for mt in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        pp(
            f"match_thresh={mt}",
            run(
                cached,
                "bytetrack_native",
                track_high_thresh=CONF,
                new_track_thresh=CONF,
                match_thresh=mt,
                track_buffer=30,
                frame_rate=30,
            ),
        )

    print("\n=== bytetrack_native: track_buffer sweep (match_thresh=0.95) ===")
    for tb in [5, 10, 30, 60, 120, 240]:
        pp(
            f"track_buffer={tb}",
            run(
                cached,
                "bytetrack_native",
                track_high_thresh=CONF,
                new_track_thresh=CONF,
                match_thresh=0.95,
                track_buffer=tb,
                frame_rate=30,
            ),
        )

    print("\n=== bytetrack_native: frame_rate hint (with track_buffer=30) ===")
    # buffer_size = int(frame_rate / 30 * track_buffer) — tell it 3Hz.
    for fr in [30, 10, 3]:
        pp(
            f"frame_rate={fr}",
            run(
                cached,
                "bytetrack_native",
                track_high_thresh=CONF,
                new_track_thresh=CONF,
                match_thresh=0.95,
                track_buffer=30,
                frame_rate=fr,
            ),
        )

    print("\n=== best-of-both combo ===")
    pp(
        "native mt=0.95 tb=240",
        run(
            cached,
            "bytetrack_native",
            track_high_thresh=CONF,
            new_track_thresh=CONF,
            match_thresh=0.95,
            track_buffer=240,
            frame_rate=30,
        ),
    )
    pp("ocsort iou=0.05 ma=60", run(cached, "ocsort", det_thresh=CONF, max_age=60, min_hits=1, iou_threshold=0.05))


if __name__ == "__main__":
    main()
