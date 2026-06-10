"""Sweep OC-SORT iou_threshold and max_age on the same 3Hz video.

We learned from sweep_q.py that Q is nearly inert here. The real
suspects at 3Hz are:

* ``iou_threshold`` — the hard IoU gate. Too tight (0.3 default) means
  any frame-to-frame box drift > 70% IoU loss kills the association and
  spawns a new ID. At 3Hz, frame-to-frame drift is huge, so loosening
  this should help — until it starts merging different cars.
* ``max_age`` — how many frames a track can stay Lost before deletion.
  At 3Hz, max_age=10 = 3.3s; brief occlusions need more buffer.

We reuse cached detections from sweep_q's logic for fairness, but inline
the helpers so this file is standalone.
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


def run(cached, *, iou_threshold: float, max_age: int):
    tracker = create_tracker(
        "ocsort",
        det_thresh=CONF,
        max_age=max_age,
        min_hits=1,
        iou_threshold=iou_threshold,
    )
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
        return {"num_ids": 0}
    return {
        "num_ids": len(lifetimes),
        "mean_life": float(np.mean(lifetimes)),
        "median_life": float(np.median(lifetimes)),
        "max_life": int(np.max(lifetimes)),
        "short_id_ratio": sum(1 for L in lifetimes if L <= 2) / len(lifetimes),
        "mean_active": float(np.mean(active)),
    }


def main():
    cached = cache_detections()

    print("\n=== iou_threshold sweep (max_age=10) ===")
    hdr = f"{'iou_thr':>8} {'num_ids':>8} {'mean_life':>10} {'median':>8} {'max':>6} {'short<=2':>10}"
    print(hdr)
    print("-" * len(hdr))
    for iou in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        m = run(cached, iou_threshold=iou, max_age=10)
        print(
            f"{iou:>8.2f} {m['num_ids']:>8d} {m['mean_life']:>10.2f} "
            f"{m['median_life']:>8.1f} {m['max_life']:>6d} {m['short_id_ratio']:>10.2%}"
        )

    print("\n=== max_age sweep (iou_threshold=0.15) ===")
    hdr = f"{'max_age':>8} {'num_ids':>8} {'mean_life':>10} {'median':>8} {'max':>6} {'short<=2':>10}"
    print(hdr)
    print("-" * len(hdr))
    for ma in [3, 5, 10, 20, 30, 60]:
        m = run(cached, iou_threshold=0.15, max_age=ma)
        print(
            f"{ma:>8d} {m['num_ids']:>8d} {m['mean_life']:>10.2f} "
            f"{m['median_life']:>8.1f} {m['max_life']:>6d} {m['short_id_ratio']:>10.2%}"
        )


if __name__ == "__main__":
    main()
