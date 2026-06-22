"""Sweep OC-SORT process-noise Q scaling on a 3Hz video.

Without ground-truth IDs we use unsupervised proxy metrics:

* ``num_ids`` — total unique tracker IDs ever emitted. Lower is better
  (fewer ID switches; per-physical-car ID inflation is the dominant
  failure mode at 3Hz).
* ``mean_life`` — mean lifetime (in tracked frames) of an ID. Higher is
  better (an ID that survives many frames probably corresponds to one
  real car).
* ``short_id_ratio`` — fraction of IDs that live <= 2 frames. Lower is
  better (these are mostly fragments).
* ``mean_active`` — mean number of active IDs per frame (just for sanity;
  should be similar across Q if detector output is identical).

The Q matrix lives in :class:`KalmanFilterXYSR` as ``self._Q``. We scale
the default Q by ``q_scale`` after constructing the tracker — same Q for
position/velocity blocks, simplest knob with clear semantics.

Run::

    .venv/bin/python tools/tracking/sweep_q.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

from onnxtools import create_detector
from onnxtools.tracking import create_tracker

VIDEO = "data/R39_Bn_CamW__102_rss-ai-prd-suz-wt-gpu30_1920X1080_2025-05-14-07-29-59-898.mp4"
MODEL = "models/rfdetr-20250811.onnx"
STRIDE = 20  # 60fps source -> 3Hz
CONF = 0.4
Q_SCALES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def cache_detections(video: str, model: str, stride: int, conf: float):
    """Run the detector once and return a list of sv.Detections per kept frame."""
    detector = create_detector("rfdetr", model, conf_thres=conf, iou_thres=0.5)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video}")

    cached = []
    t0 = time.time()
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            result = detector(frame)
            # Convert Result -> sv.Detections so tracker.update accepts it.
            det = sv.Detections(
                xyxy=np.asarray(result.boxes, dtype=np.float32).reshape(-1, 4),
                confidence=np.asarray(result.scores, dtype=np.float32).reshape(-1),
                class_id=np.asarray(result.class_ids, dtype=int).reshape(-1),
            )
            cached.append((frame, det))
            kept += 1
        idx += 1
    cap.release()
    print(f"[detect] cached {kept} frames (stride={stride}) in {time.time() - t0:.1f}s")
    return cached


def run_tracker_with_q(cached, q_scale: float):
    """Re-run OC-SORT over cached detections with a scaled Q matrix.

    Returns a dict of proxy metrics.
    """
    tracker = create_tracker(
        "ocsort",
        det_thresh=CONF,
        max_age=10,  # ~3.3s at 3Hz — modest, so dead tracks don't hide ID churn
        min_hits=1,  # emit immediately so we count first-frame IDs too
        iou_threshold=0.2,  # looser than default; 3Hz needs more slack
    )
    # Scale every per-track KF's Q. OC-SORT builds one KF per tracker on
    # creation, so we patch the *class-level* shared instance the tracker
    # constructs each KalmanBoxTracker from. Simpler: patch the prototype
    # by monkey-patching the constructor result.
    # Each new KalmanBoxTracker creates its own KalmanFilterXYSR(); we wrap
    # the class so newly-constructed KFs get the scaled Q.
    from onnxtools.tracking import kalman as kalman_mod

    orig_init = kalman_mod.KalmanFilterXYSR.__init__

    def patched_init(self):
        orig_init(self)
        self._Q = (self._Q * q_scale).astype(np.float32)

    kalman_mod.KalmanFilterXYSR.__init__ = patched_init
    try:
        id_frames: dict[int, list[int]] = defaultdict(list)
        active_per_frame = []
        t0 = time.time()
        for fi, (frame, det) in enumerate(cached):
            tracked = tracker.update(det, frame)
            ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
            active_per_frame.append(len(ids))
            for tid in ids:
                id_frames[int(tid)].append(fi)
        elapsed = time.time() - t0
    finally:
        kalman_mod.KalmanFilterXYSR.__init__ = orig_init

    lifetimes = [len(v) for v in id_frames.values()]
    if not lifetimes:
        return {"q_scale": q_scale, "num_ids": 0}
    short = sum(1 for L in lifetimes if L <= 2)
    return {
        "q_scale": q_scale,
        "num_ids": len(lifetimes),
        "mean_life": float(np.mean(lifetimes)),
        "median_life": float(np.median(lifetimes)),
        "max_life": int(np.max(lifetimes)),
        "short_id_ratio": short / len(lifetimes),
        "mean_active": float(np.mean(active_per_frame)),
        "track_time_s": elapsed,
    }


def main():
    cached = cache_detections(VIDEO, MODEL, STRIDE, CONF)
    if not cached:
        print("no frames cached, abort")
        return

    rows = []
    for q in Q_SCALES:
        print(f"\n[sweep] q_scale={q}")
        m = run_tracker_with_q(cached, q)
        print(f"  -> {m}")
        rows.append(m)

    print("\n=== summary (3Hz, 60s window) ===")
    hdr = f"{'q_scale':>8} {'num_ids':>8} {'mean_life':>10} {'median':>8} {'max':>6} {'short<=2':>10} {'mean_act':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['q_scale']:>8.2f} {r['num_ids']:>8d} {r['mean_life']:>10.2f} "
            f"{r['median_life']:>8.1f} {r['max_life']:>6d} "
            f"{r['short_id_ratio']:>10.2%} {r['mean_active']:>10.2f}"
        )


if __name__ == "__main__":
    main()
