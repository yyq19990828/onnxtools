"""Performance benchmarks for hand-rolled trackers.

Realistic budget @ 200 persistent detections on x86: 5-15 ms / frame
(100+ FPS). Runs without pytest-benchmark — uses time.perf_counter +
mean over iterations. The per-frame ceiling is relaxed on shared CI
runners (env ``CI``), whose timing is too noisy for a tight absolute
bound; it stays strict locally so genuine regressions still surface.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import supervision as sv

from onnxtools.tracking import create_tracker


def _persistent_dets(rng: np.random.Generator, base: np.ndarray) -> sv.Detections:
    """Jittered version of a fixed base — simulates ~200 persistent targets."""
    jitter = rng.normal(0, 2, base.shape).astype(np.float32)
    xyxy = (base + jitter).astype(np.float32)
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        confidence=rng.uniform(0.7, 0.95, size=n).astype(np.float32),
        class_id=np.zeros(n, dtype=int),
    )


def _make_base(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    tl = np.column_stack([rng.uniform(0, 1600, n), rng.uniform(0, 900, n)]).astype(np.float32)
    wh = np.column_stack([rng.uniform(40, 120, n), rng.uniform(40, 120, n)]).astype(np.float32)
    return np.concatenate([tl, tl + wh], axis=1)


@pytest.mark.parametrize("algo", ["bytetrack_native", "ocsort", "botsort"])
def test_tracker_200_dets_perf(algo):
    tracker = create_tracker(algo)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    base = _make_base(200)
    rng = np.random.default_rng(123)

    # Warm up so the tracked pool stabilises around 200.
    for _ in range(15):
        tracker.update(_persistent_dets(rng, base), frame)

    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        tracker.update(_persistent_dets(rng, base), frame)
    mean_ms = (time.perf_counter() - start) / iters * 1e3

    # Locally ~16ms (still > 60 FPS) catches regressions tightly. Shared CI
    # runners are too noisy for that bound, so relax it there — a coarse 40ms
    # cap still flags order-of-magnitude regressions.
    ceiling = 40.0 if os.environ.get("CI") else 16.0
    assert mean_ms < ceiling, f"{algo}: {mean_ms:.2f}ms over {ceiling:.0f}ms ceiling"
