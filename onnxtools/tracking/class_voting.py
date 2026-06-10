"""Class-label voting wrapper for trackers.

The detector may classify the same physical object as different classes
on different frames (e.g. a partially-occluded car flipping between
``car`` and ``truck``). The tracker_id is stable but ``class_id`` jumps
around, which shows up as label flicker in the visualisation and as
churn in any per-track class statistics.

:class:`ClassVotingTracker` wraps any :class:`BaseTracker` and replaces
each emitted ``class_id`` with the confidence-weighted majority over all
past observations of that ``tracker_id``. Confidence-weighted is more
robust than count-only majority because early low-score classifications
(often the wrong one) contribute less than mature high-score ones.

Memory note: the vote table grows with the number of distinct tracker
IDs ever seen. For long-running streams pass ``max_table_size`` to LRU-
evict the oldest entries — voting state for genuinely dead IDs is
useless and only wastes RAM.

Usage::

    from onnxtools.tracking import create_tracker
    from onnxtools.tracking.class_voting import ClassVotingTracker

    inner = create_tracker("bytetrack_native", ...)
    tracker = ClassVotingTracker(inner)
    tracked = tracker.update(detections, frame)  # class_id is voted
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import supervision as sv

from . import BaseTracker


class ClassVotingTracker(BaseTracker):
    """Wrap any tracker and stabilise its ``class_id`` via weighted voting.

    Args:
        inner: The underlying tracker. ``update`` / ``reset`` are delegated.
        decay: Exponential decay applied to all stored votes every frame.
            ``1.0`` accumulates forever; ``< 1.0`` lets recent frames
            dominate. ``0.99`` ≈ "last 100 frames matter most" half-life
            of ~69 frames; at 3Hz that's ~23 seconds.
        max_table_size: LRU cap on number of tracker IDs whose vote tables
            are kept. ``0`` disables the cap.
    """

    name = "class_voting"

    def __init__(
        self,
        inner: BaseTracker,
        decay: float = 1.0,
        max_table_size: int = 4096,
    ):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay!r}")
        self.inner = inner
        self.decay = float(decay)
        self.max_table_size = int(max_table_size)
        # tracker_id -> {class_id: weighted_count}. OrderedDict for LRU.
        self._votes: OrderedDict[int, dict[int, float]] = OrderedDict()

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        out = self.inner.update(detections, frame)
        if getattr(out, "tracker_id", None) is None or len(out) == 0 or getattr(out, "class_id", None) is None:
            return out

        # 1. Decay all existing votes (skipped when decay=1.0 for speed).
        if self.decay < 1.0:
            for table in self._votes.values():
                for cid in table:
                    table[cid] *= self.decay

        # 2. Add this frame's observations.
        scores = (
            np.asarray(out.confidence, dtype=np.float32)
            if getattr(out, "confidence", None) is not None
            else np.ones(len(out), dtype=np.float32)
        )
        for i, tid in enumerate(out.tracker_id):
            tid = int(tid)
            cid = int(out.class_id[i])
            w = float(scores[i])
            if tid in self._votes:
                # Refresh LRU position.
                self._votes.move_to_end(tid)
                table = self._votes[tid]
            else:
                table = {}
                self._votes[tid] = table
            table[cid] = table.get(cid, 0.0) + w

        # 3. LRU evict if over budget.
        if self.max_table_size > 0:
            while len(self._votes) > self.max_table_size:
                self._votes.popitem(last=False)

        # 4. Rewrite class_id with the winning vote per tid.
        voted = np.empty(len(out), dtype=int)
        for i, tid in enumerate(out.tracker_id):
            table = self._votes[int(tid)]
            # Tie-break on insertion order (Python dict preserves it),
            # which favours the first class ever seen for this track —
            # mildly more stable than argmax over equal weights.
            voted[i] = max(table.items(), key=lambda kv: kv[1])[0]
        out.class_id = voted
        return out

    def reset(self) -> None:
        self.inner.reset()
        self._votes.clear()


__all__ = ["ClassVotingTracker"]
