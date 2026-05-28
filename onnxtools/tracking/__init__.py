"""2D Multi-Object Tracking — unified interface over supervision and
hand-rolled vectorised back-ends.

All trackers accept and emit ``supervision.Detections``, so they are drop-in
replacements for each other at any call site that already speaks supervision.

Quick start
-----------

    from onnxtools.tracking import create_tracker

    tracker = create_tracker('bytetrack')         # supervision built-in (default)
    tracker = create_tracker('bytetrack_native')  # native vectorised ByteTrack
    tracker = create_tracker('ocsort')            # native OC-SORT
    tracked = tracker.update(detections, frame)   # adds .tracker_id
    tracker.reset()                                # restart IDs from 1

The native back-ends (``bytetrack_native`` and ``ocsort``) are hand-written on
top of numpy with optional ``lap.lapjv`` acceleration (``[tracking-fast]``
extra). They share the kalman + matching primitives in :mod:`.kalman` and
:mod:`.matching`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTracker(ABC):
    """Common interface for all 2D trackers.

    Implementations must:
      * Accept ``sv.Detections`` (possibly empty) and the current frame.
      * Return a ``sv.Detections`` whose ``tracker_id`` attribute is populated
        for every emitted detection (length may be < input length if the
        tracker dropped low-confidence detections).
      * Tick state on every call — including frames with zero detections — so
        the lost-track buffer ages correctly.
    """

    name: str = "base"

    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections: ...

    @abstractmethod
    def reset(self) -> None:
        """Restart tracker state so IDs begin at 1 again."""


# ---------------------------------------------------------------------------
# supervision.ByteTrack adapter (no extra dependencies)
# ---------------------------------------------------------------------------


class SupervisionByteTrack(BaseTracker):
    """Thin wrapper around ``sv.ByteTrack``.

    Param mapping (kept identical to supervision so existing callers keep
    working):

        track_activation_threshold: float = 0.25
        lost_track_buffer:          int   = 30
        minimum_matching_threshold: float = 0.8
        frame_rate:                 int   = 30
    """

    name = "bytetrack"

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        **_: Any,  # absorb unknown kwargs so factory mapping is permissive
    ):
        self._kwargs = dict(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._tracker = sv.ByteTrack(**self._kwargs)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        # supervision ignores the frame argument
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        self._tracker = sv.ByteTrack(**self._kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


SUPPORTED_TRACKERS = ("bytetrack", "bytetrack_native", "ocsort")


def create_tracker(algo: str = "bytetrack", **kwargs: Any) -> BaseTracker:
    """Create a tracker by name.

    Args:
        algo: One of ``SUPPORTED_TRACKERS``.

            * ``"bytetrack"`` — supervision built-in (default; back-compat).
            * ``"bytetrack_native"`` — hand-rolled vectorised ByteTrack.
            * ``"ocsort"`` — hand-rolled vectorised OC-SORT.
        **kwargs: Tracker parameters. The supervision-style aliases
            (``track_activation_threshold`` / ``lost_track_buffer`` /
            ``minimum_matching_threshold`` / ``frame_rate``) are accepted by
            all back-ends so a single shared kwargs dict can be forwarded.

    Returns:
        A :class:`BaseTracker` instance.

    Raises:
        ValueError: If ``algo`` is unknown.
    """
    if algo == "bytetrack":
        return SupervisionByteTrack(**kwargs)
    if algo == "bytetrack_native":
        # Delayed import — scipy is only required for native back-ends.
        from .bytetrack import ByteTrackNative

        return ByteTrackNative(**kwargs)
    if algo == "ocsort":
        from .ocsort import OCSORT

        return OCSORT(**kwargs)
    raise ValueError(f"Unknown tracker algorithm: {algo!r}. Supported: {SUPPORTED_TRACKERS}")


__all__ = [
    "BaseTracker",
    "SupervisionByteTrack",
    "create_tracker",
    "SUPPORTED_TRACKERS",
]
