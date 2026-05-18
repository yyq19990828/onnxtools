"""Shared primitives for 2D tracker implementations.

Provides :class:`TrackState` (an integer enum used by both ByteTrack and
OC-SORT internals) and :class:`TrackRecord` (a lightweight descriptor for
external introspection — algorithm internals keep their own richer state).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class TrackRecord:
    """Lightweight, framework-agnostic snapshot of a single track.

    Not used by the algorithms internally — emitted only when callers want a
    structured view that does not depend on ``supervision``.
    """

    track_id: int
    xyxy: np.ndarray
    score: float
    class_id: int
    age: int
    hit_streak: int
    state: TrackState
