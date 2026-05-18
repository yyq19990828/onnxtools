"""Archived: BoxMOT adapter for ``onnxtools.tracking``.

Not imported by the live package. Kept for reference — see the directory
README for why it was removed and how to revive it.

Quick recap: BoxMOT 18+ hard-depends on torch/torchvision/filterpy/lapx/etc.
and its ``boxmot.trackers.__init__`` eagerly imports every tracker class
(including ReID-based ones), so a bare ``import boxmot`` pulls in the full
PyTorch stack. This file used to live in ``onnxtools/tracking/__init__.py``
and was wired through ``create_tracker('boxmot-ocsort')`` etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import supervision as sv

from onnxtools.tracking import BaseTracker

logger = logging.getLogger(__name__)


_BOXMOT_ALGOS: Dict[str, str] = {
    # public_name -> dotted import path
    "boxmot-bytetrack": "boxmot.trackers.bytetrack.bytetrack.ByteTrack",
    "boxmot-ocsort": "boxmot.trackers.ocsort.ocsort.OcSort",
    "boxmot-botsort": "boxmot.trackers.botsort.botsort.BotSort",
}


def _import_boxmot_class(algo: str):
    """Import a BoxMOT tracker class lazily; raise helpful error if missing."""
    from importlib import import_module

    path = _BOXMOT_ALGOS[algo]
    module_path, class_name = path.rsplit(".", 1)
    try:
        module = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Tracker '{algo}' requires BoxMOT. Install with `pip install boxmot` "
            f"and accept the PyTorch dependency."
        ) from e
    return getattr(module, class_name)


_COMMON_PARAM_NAMES = {
    "track_activation_threshold",
    "lost_track_buffer",
    "minimum_matching_threshold",
    "frame_rate",
}


_BOXMOT_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "boxmot-bytetrack": {
        "track_activation_threshold": "track_thresh",
        "lost_track_buffer": "track_buffer",
        "minimum_matching_threshold": "match_thresh",
        "frame_rate": "frame_rate",
    },
    "boxmot-ocsort": {
        # OC-SORT only honours frame_rate; activation/match thresholds drop.
        "frame_rate": "frame_rate",
    },
    "boxmot-botsort": {
        "track_activation_threshold": "new_track_thresh",
        "lost_track_buffer": "track_buffer",
        "minimum_matching_threshold": "match_thresh",
        "frame_rate": "frame_rate",
    },
}


def _translate_kwargs(algo: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    mapping = _BOXMOT_PARAM_MAP.get(algo, {})
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in mapping:
            out[mapping[key]] = value
        elif key in _COMMON_PARAM_NAMES and key not in mapping:
            logger.debug("Tracker '%s' ignores unified param '%s'", algo, key)
        else:
            out[key] = value
    return out


class BoxMOTTracker(BaseTracker):
    """Wraps a BoxMOT tracker as a ``BaseTracker``.

    Input: ``(N, 6)`` ``[x1,y1,x2,y2,conf,cls]``.
    Output (BoxMOT): ``(M, 8)`` ``[x1,y1,x2,y2,id,conf,cls,det_ind]``.
    ``det_ind`` is the index into the input dets — used to preserve any
    per-detection metadata when re-wrapping back into ``sv.Detections``.
    """

    def __init__(self, algo: str, **kwargs: Any):
        if algo not in _BOXMOT_ALGOS:
            raise ValueError(
                f"Unknown BoxMOT algorithm: {algo!r}. "
                f"Supported: {sorted(_BOXMOT_ALGOS)}"
            )
        self.name = algo
        self._algo = algo
        self._cls = _import_boxmot_class(algo)
        self._kwargs = _translate_kwargs(algo, dict(kwargs))
        if algo == "boxmot-botsort" and "with_reid" not in self._kwargs:
            # Default to motion-only — no ReID model required.
            self._kwargs["with_reid"] = False
        self._tracker = self._cls(**self._kwargs)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        n = len(detections)
        if n == 0:
            self._tracker.update(np.zeros((0, 6), dtype=np.float32), frame)
            return sv.Detections.empty()

        conf = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(n, dtype=np.float32)
        )
        cls = (
            detections.class_id
            if detections.class_id is not None
            else np.zeros(n, dtype=np.int64)
        )
        dets = np.concatenate(
            [
                detections.xyxy.astype(np.float32),
                conf.reshape(-1, 1).astype(np.float32),
                cls.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )

        result = np.asarray(self._tracker.update(dets, frame))
        if result.size == 0:
            return sv.Detections.empty()

        det_idx = result[:, 7].astype(int)
        return sv.Detections(
            xyxy=result[:, :4].astype(np.float32),
            confidence=result[:, 5].astype(np.float32),
            class_id=result[:, 6].astype(int),
            tracker_id=result[:, 4].astype(int),
            data={
                key: (value[det_idx] if hasattr(value, "__getitem__") else value)
                for key, value in detections.data.items()
                if value is not None
            }
            if detections.data
            else {},
        )

    def reset(self) -> None:
        try:
            module_path = self._cls.__module__.rsplit(".", 1)[0]
            base = __import__(f"{module_path}.basetrack", fromlist=["BaseTrack"])
            if hasattr(base, "BaseTrack") and hasattr(base.BaseTrack, "clear_count"):
                base.BaseTrack.clear_count()
        except (ImportError, AttributeError):
            pass
        self._tracker = self._cls(**self._kwargs)
