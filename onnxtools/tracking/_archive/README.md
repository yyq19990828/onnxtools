# `tracking/_archive`

Reference implementations of tracker adapters that were prototyped and then
removed from the live package. Kept here so future work doesn't have to
re-derive the API translation; nothing in this directory is imported at
runtime.

## `boxmot_adapter.py`

A `BoxMOTTracker` adapter for the [BoxMOT](https://github.com/mikel-brostrom/boxmot)
multi-object-tracking library, wrapping ByteTrack / OC-SORT / BoT-SORT behind
the same `BaseTracker` interface used by `SupervisionByteTrack`.

**Why removed**: BoxMOT 18+ has hard runtime dependencies on `torch`,
`torchvision`, `filterpy`, `lapx`, `pandas`, `scikit-learn` and others.
`boxmot/trackers/__init__.py` eagerly imports every tracker class including
the ReID-based DeepOCSORT/StrongSORT, so even using only the pure-motion
trackers triggers the full PyTorch import (+2 GB env, +1-2 s startup). That
violates the repository's "pure ONNX, no torch" invariant.

**When to revive**: pick one of these paths instead of re-introducing BoxMOT
as-is:

1. **Port OC-SORT directly** (~500 lines pure numpy + a small Kalman filter).
   The OC-SORT paper's reference code has no torch dependency at all. This
   gives a quality bump over ByteTrack without any new heavy deps.
2. **Vendor BoxMOT lazily**: put `import boxmot` inside the adapter method
   bodies (not module top-level) *and* monkey-patch `sys.modules` to stub
   out the ReID-tracker submodules before `boxmot.trackers.__init__` runs.
   Fragile; pinned to a specific BoxMOT version.
3. **Vendor only the algorithms we need**: copy `bytetrack.py` + `ocsort.py`
   from BoxMOT into this repo with their MIT-licence headers, prune
   `boxmot/trackers/__init__.py` of the torch-dependent imports. License-clean
   but ongoing maintenance burden.

For now, `onnxtools.tracking.create_tracker` only ships the supervision
ByteTrack adapter; the abstraction (`BaseTracker`) is kept so adding a new
backend later is a localised change.
