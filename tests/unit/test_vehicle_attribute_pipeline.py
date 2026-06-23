"""Unit tests for VehicleAttributePipeline (detection → motor-vehicle ROI → attributes).

Uses mock detector / classifier injection so no real ONNX model or GPU is needed.
"""

import numpy as np
import pytest

from onnxtools.infer_onnx.onnx_cls import ClsResult
from onnxtools.pipeline import VehicleAttributePipeline

# rtdetr-2024080100 class layout (onnxtools.config.DET_CLASSES)
DET_CLASS_NAMES = [
    "car",
    "truck",
    "heavy_truck",
    "van",
    "bus",
    "bicycle",
    "cyclist",
    "tricycle",
    "trolley",
    "pedestrian",
    "cone",
    "animal",
    "other",
    "plate",
    "motorcycle",
]


class _FakeResult:
    """Minimal stand-in for onnxtools Result with the fields the pipeline reads."""

    def __init__(self, boxes, scores, class_ids):
        self.boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.array(scores, dtype=np.float32)
        self.class_ids = np.array(class_ids, dtype=np.int64)

    def __len__(self):
        return len(self.boxes)


class _FakeDetector:
    class_names = None

    def __init__(self, result):
        self._result = result

    def __call__(self, frame):
        return self._result


class _FakeVA:
    """Records every crop it is called on; returns a fixed dual-branch ClsResult."""

    def __init__(self, labels=("school_bus", "blue"), confs=(0.93, 0.88)):
        self._labels = list(labels)
        self._confs = list(confs)
        self.calls = []

    def __call__(self, crop):
        self.calls.append(crop)
        return ClsResult(labels=self._labels, confidences=self._confs, avg_confidence=sum(self._confs) / 2)


def _make_pipeline(result, va=None, class_names=None):
    """Build a pipeline with injected mocks, bypassing __init__ (no model loading)."""
    p = VehicleAttributePipeline.__new__(VehicleAttributePipeline)
    p.detector = _FakeDetector(result)
    p.va_classifier = va if va is not None else _FakeVA()
    p.class_names = class_names if class_names is not None else DET_CLASS_NAMES
    p.roi_pad_ratio = 0.1
    return p


class TestVehicleAttributePipelineCall:
    """Tests for VehicleAttributePipeline.__call__ orchestration."""

    def test_motor_vehicle_gets_attributes(self):
        """A car detection should carry vehicle_type + color attributes."""
        result = _FakeResult(boxes=[[10, 10, 110, 110]], scores=[0.97], class_ids=[0])
        pipeline = _make_pipeline(result)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        out = pipeline(frame)

        assert len(out) == 1
        item = out[0]
        assert item["type"] == "car"
        assert item["box2d"] == [10.0, 10.0, 110.0, 110.0]
        assert item["score"] == pytest.approx(0.97)
        assert item["vehicle_type"] == "school_bus"
        assert item["vehicle_type_conf"] == pytest.approx(0.93)
        assert item["color"] == "blue"
        assert item["color_conf"] == pytest.approx(0.88)

    def test_non_motor_vehicle_no_attributes(self):
        """Pedestrian / plate boxes get geometry only — classifier is never called."""
        result = _FakeResult(
            boxes=[[0, 0, 20, 40], [5, 5, 25, 15]], scores=[0.8, 0.9], class_ids=[9, 13]
        )  # pedestrian, plate
        va = _FakeVA()
        pipeline = _make_pipeline(result, va=va)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        out = pipeline(frame)

        assert [d["type"] for d in out] == ["pedestrian", "plate"]
        assert all("vehicle_type" not in d for d in out)
        assert va.calls == []

    def test_va_called_only_for_motor_vehicles(self):
        """Mixed frame: classifier runs once per motor vehicle, not per detection."""
        result = _FakeResult(
            boxes=[[0, 0, 30, 30], [40, 40, 70, 70], [10, 10, 20, 20]],
            scores=[0.9, 0.9, 0.9],
            class_ids=[0, 9, 14],  # car, pedestrian, motorcycle
        )
        va = _FakeVA()
        pipeline = _make_pipeline(result, va=va)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        out = pipeline(frame)

        assert len(out) == 3
        assert len(va.calls) == 2  # car + motorcycle only
        assert "vehicle_type" in out[0] and "vehicle_type" not in out[1] and "vehicle_type" in out[2]

    def test_always_has_base_fields(self):
        """Every item has type / box2d / score regardless of class."""
        result = _FakeResult(boxes=[[1, 2, 3, 4]], scores=[0.5], class_ids=[13])
        pipeline = _make_pipeline(result)
        out = pipeline(np.zeros((50, 50, 3), dtype=np.uint8))

        assert set(out[0]) >= {"type", "box2d", "score"}

    def test_empty_detection_returns_empty(self):
        """No detections → empty list."""
        result = _FakeResult(boxes=np.empty((0, 4)), scores=[], class_ids=[])
        pipeline = _make_pipeline(result)
        assert pipeline(np.zeros((50, 50, 3), dtype=np.uint8)) == []

    def test_empty_roi_skips_classification(self):
        """A degenerate box producing an empty crop must not call the classifier."""
        result = _FakeResult(boxes=[[10, 10, 10, 10]], scores=[0.9], class_ids=[0])  # zero-area
        va = _FakeVA()
        pipeline = _make_pipeline(result, va=va)
        out = pipeline(np.zeros((100, 100, 3), dtype=np.uint8))

        assert out[0]["type"] == "car"
        assert "vehicle_type" not in out[0]
        assert va.calls == []


class TestCropRoi:
    """Tests for VehicleAttributePipeline._crop_roi."""

    def test_padding_applied(self):
        """ROI is expanded by roi_pad_ratio on each side."""
        pipeline = _make_pipeline(_FakeResult([[0, 0, 1, 1]], [0.0], [0]))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = pipeline._crop_roi(frame, [10, 10, 30, 30], 100, 100)  # w=h=20, pad=2

        assert roi.shape == (24, 24, 3)  # [8,8] .. [32,32]

    def test_clipped_to_image_bounds(self):
        """Expansion never escapes the image."""
        pipeline = _make_pipeline(_FakeResult([[0, 0, 1, 1]], [0.0], [0]))
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        roi = pipeline._crop_roi(frame, [0, 0, 50, 50], 50, 50)

        assert roi.shape == (50, 50, 3)


class TestResolveClassNames:
    """Tests for VehicleAttributePipeline._resolve_class_names."""

    def test_from_dict(self):
        """Dict config fills gaps with class_<i> placeholders by index."""
        pipeline = _make_pipeline(_FakeResult([[0, 0, 1, 1]], [0.0], [0]))
        names = pipeline._resolve_class_names({0: "car", 2: "bus"})

        assert names == ["car", "class_1", "bus"]

    def test_default_falls_back_to_det_classes(self):
        """None config + no model metadata → built-in DET_CLASSES."""
        pipeline = _make_pipeline(_FakeResult([[0, 0, 1, 1]], [0.0], [0]))
        pipeline.detector.class_names = None
        names = pipeline._resolve_class_names(None)

        assert names[0] == "car"
        assert names[13] == "plate"
        assert names[14] == "motorcycle"


class TestMotorVehicleClasses:
    """Tests for the MOTOR_VEHICLE_CLASSES membership set."""

    def test_expected_members(self):
        """Six motor-vehicle classes, plate / pedestrian excluded."""
        mv = VehicleAttributePipeline.MOTOR_VEHICLE_CLASSES
        assert mv == {"car", "truck", "heavy_truck", "van", "bus", "motorcycle"}
        assert "plate" not in mv
        assert "pedestrian" not in mv
