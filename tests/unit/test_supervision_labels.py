"""Unit tests for onnxtools/utils/supervision_labels.py

Tests label generation functions:
- create_confidence_labels
- create_ocr_labels
"""

import numpy as np

from onnxtools.utils.supervision_labels import create_confidence_labels, create_ocr_labels


class TestCreateConfidenceLabels:
    """Test create_confidence_labels function."""

    def test_basic(self):
        scores = np.array([0.95, 0.87, 0.5])
        labels = create_confidence_labels(scores)
        assert len(labels) == 3
        assert labels[0] == "0.95"
        assert labels[1] == "0.87"
        assert labels[2] == "0.50"

    def test_empty(self):
        scores = np.array([])
        labels = create_confidence_labels(scores)
        assert labels == []

    def test_single(self):
        scores = np.array([0.123])
        labels = create_confidence_labels(scores)
        assert labels == ["0.12"]

    def test_format_precision(self):
        scores = np.array([0.999, 0.001])
        labels = create_confidence_labels(scores)
        assert labels[0] == "1.00"
        assert labels[1] == "0.00"


class TestCreateOcrLabels:
    """Test create_ocr_labels function."""

    def test_basic_detection_labels(self):
        boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
        scores = np.array([0.95, 0.87])
        class_ids = np.array([0, 1])
        class_names = {0: "car", 1: "truck"}
        plate_results = [None, None]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert len(labels) == 2
        assert "car 0.95" in labels[0]
        assert "truck 0.87" in labels[1]

    def test_plate_with_ocr_info(self):
        boxes = np.array([[0, 0, 100, 50]], dtype=np.float32)
        scores = np.array([0.9])
        class_ids = np.array([13])  # plate class
        class_names = {13: "plate"}
        plate_results = [
            {
                "should_display_ocr": True,
                "plate_text": "京A12345",
                "color": "blue",
                "layer": "single",
            }
        ]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert len(labels) == 1
        assert "京A12345" in labels[0]
        assert "blue" in labels[0]

    def test_plate_without_ocr_display(self):
        boxes = np.array([[0, 0, 100, 50]], dtype=np.float32)
        scores = np.array([0.9])
        class_ids = np.array([13])
        class_names = {13: "plate"}
        plate_results = [{"should_display_ocr": False}]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert "plate 0.90" in labels[0]
        # Should NOT contain OCR text
        assert "\n" not in labels[0]

    def test_class_names_as_list(self):
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        scores = np.array([0.9])
        class_ids = np.array([0])
        class_names = ["car", "truck"]
        plate_results = [None]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert "car 0.90" in labels[0]

    def test_unknown_class_id(self):
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        scores = np.array([0.9])
        class_ids = np.array([99])
        class_names = {0: "car"}
        plate_results = [None]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert "unknown_99" in labels[0]

    def test_empty_detections(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.array([])
        class_ids = np.array([], dtype=np.int32)
        class_names = {0: "car"}
        plate_results = []

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert labels == []

    def test_plate_with_empty_text(self):
        boxes = np.array([[0, 0, 100, 50]], dtype=np.float32)
        scores = np.array([0.9])
        class_ids = np.array([13])
        class_names = {13: "plate"}
        plate_results = [
            {
                "should_display_ocr": True,
                "plate_text": "",
                "color": "blue",
                "layer": "single",
            }
        ]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        # Empty plate_text -> just base label
        assert labels[0] == "plate 0.90"

    def test_mixed_detections(self):
        """Mix of plate and non-plate detections."""
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 250],
                [400, 400, 500, 500],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.85, 0.7])
        class_ids = np.array([0, 13, 1])
        class_names = {0: "car", 1: "truck", 13: "plate"}
        plate_results = [
            None,
            {"should_display_ocr": True, "plate_text": "川B99999", "color": "yellow", "layer": "single"},
            None,
        ]

        labels = create_ocr_labels(boxes, scores, class_ids, plate_results, class_names)
        assert len(labels) == 3
        assert "car" in labels[0]
        assert "川B99999" in labels[1]
        assert "truck" in labels[2]
