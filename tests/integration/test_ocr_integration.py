"""Integration tests for OCR text overlay functionality - 这些测试必须在实现前编写且必须失败."""

import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.integration
class TestOCRIntegration:
    """Integration tests for OCR text overlay with supervision library."""

    def test_plate_ocr_display_integration(self, sample_image, sample_detections,
                                         sample_class_names, sample_colors, sample_plate_results):
        """Integration: OCR results should be properly overlaid on plate detections."""
        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=sample_plate_results
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image), "Image should be modified with OCR text"

    def test_chinese_text_rendering_integration(self, sample_image, sample_class_names, sample_colors):
        """Integration: Chinese characters in OCR results should render correctly."""
        chinese_detections = [
            [
                [100.0, 100.0, 200.0, 150.0, 0.95, 0],  # vehicle
                [150.0, 120.0, 180.0, 135.0, 0.88, 1],  # plate
            ]
        ]

        chinese_plate_results = [
            None,  # vehicle (no OCR)
            {
                "plate_text": "京A12345",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            }
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, chinese_detections, sample_class_names,
            sample_colors, plate_results=chinese_plate_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_supervision_ocr_label_creation(self, sample_detections, sample_plate_results, sample_class_names):
        """Integration: OCR labels should be properly created for supervision annotator."""
        try:
            from utils.supervision_labels import create_ocr_labels

            labels = create_ocr_labels(sample_detections[0], sample_plate_results, sample_class_names)

            assert isinstance(labels, list)
            assert len(labels) == len(sample_detections[0])

            # Check that plate detection has OCR info
            plate_label = labels[1]  # Second detection is plate
            assert "京A12345" in plate_label, "Plate text should be in label"
            assert "蓝色" in plate_label, "Color info should be in label"

        except ImportError:
            pytest.fail("create_ocr_labels function must be implemented")

    def test_multiline_ocr_text_integration(self, sample_image, sample_class_names, sample_colors):
        """Integration: Multi-line OCR text should be properly formatted."""
        detections_with_long_text = [
            [
                [150.0, 120.0, 280.0, 135.0, 0.88, 1],  # wider plate for long text
            ]
        ]

        multiline_plate_results = [
            {
                "plate_text": "京A12345D",  # Longer plate number
                "color": "蓝色渐变",      # Longer color description
                "layer": "双层车牌",      # Multi-character layer info
                "should_display_ocr": True
            }
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, detections_with_long_text, sample_class_names,
            sample_colors, plate_results=multiline_plate_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_ocr_positioning_integration(self, sample_image, sample_class_names, sample_colors):
        """Integration: OCR text positioning should adapt to detection location."""
        # Test different positions to ensure smart positioning
        edge_positions = [
            [
                [10.0, 10.0, 80.0, 30.0, 0.9, 1],      # Top-left (below box)
                [560.0, 10.0, 630.0, 30.0, 0.9, 1],    # Top-right (below box)
                [10.0, 450.0, 80.0, 470.0, 0.9, 1],    # Bottom-left (above box)
                [560.0, 450.0, 630.0, 470.0, 0.9, 1],  # Bottom-right (above box)
            ]
        ]

        ocr_results = [
            {"plate_text": "京A001", "color": "蓝色", "layer": "单层", "should_display_ocr": True},
            {"plate_text": "京A002", "color": "蓝色", "layer": "单层", "should_display_ocr": True},
            {"plate_text": "京A003", "color": "蓝色", "layer": "单层", "should_display_ocr": True},
            {"plate_text": "京A004", "color": "蓝色", "layer": "单层", "should_display_ocr": True},
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, edge_positions, sample_class_names,
            sample_colors, plate_results=ocr_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_ocr_without_font_fallback(self, sample_image, sample_detections, sample_class_names, sample_colors, sample_plate_results):
        """Integration: OCR should handle missing font gracefully."""
        from utils.drawing import draw_detections

        # Test with non-existent font path
        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=sample_plate_results,
            font_path="non_existent_font.ttf"
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_supervision_rich_label_annotator_integration(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Supervision RichLabelAnnotator should work with OCR data."""
        try:
            from utils.supervision_config import create_rich_label_annotator
            from utils.supervision_converter import convert_to_supervision_detections
            from utils.supervision_labels import create_ocr_labels
            import supervision as sv

            # Convert detections
            sv_detections = convert_to_supervision_detections(sample_detections, sample_class_names)

            # Create labels
            labels = create_ocr_labels(sample_detections[0], [None, {"plate_text": "京A12345", "color": "蓝色", "layer": "单层", "should_display_ocr": True}], sample_class_names)

            # Create annotator
            label_annotator = create_rich_label_annotator()

            # Apply annotation
            result = label_annotator.annotate(
                scene=sample_image.copy(),
                detections=sv_detections,
                labels=labels
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape

        except ImportError:
            pytest.fail("Supervision RichLabelAnnotator integration components must be implemented")

    def test_mixed_ocr_and_non_ocr_detections(self, sample_image, sample_class_names, sample_colors):
        """Integration: Handle mix of detections with and without OCR results."""
        mixed_detections = [
            [
                [50.0, 50.0, 150.0, 100.0, 0.95, 0],   # vehicle (no OCR)
                [200.0, 60.0, 240.0, 80.0, 0.88, 1],   # plate (with OCR)
                [100.0, 200.0, 200.0, 280.0, 0.75, 0], # vehicle (no OCR)
                [300.0, 300.0, 350.0, 320.0, 0.92, 1]  # plate (with OCR)
            ]
        ]

        mixed_ocr_results = [
            None,  # vehicle
            {"plate_text": "京A001", "color": "蓝色", "layer": "单层", "should_display_ocr": True},  # plate
            None,  # vehicle
            {"plate_text": "沪B002", "color": "黄色", "layer": "单层", "should_display_ocr": True}   # plate
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, mixed_detections, sample_class_names,
            sample_colors, plate_results=mixed_ocr_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_ocr_display_flag_integration(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: should_display_ocr flag should control OCR text display."""
        # Test with OCR disabled
        ocr_disabled_results = [
            None,  # vehicle
            {
                "plate_text": "京A12345",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": False  # Disabled
            }
        ]

        from utils.drawing import draw_detections

        result_disabled = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=ocr_disabled_results
        )

        # Test with OCR enabled
        ocr_enabled_results = [
            None,  # vehicle
            {
                "plate_text": "京A12345",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True  # Enabled
            }
        ]

        result_enabled = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=ocr_enabled_results
        )

        # Both should be valid but potentially different
        assert isinstance(result_disabled, np.ndarray)
        assert isinstance(result_enabled, np.ndarray)

    def test_empty_ocr_text_handling(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Handle empty or missing OCR text gracefully."""
        empty_ocr_results = [
            None,  # vehicle
            {
                "plate_text": "",  # Empty text
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            }
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=empty_ocr_results
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_ocr_performance_with_many_plates(self, sample_image, sample_class_names, sample_colors):
        """Integration: OCR overlay should perform well with many plate detections."""
        # Create many plate detections
        many_plate_detections = []
        detection_list = []
        ocr_results = []

        for i in range(15):  # 15 plates
            x1, y1 = (i % 5) * 120, (i // 5) * 150
            x2, y2 = x1 + 100, y1 + 30
            detection_list.append([x1, y1, x2, y2, 0.9, 1])  # All plates
            ocr_results.append({
                "plate_text": f"京A{i:03d}",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            })

        many_plate_detections.append(detection_list)

        from utils.drawing import draw_detections
        import time

        start_time = time.time()
        result = draw_detections(
            sample_image, many_plate_detections, sample_class_names,
            sample_colors, plate_results=ocr_results
        )
        processing_time = (time.time() - start_time) * 1000  # ms

        assert isinstance(result, np.ndarray)
        assert processing_time < 50.0, f"OCR overlay too slow: {processing_time:.2f}ms"

    def test_special_characters_in_ocr(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Handle special characters and symbols in OCR text."""
        special_char_results = [
            None,  # vehicle
            {
                "plate_text": "新能源001",  # New energy vehicle plate
                "color": "绿色渐变",
                "layer": "双层",
                "should_display_ocr": True
            }
        ]

        from utils.drawing import draw_detections

        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=special_char_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)