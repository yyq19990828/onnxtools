"""Unit tests for onnxtools/config.py configuration module.

Tests configuration constants and config loading functions:
- DET_CLASSES, COCO_CLASSES, OCR_CHARACTER_DICT, COLOR_MAP, LAYER_MAP
- load_det_config(), load_plate_config(), get_ocr_character_list()
- load_visualization_config()
"""

import tempfile

import pytest
import yaml

from onnxtools.config import (
    COLOR_MAP,
    DET_CLASSES,
    DET_COLORS,
    LAYER_MAP,
    OCR_CHARACTER_DICT,
    VEHICLE_COLOR_MAP,
    VEHICLE_TYPE_MAP,
    VISUALIZATION_PRESETS,
    get_ocr_character_list,
    load_det_config,
    load_plate_config,
    load_visualization_config,
)


class TestConfigConstants:
    """Test configuration constants are well-formed."""

    def test_det_classes_is_dict(self):
        assert isinstance(DET_CLASSES, dict)
        assert len(DET_CLASSES) == 15

    def test_det_classes_keys_are_sequential(self):
        assert set(DET_CLASSES.keys()) == set(range(15))

    def test_det_classes_values_are_strings(self):
        for v in DET_CLASSES.values():
            assert isinstance(v, str) and len(v) > 0

    def test_det_colors_length_matches(self):
        assert len(DET_COLORS) >= len(DET_CLASSES)

    def test_color_map_completeness(self):
        assert isinstance(COLOR_MAP, dict)
        assert len(COLOR_MAP) == 5
        assert set(COLOR_MAP.keys()) == {0, 1, 2, 3, 4}

    def test_layer_map_completeness(self):
        assert isinstance(LAYER_MAP, dict)
        assert len(LAYER_MAP) == 2
        assert set(LAYER_MAP.values()) == {"single", "double"}

    def test_ocr_character_dict_non_empty(self):
        assert isinstance(OCR_CHARACTER_DICT, list)
        assert len(OCR_CHARACTER_DICT) > 0
        # Should contain digits and letters
        assert "0" in OCR_CHARACTER_DICT
        assert "A" in OCR_CHARACTER_DICT
        # Should contain Chinese province abbreviations
        assert "京" in OCR_CHARACTER_DICT
        assert "川" in OCR_CHARACTER_DICT

    def test_vehicle_type_map(self):
        assert isinstance(VEHICLE_TYPE_MAP, dict)
        assert len(VEHICLE_TYPE_MAP) == 13
        assert "car" in VEHICLE_TYPE_MAP.values()

    def test_vehicle_color_map(self):
        assert isinstance(VEHICLE_COLOR_MAP, dict)
        assert len(VEHICLE_COLOR_MAP) == 11
        assert "white" in VEHICLE_COLOR_MAP.values()

    def test_visualization_presets_keys(self):
        expected = {"box_only", "standard", "lightweight", "privacy", "debug", "high_contrast"}
        assert set(VISUALIZATION_PRESETS.keys()) == expected

    def test_visualization_presets_have_annotators(self):
        for name, preset in VISUALIZATION_PRESETS.items():
            assert "annotators" in preset, f"Preset '{name}' missing annotators"
            assert isinstance(preset["annotators"], list)
            assert len(preset["annotators"]) > 0


class TestLoadDetConfig:
    """Test load_det_config function."""

    def test_default_config(self):
        config = load_det_config()
        assert "class_names" in config
        assert "visual_colors" in config
        assert config["class_names"] == DET_CLASSES
        assert config["visual_colors"] == DET_COLORS

    def test_load_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(
                {
                    "class_names": {0: "cat", 1: "dog"},
                    "visual_colors": ["#FF0000", "#00FF00"],
                },
                f,
            )
            f.flush()
            config = load_det_config(f.name)
            assert config["class_names"] == {0: "cat", 1: "dog"}

    def test_load_from_yaml_list_format(self):
        """class_names as list should be converted to dict."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(
                {
                    "class_names": ["cat", "dog"],
                    "visual_colors": ["#FF0000", "#00FF00"],
                },
                f,
            )
            f.flush()
            config = load_det_config(f.name)
            assert config["class_names"] == {0: "cat", 1: "dog"}

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_det_config("/nonexistent/path.yaml")


class TestLoadPlateConfig:
    """Test load_plate_config function."""

    def test_default_config(self):
        config = load_plate_config()
        assert "ocr_dict" in config
        assert "color_dict" in config
        assert "layer_dict" in config
        assert config["ocr_dict"] == OCR_CHARACTER_DICT
        assert config["color_dict"] == COLOR_MAP
        assert config["layer_dict"] == LAYER_MAP

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(
                {
                    "ocr_dict": ["A", "B"],
                    "color_dict": {0: "red"},
                    "layer_dict": {0: "single"},
                },
                f,
            )
            f.flush()
            config = load_plate_config(f.name)
            assert config["ocr_dict"] == ["A", "B"]

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_plate_config("/nonexistent/path.yaml")


class TestGetOcrCharacterList:
    """Test get_ocr_character_list function."""

    def test_default_with_blank_and_space(self):
        chars = get_ocr_character_list()
        assert chars[0] == "blank"
        assert chars[-1] == " "
        assert len(chars) == len(OCR_CHARACTER_DICT) + 2  # +blank +space

    def test_without_blank(self):
        chars = get_ocr_character_list(add_blank=False)
        assert chars[0] != "blank"
        assert len(chars) == len(OCR_CHARACTER_DICT) + 1  # +space only

    def test_without_space(self):
        chars = get_ocr_character_list(add_space=False)
        assert chars[-1] != " "
        assert len(chars) == len(OCR_CHARACTER_DICT) + 1  # +blank only

    def test_without_blank_and_space(self):
        chars = get_ocr_character_list(add_blank=False, add_space=False)
        assert len(chars) == len(OCR_CHARACTER_DICT)


class TestLoadVisualizationConfig:
    """Test load_visualization_config function."""

    def test_default_config(self):
        config = load_visualization_config()
        assert "presets" in config
        assert "standard" in config["presets"]

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump({"presets": {"custom": {"annotators": [{"type": "box"}]}}}, f)
            f.flush()
            config = load_visualization_config(f.name)
            assert "custom" in config["presets"]

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_visualization_config("/nonexistent/path.yaml")
