"""Unit tests for onnxtools/config.py configuration module.

Tests configuration constants and config loading functions:
- DET_CLASSES, COCO_CLASSES, OCR_CHARACTER_DICT, COLOR_MAP, LAYER_MAP
- load_det_config(), load_plate_config(), get_ocr_character_list()
- load_visualization_config()
"""

from pathlib import Path

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

    def test_det_classes_is_dict(self) -> None:
        assert isinstance(DET_CLASSES, dict)
        assert len(DET_CLASSES) == 15

    def test_det_classes_keys_are_sequential(self) -> None:
        assert set(DET_CLASSES.keys()) == set(range(15))

    def test_det_classes_values_are_strings(self) -> None:
        for v in DET_CLASSES.values():
            assert isinstance(v, str) and len(v) > 0

    def test_det_colors_length_matches(self) -> None:
        assert len(DET_COLORS) >= len(DET_CLASSES)

    def test_color_map_completeness(self) -> None:
        assert isinstance(COLOR_MAP, dict)
        assert len(COLOR_MAP) == 5
        assert set(COLOR_MAP.keys()) == {0, 1, 2, 3, 4}

    def test_layer_map_completeness(self) -> None:
        assert isinstance(LAYER_MAP, dict)
        assert len(LAYER_MAP) == 2
        assert set(LAYER_MAP.values()) == {"single", "double"}

    def test_ocr_character_dict_non_empty(self) -> None:
        assert isinstance(OCR_CHARACTER_DICT, list)
        assert len(OCR_CHARACTER_DICT) > 0
        # Should contain digits and letters
        assert "0" in OCR_CHARACTER_DICT
        assert "A" in OCR_CHARACTER_DICT
        # Should contain Chinese province abbreviations
        assert "京" in OCR_CHARACTER_DICT
        assert "川" in OCR_CHARACTER_DICT

    def test_vehicle_type_map(self) -> None:
        assert isinstance(VEHICLE_TYPE_MAP, dict)
        assert len(VEHICLE_TYPE_MAP) == 13
        assert "car" in VEHICLE_TYPE_MAP.values()

    def test_vehicle_color_map(self) -> None:
        assert isinstance(VEHICLE_COLOR_MAP, dict)
        assert len(VEHICLE_COLOR_MAP) == 11
        assert "white" in VEHICLE_COLOR_MAP.values()

    def test_visualization_presets_keys(self) -> None:
        expected = {"box_only", "standard", "lightweight", "privacy", "debug", "high_contrast"}
        assert set(VISUALIZATION_PRESETS.keys()) == expected

    def test_visualization_presets_have_annotators(self) -> None:
        for name, preset in VISUALIZATION_PRESETS.items():
            assert "annotators" in preset, f"Preset '{name}' missing annotators"
            assert isinstance(preset["annotators"], list)
            assert len(preset["annotators"]) > 0


class TestLoadDetConfig:
    """Test load_det_config function."""

    def test_default_config(self) -> None:
        config = load_det_config()
        assert "class_names" in config
        assert "visual_colors" in config
        assert config["class_names"] == DET_CLASSES
        assert config["visual_colors"] == DET_COLORS

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "det_config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "class_names": {0: "cat", 1: "dog"},
                    "visual_colors": ["#FF0000", "#00FF00"],
                }
            ),
            encoding="utf-8",
        )
        config = load_det_config(str(config_file))
        assert config["class_names"] == {0: "cat", 1: "dog"}

    def test_load_from_yaml_list_format(self, tmp_path: Path) -> None:
        """class_names as list should be converted to dict."""
        config_file = tmp_path / "det_config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "class_names": ["cat", "dog"],
                    "visual_colors": ["#FF0000", "#00FF00"],
                }
            ),
            encoding="utf-8",
        )
        config = load_det_config(str(config_file))
        assert config["class_names"] == {0: "cat", 1: "dog"}

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_det_config("/nonexistent/path.yaml")


class TestLoadPlateConfig:
    """Test load_plate_config function."""

    def test_default_config(self) -> None:
        config = load_plate_config()
        assert "ocr_dict" in config
        assert "color_dict" in config
        assert "layer_dict" in config
        assert config["ocr_dict"] == OCR_CHARACTER_DICT
        assert config["color_dict"] == COLOR_MAP
        assert config["layer_dict"] == LAYER_MAP

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "plate.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "ocr_dict": ["A", "B"],
                    "color_dict": {0: "red"},
                    "layer_dict": {0: "single"},
                }
            ),
            encoding="utf-8",
        )
        config = load_plate_config(str(config_file))
        assert config["ocr_dict"] == ["A", "B"]

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_plate_config("/nonexistent/path.yaml")


class TestGetOcrCharacterList:
    """Test get_ocr_character_list function."""

    def test_default_with_blank_and_space(self) -> None:
        chars = get_ocr_character_list()
        assert chars[0] == "blank"
        assert chars[-1] == " "
        assert len(chars) == len(OCR_CHARACTER_DICT) + 2  # +blank +space

    def test_without_blank(self) -> None:
        chars = get_ocr_character_list(add_blank=False)
        assert chars[0] != "blank"
        assert len(chars) == len(OCR_CHARACTER_DICT) + 1  # +space only

    def test_without_space(self) -> None:
        chars = get_ocr_character_list(add_space=False)
        assert chars[-1] != " "
        assert len(chars) == len(OCR_CHARACTER_DICT) + 1  # +blank only

    def test_without_blank_and_space(self) -> None:
        chars = get_ocr_character_list(add_blank=False, add_space=False)
        assert len(chars) == len(OCR_CHARACTER_DICT)


class TestLoadVisualizationConfig:
    """Test load_visualization_config function."""

    def test_default_config(self) -> None:
        config = load_visualization_config()
        assert "presets" in config
        assert "standard" in config["presets"]

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "vis_presets.yaml"
        config_file.write_text(
            yaml.dump({"presets": {"custom": {"annotators": [{"type": "box"}]}}}),
            encoding="utf-8",
        )
        config = load_visualization_config(str(config_file))
        assert "custom" in config["presets"]

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_visualization_config("/nonexistent/path.yaml")
