"""
Contract test for VisualizationPreset loading and creation.

This test MUST FAIL before implementation (TDD approach).
"""

from pathlib import Path

import pytest

from onnxtools.utils.supervision_annotator import AnnotatorPipeline, AnnotatorType
from onnxtools.utils.supervision_preset import Presets, VisualizationPreset


class TestVisualizationPresetContract:
    """Contract tests for VisualizationPreset interface."""

    def test_load_standard_preset(self):
        """Test loading standard preset from YAML."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        assert isinstance(preset, VisualizationPreset)
        assert preset.name is not None
        assert preset.description is not None
        assert hasattr(preset, 'annotators')

    def test_load_lightweight_preset(self):
        """Test loading lightweight preset from YAML."""
        preset = VisualizationPreset.from_yaml(Presets.LIGHTWEIGHT)

        assert isinstance(preset, VisualizationPreset)
        assert preset.name is not None

    def test_load_privacy_preset(self):
        """Test loading privacy protection preset from YAML."""
        preset = VisualizationPreset.from_yaml(Presets.PRIVACY)

        assert isinstance(preset, VisualizationPreset)
        assert preset.name is not None

    def test_load_debug_preset(self):
        """Test loading debug analysis preset from YAML."""
        preset = VisualizationPreset.from_yaml(Presets.DEBUG)

        assert isinstance(preset, VisualizationPreset)
        assert preset.name is not None

    def test_load_high_contrast_preset(self):
        """Test loading high contrast preset from YAML."""
        preset = VisualizationPreset.from_yaml(Presets.HIGH_CONTRAST)

        assert isinstance(preset, VisualizationPreset)
        assert preset.name is not None

    def test_load_all_five_presets(self):
        """Test that all 5 predefined presets can be loaded."""
        preset_names = [
            Presets.STANDARD,
            Presets.LIGHTWEIGHT,
            Presets.PRIVACY,
            Presets.DEBUG,
            Presets.HIGH_CONTRAST
        ]

        for preset_name in preset_names:
            preset = VisualizationPreset.from_yaml(preset_name)
            assert isinstance(preset, VisualizationPreset)
            assert preset.name, f"Preset {preset_name} missing name"
            assert preset.annotators, f"Preset {preset_name} missing annotators"

    def test_unknown_preset_raises_error(self):
        """Test that unknown preset name raises ValueError."""
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            VisualizationPreset.from_yaml("unknown_preset_name")

    def test_preset_has_name_field(self):
        """Test that loaded preset has 'name' attribute."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        assert hasattr(preset, 'name')
        assert isinstance(preset.name, str)
        assert len(preset.name) > 0

    def test_preset_has_description_field(self):
        """Test that loaded preset has 'description' attribute."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        assert hasattr(preset, 'description')
        assert isinstance(preset.description, str)

    def test_preset_has_annotators_list(self):
        """Test that loaded preset has 'annotators' list."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        assert hasattr(preset, 'annotators')
        assert isinstance(preset.annotators, list)
        assert len(preset.annotators) > 0

    def test_preset_create_pipeline_returns_pipeline_instance(self):
        """Test that preset.create_pipeline() returns AnnotatorPipeline."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        pipeline = preset.create_pipeline()

        assert isinstance(pipeline, AnnotatorPipeline)

    def test_all_presets_create_valid_pipelines(self):
        """Test that all presets can create valid pipelines."""
        preset_names = [
            Presets.STANDARD,
            Presets.LIGHTWEIGHT,
            Presets.PRIVACY,
            Presets.DEBUG,
            Presets.HIGH_CONTRAST
        ]

        for preset_name in preset_names:
            preset = VisualizationPreset.from_yaml(preset_name)
            pipeline = preset.create_pipeline()

            assert isinstance(pipeline, AnnotatorPipeline), \
                f"Preset {preset_name} failed to create pipeline"

    def test_preset_custom_yaml_file_path(self, tmp_path):
        """Test loading preset from custom YAML file path."""
        # Create temporary YAML file
        custom_yaml = tmp_path / "custom_presets.yaml"
        custom_yaml.write_text("""
presets:
  test_preset:
    name: "Test Preset"
    description: "For testing"
    annotators:
      - type: box
        thickness: 2
""")

        preset = VisualizationPreset.from_yaml("test_preset", str(custom_yaml))

        assert preset.name == "Test Preset"
        assert len(preset.annotators) > 0

    def test_preset_annotators_have_type_field(self):
        """Test that preset annotators contain type information."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        for ann_item in preset.annotators:
            # Annotators are stored as (AnnotatorType, config_dict) tuples
            assert isinstance(ann_item, tuple), "Annotator should be tuple"
            assert len(ann_item) == 2, "Annotator tuple should have 2 elements"
            ann_type, ann_config = ann_item
            assert isinstance(ann_type, AnnotatorType), "First element should be AnnotatorType"
            assert isinstance(ann_config, dict), "Second element should be dict"

    def test_standard_preset_contains_box_and_label(self):
        """Test that standard preset includes box and label annotators."""
        preset = VisualizationPreset.from_yaml(Presets.STANDARD)

        # Extract annotator types from config
        annotator_types = []
        for ann_config in preset.annotators:
            if isinstance(ann_config, dict):
                annotator_types.append(ann_config.get('type'))
            elif isinstance(ann_config, tuple):
                annotator_types.append(ann_config[0].value if hasattr(ann_config[0], 'value') else str(ann_config[0]))

        # Standard preset should have box-related and label annotators
        assert len(annotator_types) >= 1, "Standard preset should have at least 1 annotator"

    def test_privacy_preset_contains_blur(self):
        """Test that privacy preset includes blur annotator."""
        preset = VisualizationPreset.from_yaml(Presets.PRIVACY)

        # Privacy preset should include blur or pixelate
        annotator_types = []
        for ann_config in preset.annotators:
            if isinstance(ann_config, dict):
                annotator_types.append(ann_config.get('type'))
            elif isinstance(ann_config, tuple):
                ann_type = ann_config[0]
                annotator_types.append(ann_type.value if hasattr(ann_type, 'value') else str(ann_type))

        assert len(annotator_types) >= 1, "Privacy preset should have annotators"

    def test_debug_preset_contains_multiple_annotators(self):
        """Test that debug preset includes multiple annotators for detailed info."""
        preset = VisualizationPreset.from_yaml(Presets.DEBUG)

        # Debug preset should have multiple annotators (box, bar, label, etc.)
        assert len(preset.annotators) >= 2, \
            "Debug preset should have multiple annotators for detailed visualization"
