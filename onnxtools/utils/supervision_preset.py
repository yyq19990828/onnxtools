"""
Visualization Preset management for supervision annotators.

This module provides predefined annotator combinations for common use cases.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import supervision as sv

from .. import config
from .supervision_annotator import AnnotatorPipeline, AnnotatorType


class Presets:
    """Predefined preset names."""
    STANDARD = "standard"
    LIGHTWEIGHT = "lightweight"
    PRIVACY = "privacy"
    DEBUG = "debug"
    HIGH_CONTRAST = "high_contrast"


@dataclass
class VisualizationPreset:
    """Visualization preset configuration."""
    name: str
    description: str
    annotators: List[Tuple[AnnotatorType, Dict[str, Any]]]
    label_type: Optional[str] = None  # e.g., "confidence_only" for confidence-only labels

    @classmethod
    def from_yaml(
        cls,
        preset_name: str,
        config_path: Optional[str] = None
    ) -> 'VisualizationPreset':
        """
        Load preset configuration.

        优先级:
        1. config_path显式指定的外部YAML文件(如果提供)
        2. config模块的硬编码配置(默认)

        Args:
            preset_name: Name of the preset to load
            config_path: Optional external config file path (YAML format)

        Returns:
            VisualizationPreset instance

        Raises:
            ValueError: Unknown preset name
            FileNotFoundError: YAML file not found (if config_path provided)

        Note:
            configs/visualization_presets.yaml仅作为外部配置示例保留,
            默认情况下使用config模块的硬编码常量。
        """
        # 使用config模块加载配置（默认硬编码或外部YAML）
        data = config.load_visualization_config(config_path)

        if 'presets' not in data or preset_name not in data['presets']:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset_data = data['presets'][preset_name]

        # Parse annotators
        annotators = []
        for ann_config in preset_data['annotators']:
            # Create a copy to avoid modifying the original config
            config_copy = ann_config.copy()
            ann_type_str = config_copy.pop('type')
            ann_type = AnnotatorType(ann_type_str)

            # Convert position string to sv.Position enum if present
            if 'position' in config_copy:
                position_str = config_copy['position']
                if isinstance(position_str, str):
                    # Map string to sv.Position enum
                    position_map = {
                        'CENTER': sv.Position.CENTER,
                        'TOP_LEFT': sv.Position.TOP_LEFT,
                        'TOP_CENTER': sv.Position.TOP_CENTER,
                        'TOP_RIGHT': sv.Position.TOP_RIGHT,
                        'CENTER_LEFT': sv.Position.CENTER_LEFT,
                        'CENTER_RIGHT': sv.Position.CENTER_RIGHT,
                        'BOTTOM_LEFT': sv.Position.BOTTOM_LEFT,
                        'BOTTOM_CENTER': sv.Position.BOTTOM_CENTER,
                        'BOTTOM_RIGHT': sv.Position.BOTTOM_RIGHT,
                    }
                    config_copy['position'] = position_map.get(
                        position_str.upper(),
                        sv.Position.CENTER
                    )

            # Convert color string to sv.Color if present
            if 'color' in config_copy:
                color_str = config_copy['color']
                if isinstance(color_str, str):
                    if color_str == 'black':
                        config_copy['color'] = sv.Color.BLACK
                    elif color_str == 'white':
                        config_copy['color'] = sv.Color.WHITE
                    elif color_str.startswith('#'):
                        config_copy['color'] = sv.Color.from_hex(color_str)

            annotators.append((ann_type, config_copy))

        return cls(
            name=preset_data['name'],
            description=preset_data['description'],
            annotators=annotators,
            label_type=preset_data.get('label_type', None)
        )

    def create_pipeline(self) -> AnnotatorPipeline:
        """
        Create AnnotatorPipeline from this preset.

        Returns:
            Configured AnnotatorPipeline instance
        """
        pipeline = AnnotatorPipeline()

        for ann_type, ann_config in self.annotators:
            pipeline.add(ann_type, ann_config)

        return pipeline


@lru_cache(maxsize=32)
def load_preset_cached(preset_name: str, config_path: Optional[str] = None) -> VisualizationPreset:
    """
    Load preset with caching.

    This is a cached version of VisualizationPreset.from_yaml() that avoids
    repeated config loading for the same presets.

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. config模块的硬编码配置(默认)

    Args:
        preset_name: Name of the preset to load
        config_path: Optional external config file path (YAML format)

    Returns:
        Cached VisualizationPreset instance

    Raises:
        ValueError: Unknown preset name
        FileNotFoundError: YAML file not found (if config_path provided)
    """
    return VisualizationPreset.from_yaml(preset_name, config_path)


def create_preset_pipeline(preset_name: str, config_path: Optional[str] = None) -> AnnotatorPipeline:
    """
    Create AnnotatorPipeline from preset with caching.

    This convenience function combines preset loading and pipeline creation
    with automatic caching to improve performance for repeated calls.

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. config模块的硬编码配置(默认)

    Args:
        preset_name: Name of the preset to use
        config_path: Optional external config file path (YAML format)

    Returns:
        Configured AnnotatorPipeline instance

    Example:
        >>> pipeline = create_preset_pipeline(Presets.DEBUG)
        >>> annotated = pipeline.annotate(image, detections)
    """
    preset = load_preset_cached(preset_name, config_path)
    return preset.create_pipeline()
