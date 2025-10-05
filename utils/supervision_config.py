"""Configuration classes and utilities for supervision annotators."""

import supervision as sv
from typing import Optional, Dict, Any
from .font_utils import get_fallback_font_path


try:
    from .annotator_factory import AnnotatorType
except ImportError:
    # Fallback if annotator_factory is not available
    AnnotatorType = None


def create_box_annotator(thickness: int = 1,
                        color_palette: Optional[sv.ColorPalette] = None) -> sv.BoxAnnotator:
    """
    Create BoxAnnotator with optimized settings for vehicle detection.

    Args:
        thickness: Border thickness for detection boxes
        color_palette: Color palette for different classes

    Returns:
        Configured BoxAnnotator instance
    """
    return sv.BoxAnnotator(
        color=color_palette or sv.ColorPalette.DEFAULT,
        thickness=thickness,
        color_lookup=sv.ColorLookup.CLASS
    )


def create_rich_label_annotator(font_path: Optional[str] = None,
                               font_size: int = 16,
                               text_padding: int = 10,
                               color_palette: Optional[sv.ColorPalette] = None) -> sv.RichLabelAnnotator:
    """
    Create RichLabelAnnotator with Chinese font support and optimized settings.

    Args:
        font_path: Path to font file (will use fallback if None or invalid)
        font_size: Font size for text rendering
        text_padding: Padding around text
        color_palette: Color palette for label backgrounds

    Returns:
        Configured RichLabelAnnotator instance
    """
    # Get valid font path with fallback support
    valid_font_path = get_fallback_font_path(font_path)

    return sv.RichLabelAnnotator(
        color=color_palette or sv.ColorPalette.DEFAULT,
        text_color=sv.Color.BLACK,
        font_path=valid_font_path,
        font_size=font_size,
        text_padding=text_padding,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=sv.ColorLookup.CLASS,
        border_radius=3,
        smart_position=True  # Intelligent positioning to avoid overlaps
    )


class BoxAnnotatorConfig:
    """Configuration container for BoxAnnotator settings."""

    def __init__(self,
                 thickness: int = 3,
                 color_palette: Optional[sv.ColorPalette] = None,
                 color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS):
        self.thickness = thickness
        self.color_palette = color_palette or sv.ColorPalette.DEFAULT
        self.color_lookup = color_lookup

    def create_annotator(self) -> sv.BoxAnnotator:
        """Create BoxAnnotator from this configuration."""
        return sv.BoxAnnotator(
            color=self.color_palette,
            thickness=self.thickness,
            color_lookup=self.color_lookup
        )


class RichLabelAnnotatorConfig:
    """Configuration container for RichLabelAnnotator settings."""

    def __init__(self,
                 font_path: Optional[str] = None,
                 font_size: int = 16,
                 text_color: sv.Color = sv.Color.WHITE,
                 text_padding: int = 10,
                 text_position: sv.Position = sv.Position.TOP_LEFT,
                 color_palette: Optional[sv.ColorPalette] = None,
                 color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
                 border_radius: int = 3,
                 smart_position: bool = True):
        self.font_path = get_fallback_font_path(font_path)
        self.font_size = font_size
        self.text_color = text_color
        self.text_padding = text_padding
        self.text_position = text_position
        self.color_palette = color_palette or sv.ColorPalette.DEFAULT
        self.color_lookup = color_lookup
        self.border_radius = border_radius
        self.smart_position = smart_position

    def create_annotator(self) -> sv.RichLabelAnnotator:
        """Create RichLabelAnnotator from this configuration."""
        return sv.RichLabelAnnotator(
            color=self.color_palette,
            text_color=self.text_color,
            font_path=self.font_path,
            font_size=self.font_size,
            text_padding=self.text_padding,
            text_position=self.text_position,
            color_lookup=self.color_lookup,
            border_radius=self.border_radius,
            smart_position=self.smart_position
        )


def get_default_vehicle_detection_config() -> tuple[BoxAnnotatorConfig, RichLabelAnnotatorConfig]:
    """
    Get default configuration optimized for vehicle detection use case.

    Returns:
        Tuple of (BoxAnnotatorConfig, RichLabelAnnotatorConfig)
    """
    box_config = BoxAnnotatorConfig(
        thickness=3,
        color_palette=sv.ColorPalette.DEFAULT,
        color_lookup=sv.ColorLookup.CLASS
    )

    label_config = RichLabelAnnotatorConfig(
        font_path="SourceHanSans-VF.ttf",  # Will fallback if not found
        font_size=16,
        text_color=sv.Color.WHITE,
        text_padding=10,
        text_position=sv.Position.TOP_LEFT,
        color_palette=sv.ColorPalette.DEFAULT,
        color_lookup=sv.ColorLookup.CLASS,
        border_radius=3,
        smart_position=True
    )

    return box_config, label_config


def get_default_annotator_config(annotator_type: 'AnnotatorType') -> Dict[str, Any]:
    """
    Get default configuration for each annotator type.

    This function provides sensible defaults for all 13 supported annotator types,
    optimized for vehicle detection scenarios.

    Args:
        annotator_type: AnnotatorType enum value

    Returns:
        Dictionary containing default configuration parameters

    Example:
        >>> from utils.annotator_factory import AnnotatorType
        >>> config = get_default_annotator_config(AnnotatorType.ROUND_BOX)
        >>> annotator = AnnotatorFactory.create(AnnotatorType.ROUND_BOX, config)
    """
    if AnnotatorType is None:
        raise ImportError("AnnotatorType not available. Import annotator_factory first.")

    # Default configurations for each annotator type
    default_configs = {
        AnnotatorType.BOX: {
            'thickness': 2,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.RICH_LABEL: {
            'font_path': 'SourceHanSans-VF.ttf',
            'font_size': 16,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.ROUND_BOX: {
            'thickness': 2,
            'roundness': 0.3,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.BOX_CORNER: {
            'thickness': 2,
            'corner_length': 20,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.CIRCLE: {
            'thickness': 2,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.TRIANGLE: {
            'base': 20,
            'height': 20,
            'position': sv.Position.TOP_CENTER,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS,
            'outline_thickness': 0,
            'outline_color': sv.Color.BLACK
        },
        AnnotatorType.ELLIPSE: {
            'thickness': 2,
            'start_angle': 0,
            'end_angle': 360,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.DOT: {
            'radius': 5,
            'position': sv.Position.CENTER,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS,
            'outline_thickness': 0,
            'outline_color': sv.Color.BLACK
        },
        AnnotatorType.COLOR: {
            'opacity': 0.3,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.BACKGROUND_OVERLAY: {
            'color': sv.Color.BLACK,
            'opacity': 0.5
        },
        AnnotatorType.HALO: {
            'opacity': 0.3,
            'kernel_size': 40,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS
        },
        AnnotatorType.PERCENTAGE_BAR: {
            'height': 16,
            'width': 80,
            'border_color': sv.Color.BLACK,
            'position': sv.Position.TOP_LEFT,
            'color_palette': sv.ColorPalette.DEFAULT,
            'color_lookup': sv.ColorLookup.CLASS,
            'border_thickness': 1
        },
        AnnotatorType.BLUR: {
            'kernel_size': 15
        },
        AnnotatorType.PIXELATE: {
            'pixel_size': 20
        }
    }

    config = default_configs.get(annotator_type)
    if config is None:
        raise ValueError(f"No default configuration for annotator type: {annotator_type}")

    return config.copy()  # Return a copy to avoid mutation