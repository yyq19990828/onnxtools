"""
Annotator Factory and Pipeline for supervision library integration.

This module provides:
- AnnotatorType: Enum for 13 supported annotator types
- AnnotatorFactory: Factory for creating annotator instances
- AnnotatorPipeline: Pipeline for composing multiple annotators
"""

from enum import Enum
from typing import Union, Dict, Any, List, Tuple, Optional
import numpy as np
import supervision as sv
import logging

logger = logging.getLogger(__name__)


class AnnotatorType(Enum):
    """Supported annotator types in supervision library."""

    # Existing types
    BOX = "box"
    RICH_LABEL = "rich_label"

    # Border annotators
    ROUND_BOX = "round_box"
    BOX_CORNER = "box_corner"

    # Geometric markers
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    ELLIPSE = "ellipse"
    DOT = "dot"

    # Fill annotators
    COLOR = "color"
    BACKGROUND_OVERLAY = "background_overlay"

    # Effect annotators
    HALO = "halo"
    PERCENTAGE_BAR = "percentage_bar"

    # Privacy protection annotators
    BLUR = "blur"
    PIXELATE = "pixelate"


class AnnotatorFactory:
    """Factory for creating supervision annotator instances."""

    @staticmethod
    def create(
        annotator_type: AnnotatorType,
        config: Union[Dict[str, Any], 'BaseAnnotatorConfig']
    ) -> sv.annotators.base.BaseAnnotator:
        """
        Create annotator instance from type and config.

        Args:
            annotator_type: AnnotatorType enum value
            config: Configuration dict or config object

        Returns:
            Supervision annotator instance

        Raises:
            ValueError: Unknown annotator type
            TypeError: Invalid configuration type
        """
        # Convert config object to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        else:
            config_dict = dict(config) if config else {}

        # Dispatch to creator based on type
        creator_map = {
            AnnotatorType.BOX: AnnotatorFactory._create_box,
            AnnotatorType.RICH_LABEL: AnnotatorFactory._create_rich_label,
            AnnotatorType.ROUND_BOX: AnnotatorFactory._create_round_box,
            AnnotatorType.BOX_CORNER: AnnotatorFactory._create_box_corner,
            AnnotatorType.CIRCLE: AnnotatorFactory._create_circle,
            AnnotatorType.TRIANGLE: AnnotatorFactory._create_triangle,
            AnnotatorType.ELLIPSE: AnnotatorFactory._create_ellipse,
            AnnotatorType.DOT: AnnotatorFactory._create_dot,
            AnnotatorType.COLOR: AnnotatorFactory._create_color,
            AnnotatorType.BACKGROUND_OVERLAY: AnnotatorFactory._create_background_overlay,
            AnnotatorType.HALO: AnnotatorFactory._create_halo,
            AnnotatorType.PERCENTAGE_BAR: AnnotatorFactory._create_percentage_bar,
            AnnotatorType.BLUR: AnnotatorFactory._create_blur,
            AnnotatorType.PIXELATE: AnnotatorFactory._create_pixelate,
        }

        creator = creator_map.get(annotator_type)
        if creator is None:
            raise ValueError(f"Unknown annotator type: {annotator_type}")

        return creator(config_dict)

    # Private creator methods
    @staticmethod
    def _create_box(config: Dict[str, Any]) -> sv.BoxAnnotator:
        """Create BoxAnnotator."""
        return sv.BoxAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            thickness=config.get('thickness', 2),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_rich_label(config: Dict[str, Any]) -> sv.RichLabelAnnotator:
        """Create RichLabelAnnotator."""
        return sv.RichLabelAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            font_path=config.get('font_path'),
            font_size=config.get('font_size', 16)
        )

    @staticmethod
    def _create_round_box(config: Dict[str, Any]) -> sv.RoundBoxAnnotator:
        """Create RoundBoxAnnotator."""
        return sv.RoundBoxAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            thickness=config.get('thickness', 2),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            roundness=config.get('roundness', 0.3)
        )

    @staticmethod
    def _create_box_corner(config: Dict[str, Any]) -> sv.BoxCornerAnnotator:
        """Create BoxCornerAnnotator."""
        return sv.BoxCornerAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            thickness=config.get('thickness', 2),
            corner_length=config.get('corner_length', 20),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_circle(config: Dict[str, Any]) -> sv.CircleAnnotator:
        """Create CircleAnnotator."""
        return sv.CircleAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            thickness=config.get('thickness', 2),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_triangle(config: Dict[str, Any]) -> sv.TriangleAnnotator:
        """Create TriangleAnnotator."""
        return sv.TriangleAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            base=config.get('base', 20),
            height=config.get('height', 20),
            position=config.get('position', sv.Position.TOP_CENTER),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            outline_thickness=config.get('outline_thickness', 0),
            outline_color=config.get('outline_color', sv.Color.BLACK)
        )

    @staticmethod
    def _create_ellipse(config: Dict[str, Any]) -> sv.EllipseAnnotator:
        """Create EllipseAnnotator."""
        return sv.EllipseAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            thickness=config.get('thickness', 2),
            start_angle=config.get('start_angle', 0),
            end_angle=config.get('end_angle', 360),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_dot(config: Dict[str, Any]) -> sv.DotAnnotator:
        """Create DotAnnotator."""
        return sv.DotAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            radius=config.get('radius', 5),
            position=config.get('position', sv.Position.CENTER),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            outline_thickness=config.get('outline_thickness', 0),
            outline_color=config.get('outline_color', sv.Color.BLACK)
        )

    @staticmethod
    def _create_color(config: Dict[str, Any]) -> sv.ColorAnnotator:
        """Create ColorAnnotator."""
        return sv.ColorAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            opacity=config.get('opacity', 0.3),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_background_overlay(config: Dict[str, Any]) -> sv.BackgroundOverlayAnnotator:
        """Create BackgroundOverlayAnnotator."""
        color = config.get('color', sv.Color.BLACK)
        if isinstance(color, str):
            color = sv.Color.from_hex(color) if color.startswith('#') else sv.Color.BLACK

        return sv.BackgroundOverlayAnnotator(
            color=color,
            opacity=config.get('opacity', 0.5)
        )

    @staticmethod
    def _create_halo(config: Dict[str, Any]) -> sv.HaloAnnotator:
        """Create HaloAnnotator."""
        return sv.HaloAnnotator(
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            opacity=config.get('opacity', 0.3),
            kernel_size=config.get('kernel_size', 40),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_percentage_bar(config: Dict[str, Any]) -> sv.PercentageBarAnnotator:
        """Create PercentageBarAnnotator."""
        return sv.PercentageBarAnnotator(
            height=config.get('height', 16),
            width=config.get('width', 80),
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            border_color=config.get('border_color', sv.Color.BLACK),
            position=config.get('position', sv.Position.TOP_LEFT),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            border_thickness=config.get('border_thickness', 1)
        )

    @staticmethod
    def _create_blur(config: Dict[str, Any]) -> sv.BlurAnnotator:
        """Create BlurAnnotator."""
        return sv.BlurAnnotator(
            kernel_size=config.get('kernel_size', 15)
        )

    @staticmethod
    def _create_pixelate(config: Dict[str, Any]) -> sv.PixelateAnnotator:
        """Create PixelateAnnotator."""
        return sv.PixelateAnnotator(
            pixel_size=config.get('pixel_size', 20)
        )


# Conflicting annotator pairs
CONFLICTING_PAIRS = {
    (AnnotatorType.COLOR, AnnotatorType.BLUR),
    (AnnotatorType.COLOR, AnnotatorType.PIXELATE),
    (AnnotatorType.COLOR, AnnotatorType.BACKGROUND_OVERLAY),
    (AnnotatorType.BOX, AnnotatorType.ROUND_BOX),
    (AnnotatorType.BLUR, AnnotatorType.PIXELATE),
}


class AnnotatorPipeline:
    """Pipeline for composing multiple annotators in sequence."""

    def __init__(self):
        """Initialize empty pipeline."""
        self.annotators: List[sv.annotators.base.BaseAnnotator] = []
        self.types: List[AnnotatorType] = []

    def add(
        self,
        annotator: Union[sv.annotators.base.BaseAnnotator, AnnotatorType],
        config: Optional[Dict[str, Any]] = None
    ) -> 'AnnotatorPipeline':
        """
        Add annotator to pipeline (Builder pattern).

        Args:
            annotator: Annotator instance or AnnotatorType enum
            config: Configuration dict (required if annotator is AnnotatorType)

        Returns:
            self for method chaining
        """
        if isinstance(annotator, AnnotatorType):
            # Create annotator from type and config
            if config is None:
                config = {}
            annotator_instance = AnnotatorFactory.create(annotator, config)
            self.annotators.append(annotator_instance)
            self.types.append(annotator)
        else:
            # Use provided annotator instance
            self.annotators.append(annotator)
            # Try to infer type (optional, for conflict detection)
            # For now, skip type tracking for direct instances

        return self

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections
    ) -> np.ndarray:
        """
        Apply all annotators in order.

        Args:
            scene: Input image (numpy array)
            detections: Detections to annotate

        Returns:
            Annotated image (copy)
        """
        # Create copy to avoid modifying original
        result = scene.copy()

        # Apply each annotator sequentially
        for annotator in self.annotators:
            result = annotator.annotate(result, detections)

        return result

    def check_conflicts(self) -> List[str]:
        """
        Check for conflicting annotator combinations.

        Returns:
            List of warning messages
        """
        warnings = []

        # Check all pairs
        for i, type_a in enumerate(self.types):
            for type_b in self.types[i + 1:]:
                if (type_a, type_b) in CONFLICTING_PAIRS or \
                   (type_b, type_a) in CONFLICTING_PAIRS:
                    warning_msg = (
                        f"Potential conflict: {type_a.value} + {type_b.value}. "
                        f"Visual effects may overlap or interfere."
                    )
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

        return warnings