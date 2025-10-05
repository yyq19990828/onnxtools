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

from .font_utils import  get_fallback_font_path

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
        config: Dict[str, Any]
    ) -> sv.annotators.base.BaseAnnotator:
        """
        Create annotator instance from type and config.

        Args:
            annotator_type: AnnotatorType enum value
            config: Configuration dictionary with annotator parameters

        Returns:
            Supervision annotator instance

        Raises:
            ValueError: Unknown annotator type
        """
        # Ensure config is a dict
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
            thickness=config.get('thickness', 1),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS)
        )

    @staticmethod
    def _create_rich_label(config: Dict[str, Any]) -> sv.RichLabelAnnotator:
        """Create RichLabelAnnotator with font validation."""
        # Get font path with validation
        font_path = config.get('font_path', None)
        if font_path:
            # Validate the provided font path
            import os
            if not os.path.exists(font_path) or not os.path.isfile(font_path):
                logger.warning(f"Font path not found or invalid: {font_path}. Using fallback font.")
                font_path = get_fallback_font_path()
                if not font_path:
                    logger.error("No valid font path found. Text rendering may fail.")
                    # Try to use a minimal fallback
                    font_path = None
        else:
            font_path = get_fallback_font_path()

        return sv.RichLabelAnnotator(
            text_color=sv.Color.BLACK,
            color=config.get('color_palette', sv.ColorPalette.DEFAULT),
            color_lookup=config.get('color_lookup', sv.ColorLookup.CLASS),
            font_path=font_path,
            font_size=config.get('font_size', 25),
            smart_position=True
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
        """Create BackgroundOverlayAnnotator with robust color handling."""
        color = config.get('color', sv.Color.BLACK)

        # Handle different color formats
        if isinstance(color, str):
            color_lower = color.lower().strip()

            # Named colors
            color_map = {
                'black': sv.Color.BLACK,
                'white': sv.Color.WHITE,
                'red': sv.Color.RED,
                'green': sv.Color.GREEN,
                'blue': sv.Color.BLUE,
            }

            if color_lower in color_map:
                color = color_map[color_lower]
            elif color_lower.startswith('#'):
                # Hex color - validate format
                try:
                    # Remove # and validate hex format
                    hex_value = color_lower[1:]
                    if len(hex_value) in [3, 6]:
                        # Short form #RGB or full form #RRGGBB
                        if len(hex_value) == 3:
                            # Convert #RGB to #RRGGBB
                            hex_value = ''.join([c*2 for c in hex_value])
                        color = sv.Color.from_hex(f'#{hex_value}')
                    else:
                        logger.warning(f"Invalid hex color format: {color}. Using default black.")
                        color = sv.Color.BLACK
                except Exception as e:
                    logger.warning(f"Error parsing hex color {color}: {e}. Using default black.")
                    color = sv.Color.BLACK
            elif color_lower.startswith('rgb'):
                # RGB format like rgb(255,0,0) or rgba(255,0,0,1)
                try:
                    import re
                    numbers = re.findall(r'\d+', color_lower)
                    if len(numbers) >= 3:
                        r, g, b = int(numbers[0]), int(numbers[1]), int(numbers[2])
                        # Create sv.Color from RGB values
                        color = sv.Color(r=min(255, max(0, r)),
                                       g=min(255, max(0, g)),
                                       b=min(255, max(0, b)))
                    else:
                        logger.warning(f"Invalid RGB color format: {color}. Using default black.")
                        color = sv.Color.BLACK
                except Exception as e:
                    logger.warning(f"Error parsing RGB color {color}: {e}. Using default black.")
                    color = sv.Color.BLACK
            else:
                logger.warning(f"Unknown color format: {color}. Using default black.")
                color = sv.Color.BLACK
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            # Handle RGB tuple/list
            try:
                r, g, b = color[:3]
                color = sv.Color(r=min(255, max(0, int(r))),
                               g=min(255, max(0, int(g))),
                               b=min(255, max(0, int(b))))
            except Exception as e:
                logger.warning(f"Error converting color tuple {color}: {e}. Using default black.")
                color = sv.Color.BLACK

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
            position=config.get('position', sv.Position.TOP_CENTER),
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
        self._scene_cache: Optional[Tuple[int, np.ndarray]] = None
        self._conflict_set: Optional[set] = None

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

        Raises:
            ValueError: If adding annotator would create conflicts
        """
        if isinstance(annotator, AnnotatorType):
            # Check for mutually exclusive annotators before adding
            if not self._allow_annotator(annotator):
                conflicting = self._get_conflicts(annotator)
                raise ValueError(
                    f"Cannot add {annotator.value}: conflicts with existing annotators "
                    f"{[t.value for t in conflicting]}. These annotators are mutually exclusive."
                )

            # Create annotator from type and config
            if config is None:
                config = {}
            annotator_instance = AnnotatorFactory.create(annotator, config)
            self.annotators.append(annotator_instance)
            self.types.append(annotator)

            # Invalidate conflict cache
            self._conflict_set = None
        else:
            # Use provided annotator instance
            self.annotators.append(annotator)
            # Try to infer type (optional, for conflict detection)
            # For now, skip type tracking for direct instances

        return self

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Apply all annotators in order.

        Args:
            scene: Input image (numpy array)
            detections: Detections to annotate
            use_cache: Whether to use cached scene copy for repeated calls

        Returns:
            Annotated image (copy)
        """
        # Use cached copy if available and requested
        if use_cache and self._scene_cache is not None:
            scene_id = id(scene)
            if self._scene_cache[0] == scene_id:
                result = self._scene_cache[1].copy()
            else:
                # Cache miss - create new copy and cache it
                result = scene.copy()
                self._scene_cache = (scene_id, result.copy())
        else:
            # Create copy without caching
            result = scene.copy()
            if use_cache:
                self._scene_cache = (id(scene), result.copy())

        # Apply each annotator sequentially
        for annotator in self.annotators:
            result = annotator.annotate(result, detections)

        return result

    def reset(self) -> 'AnnotatorPipeline':
        """
        Clear all annotators from the pipeline.

        Returns:
            self for method chaining
        """
        self.annotators.clear()
        self.types.clear()
        self._scene_cache = None
        self._conflict_set = None
        logger.debug("Pipeline reset - all annotators removed")
        return self

    def _allow_annotator(self, new_type: AnnotatorType) -> bool:
        """
        Check if annotator can be added without conflicts.

        Args:
            new_type: AnnotatorType to check

        Returns:
            True if annotator can be added
        """
        conflicts = self._get_conflicts(new_type)
        return len(conflicts) == 0

    def _get_conflicts(self, new_type: AnnotatorType) -> List[AnnotatorType]:
        """
        Get list of existing annotators that conflict with the new type.

        Args:
            new_type: AnnotatorType to check

        Returns:
            List of conflicting AnnotatorTypes
        """
        if self._conflict_set is None:
            self._rebuild_conflict_cache()

        conflicts = []
        for existing_type in self.types:
            if (new_type, existing_type) in self._conflict_set or \
               (existing_type, new_type) in self._conflict_set:
                conflicts.append(existing_type)

        return conflicts

    def _rebuild_conflict_cache(self):
        """Rebuild the conflict set cache for O(1) lookups."""
        self._conflict_set = set()
        for pair in CONFLICTING_PAIRS:
            self._conflict_set.add(pair)
            # Add reverse pair for bidirectional conflict checking
            self._conflict_set.add((pair[1], pair[0]))

    def check_conflicts(self) -> List[str]:
        """
        Check for conflicting annotator combinations.

        Returns:
            List of warning messages
        """
        if self._conflict_set is None:
            self._rebuild_conflict_cache()

        warnings = []
        checked_pairs = set()

        # Use set-based lookup for O(n²) → O(n²) with O(1) lookups
        for i, type_a in enumerate(self.types):
            for type_b in self.types[i + 1:]:
                # Create canonical pair to avoid duplicates
                pair = (type_a, type_b) if type_a.value < type_b.value else (type_b, type_a)

                if pair not in checked_pairs:
                    checked_pairs.add(pair)

                    # O(1) lookup in conflict set
                    if pair in self._conflict_set or (pair[1], pair[0]) in self._conflict_set:
                        warning_msg = (
                            f"Potential conflict: {type_a.value} + {type_b.value}. "
                            f"Visual effects may overlap or interfere."
                        )
                        warnings.append(warning_msg)
                        logger.warning(warning_msg)

        return warnings