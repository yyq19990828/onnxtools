"""
Classification tools for MCP server.

Provides license plate color and layer classification.
"""

from typing import TYPE_CHECKING

from ..models import ClassifyPlateColorLayerInput
from ..utils.error_handler import handle_inference_error
from ..utils.image_loader import load_image
from ..utils.model_manager import get_color_layer_classifier
from ..utils.response_formatter import format_classification_response

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_classification_tools(mcp: "FastMCP") -> None:
    """Register classification tools with the MCP server.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool(
        name="onnxtools_classify_plate_color_layer",
        annotations={
            "title": "Classify Plate Color and Layer",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def classify_plate_color_layer(params: ClassifyPlateColorLayerInput) -> str:
        """Classify license plate color and layer type.

        This tool classifies the color and layer type of a Chinese license plate image.

        Supported colors:
        - blue: Standard private vehicle plates
        - yellow: Large vehicles, taxis, driving school
        - white: Police, military, government vehicles
        - black: Foreign embassy vehicles
        - green: New energy vehicles

        Layer types:
        - single: Standard single-row plates (most common)
        - double: Double-row plates (large vehicles, motorcycles)

        Args:
            params (ClassifyPlateColorLayerInput): Classification parameters including:
                - image_path (str): Path to license plate image
                - image_source (ImageSource): Source type (file/url/base64)
                - model_path (str): Path to classification ONNX model
                - conf_threshold (float): Confidence threshold
                - response_format (ResponseFormat): Output format (json/markdown)

        Returns:
            str: Classification results in JSON or Markdown format.

            JSON format example:
            {
                "labels": ["blue", "single"],
                "confidences": [0.95, 0.88],
                "avg_confidence": 0.915
            }

            Markdown format shows each classification with confidence score.

        Examples:
            - Blue single-layer: Private car plate
            - Yellow double-layer: Large truck plate
            - Green single-layer: Electric vehicle plate

        Notes:
            - Input image should be a cropped license plate region
            - For best results, use images cropped from detection output
        """
        try:
            # Load image
            image, _ = await load_image(params.image_path, params.image_source.value)

            # Get or create classifier
            classifier = get_color_layer_classifier(
                params.model_path, params.conf_threshold
            )

            # Run classification
            result = classifier(image)

            # Extract labels and confidences
            labels = result.labels if hasattr(result, "labels") else []
            confidences = result.confidences if hasattr(result, "confidences") else []
            avg_confidence = (
                result.avg_confidence
                if hasattr(result, "avg_confidence")
                else (sum(confidences) / len(confidences) if confidences else 0.0)
            )

            # Convert to lists if needed
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            if hasattr(confidences, "tolist"):
                confidences = confidences.tolist()

            return format_classification_response(
                labels=list(labels),
                confidences=[float(c) for c in confidences],
                avg_confidence=float(avg_confidence),
                format_type=params.response_format.value,
            )

        except Exception as e:
            return handle_inference_error(e, "plate color/layer classification")
