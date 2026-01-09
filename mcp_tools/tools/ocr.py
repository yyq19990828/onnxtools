"""
OCR recognition tools for MCP server.

Provides license plate OCR recognition.
"""

from typing import TYPE_CHECKING

from ..models import RecognizePlateInput
from ..utils.error_handler import handle_inference_error
from ..utils.image_loader import load_image
from ..utils.model_manager import get_ocr_model
from ..utils.response_formatter import format_ocr_response

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_ocr_tools(mcp: "FastMCP") -> None:
    """Register OCR tools with the MCP server.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool(
        name="onnxtools_recognize_plate",
        annotations={
            "title": "Recognize License Plate Text",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def recognize_plate(params: RecognizePlateInput) -> str:
        """Recognize license plate text using OCR.

        This tool performs optical character recognition on license plate images,
        supporting both single-layer and double-layer Chinese plates. It returns
        the recognized text and character-level confidence scores.

        Single-layer plates typically have 7-8 characters (e.g., "京A12345").
        Double-layer plates are used on large vehicles and have a different format.

        Args:
            params (RecognizePlateInput): OCR parameters including:
                - image_path (str): Path to plate image file, URL, or base64 string
                - image_source (ImageSource): Source type (file/url/base64)
                - model_path (str): Path to OCR ONNX model
                - is_double_layer (bool): Whether the plate is double-layer format
                - conf_threshold (float): Confidence threshold
                - response_format (ResponseFormat): Output format (json/markdown)

        Returns:
            str: OCR results in JSON or Markdown format.

            JSON format example:
            {
                "text": "京A12345",
                "confidence": 0.95,
                "char_confidences": [0.99, 0.98, 0.95, 0.94, 0.96, 0.93, 0.92],
                "is_double_layer": false
            }

            Markdown format shows the plate text prominently with a table of
            character-level confidences.

        Examples:
            - Single-layer plate: is_double_layer=False -> "京A12345"
            - Double-layer plate: is_double_layer=True -> "京AF1234学"
            - From cropped image: image_path="crops/plate_001.jpg"

        Error cases:
            - Returns error message if image cannot be loaded
            - Returns error message if OCR fails (unclear image, not a valid plate)
        """
        try:
            # Load image
            image, _ = await load_image(params.image_path, params.image_source.value)

            # Get or create OCR model
            ocr_model = get_ocr_model(params.model_path, params.conf_threshold)

            # Run OCR
            result = ocr_model(image, is_double_layer=params.is_double_layer)

            if result is None:
                return (
                    "Error: OCR recognition failed. "
                    "The image may be unclear, too small, or not a valid license plate. "
                    "Try using a clearer image or adjusting the crop region."
                )

            text, confidence, char_confidences = result

            # Convert char_confidences to list if needed
            if hasattr(char_confidences, "tolist"):
                char_confidences = char_confidences.tolist()
            elif not isinstance(char_confidences, list):
                char_confidences = list(char_confidences)

            return format_ocr_response(
                text=text,
                confidence=float(confidence),
                char_confidences=[float(c) for c in char_confidences],
                is_double_layer=params.is_double_layer,
                format_type=params.response_format.value,
            )

        except Exception as e:
            return handle_inference_error(e, "plate OCR recognition")
