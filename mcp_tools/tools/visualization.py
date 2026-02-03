"""
Visualization tools for MCP server.

Provides image annotation and object cropping tools.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from mcp.server.fastmcp import Image as MCPImage

from ..config import DEFAULT_OUTPUT_DIR
from ..models import AnnotateImageInput, CropDetectionsInput, ViewImageInput, ZoomToObjectInput
from ..utils.error_handler import handle_inference_error
from ..utils.image_loader import load_image, to_mcp_image
from ..utils.model_manager import get_detector
from ..utils.response_formatter import format_crop_response

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


# ============================================================================
# Core Tool Implementations (shared by all registration functions)
# ============================================================================

async def _crop_detections_impl(params: CropDetectionsInput) -> str:
    """Core implementation for crop_detections tool."""
    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Get detector
        detector = get_detector(
            params.model_path,
            params.model_type.value,
            conf_thres=params.conf_threshold,
        )

        # Run detection
        result = detector(image, conf_thres=params.conf_threshold)

        if len(result) == 0:
            return json.dumps(
                {
                    "total_crops": 0,
                    "crops": [],
                    "message": "No objects detected in the image.",
                },
                indent=2,
            )

        # Get crops
        crops = result.crop(
            conf_threshold=params.conf_threshold,
            classes=params.classes,
            gain=params.gain,
            pad=params.pad,
        )

        if not crops:
            filter_msg = f" matching classes {params.classes}" if params.classes else ""
            return json.dumps(
                {
                    "total_crops": 0,
                    "crops": [],
                    "message": f"No objects found{filter_msg} to crop.",
                },
                indent=2,
            )

        # Use default output dir if not specified
        output_dir = params.output_dir or DEFAULT_OUTPUT_DIR
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = result.save_crop(
            save_dir=str(output_path),
            conf_threshold=params.conf_threshold,
            classes=params.classes,
            gain=params.gain,
            pad=params.pad,
        )

        return format_crop_response(
            crops=[
                {
                    "index": c.get("index", i),
                    "class_name": c.get("class_name", "unknown"),
                    "confidence": float(c.get("confidence", 0)),
                    "box": (
                        c["box"].tolist()
                        if hasattr(c.get("box"), "tolist")
                        else list(c.get("box", []))
                    ),
                }
                for i, c in enumerate(crops)
            ],
            saved_paths=[str(p) for p in saved_paths],
            format_type=params.response_format.value,
        )

    except Exception as e:
        return handle_inference_error(e, "detection cropping")


async def _annotate_image_impl(params: AnnotateImageInput) -> str:
    """Core implementation for annotate_image tool."""
    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Get detector
        detector = get_detector(
            params.model_path,
            params.model_type.value,
            conf_thres=params.conf_threshold,
        )

        # Run detection
        result = detector(image, conf_thres=params.conf_threshold)

        # Generate annotated image (plot is called for side effects)
        _ = result.plot(annotator_preset=params.annotator_preset.value)

        response = {
            "status": "success",
            "total_detections": len(result),
            "preset_used": params.annotator_preset.value,
        }

        # Save if output path provided
        if params.output_path:
            result.save(
                params.output_path, annotator_preset=params.annotator_preset.value
            )
            response["output_path"] = params.output_path
            response["message"] = f"Annotated image saved to {params.output_path}"
        else:
            response["message"] = (
                "Image annotated successfully. "
                "Provide output_path parameter to save the result."
            )

        if params.response_format.value == "json":
            return json.dumps(response, indent=2)
        else:
            # Markdown format
            lines = ["# Image Annotation Result", ""]
            lines.append(f"**Status**: {response['status']}")
            lines.append(f"**Total Detections**: {response['total_detections']}")
            lines.append(f"**Preset Used**: {response['preset_used']}")
            if "output_path" in response:
                lines.append(f"**Saved to**: `{response['output_path']}`")
            lines.append("")
            lines.append(response["message"])
            return "\n".join(lines)

    except Exception as e:
        return handle_inference_error(e, "image annotation")


async def _zoom_to_object_impl(params: ZoomToObjectInput):
    """Core implementation for zoom_to_object tool."""
    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Get detector
        detector = get_detector(
            params.model_path,
            params.model_type.value,
            conf_thres=params.conf_threshold,
        )

        # Run detection
        result = detector(image, conf_thres=params.conf_threshold)

        if len(result) == 0:
            return "No objects detected in the image."

        # Filter by target class
        filtered = result.filter(classes=[params.target_class])

        if len(filtered) == 0:
            available_classes = list(set(
                result.names.get(int(cid), "unknown")
                for cid in result.class_ids
            ))
            return (
                f"No '{params.target_class}' detected in the image. "
                f"Available classes: {available_classes}"
            )

        # Sort by confidence and get the best one
        best_idx = filtered.scores.argmax()
        best_result = filtered[int(best_idx)]

        # Crop the best detection
        crops = best_result.crop(
            gain=params.gain,
            pad=params.pad,
        )

        if not crops:
            return "Failed to crop the detected object."

        # Get the cropped image
        crop_img = crops[0]["image"]

        # Convert to MCP Image and return
        return to_mcp_image(crop_img, format="jpeg")

    except Exception as e:
        return handle_inference_error(e, "zoom to object")


async def _view_image_impl(params: ViewImageInput):
    """Core implementation for view_image tool."""
    # Interpolation method mapping
    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }

    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Scale image if scale != 1.0
        if params.scale != 1.0:
            h, w = image.shape[:2]
            new_w = int(w * params.scale)
            new_h = int(h * params.scale)

            interpolation = INTERPOLATION_MAP.get(
                params.interpolation.value, cv2.INTER_LANCZOS4
            )

            image = cv2.resize(
                image, (new_w, new_h), interpolation=interpolation
            )

        # Convert to MCP Image and return
        return to_mcp_image(image, format="jpeg")

    except Exception as e:
        return handle_inference_error(e, "view image")


# ============================================================================
# Tool Registration Functions
# ============================================================================

def register_visualization_tools(mcp: "FastMCP") -> None:
    """Register all visualization tools with the MCP server.

    Args:
        mcp: FastMCP server instance
    """
    register_crop_detections_tool(mcp)
    register_annotate_image_tool(mcp)
    register_zoom_to_object_tool(mcp)
    register_enlarge_image_tool(mcp)


def register_crop_detections_tool(mcp: "FastMCP") -> None:
    """Register the crop_detections tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_crop_detections",
        annotations={
            "title": "Crop Detected Objects from Image",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def crop_detections(params: CropDetectionsInput) -> str:
        """Crop detected objects from an image and save to /tmp/onnxtools_mcp/.

        This tool detects objects and extracts cropped regions. By default,
        crops are saved to /tmp/onnxtools_mcp/ directory.

        Args:
            params (CropDetectionsInput): Crop parameters including:
                - image_path (str): Path to source image
                - classes (Optional[List[str]]): Filter classes (e.g., ["plate"])
                - output_dir (Optional[str]): Directory to save crops (default: /tmp/onnxtools_mcp)
                - gain (float): Bounding box expansion gain (default 1.02)
                - pad (int): Padding pixels (default 10)

        Returns:
            str: Crop results with saved file paths.
        """
        return await _crop_detections_impl(params)


def register_annotate_image_tool(mcp: "FastMCP") -> None:
    """Register the annotate_image tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_annotate_image",
        annotations={
            "title": "Annotate Image with Detections",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def annotate_image(params: AnnotateImageInput) -> str:
        """Annotate image with detection results using visualization presets.

        Available presets:
        - standard: Box corners with simple labels
        - debug: Round boxes with confidence bars and detailed labels
        - lightweight: Dot markers with small labels
        - privacy: Boxes with blurred plate regions
        - high_contrast: Filled regions with background dimming

        Args:
            params (AnnotateImageInput): Annotation parameters

        Returns:
            str: Annotation results in JSON or Markdown format.
        """
        return await _annotate_image_impl(params)


def register_zoom_to_object_tool(mcp: "FastMCP") -> None:
    """Register the zoom_to_object tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_zoom_to_object",
        annotations={
            "title": "Zoom to Detected Object",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def zoom_to_object(params: ZoomToObjectInput):
        """Zoom into a detected object and return the cropped image.

        This tool detects objects, selects the one with highest confidence matching
        the target class, crops it from the image, and returns the cropped image
        directly to the LLM for viewing.

        Args:
            params (ZoomToObjectInput): Zoom parameters including:
                - image_path (str): Path to input image
                - target_class (str): Class to zoom to (e.g., 'plate', 'car', 'truck')
                - gain (float): Bounding box expansion gain (default 1.02)
                - pad (int): Padding pixels (default 10)

        Returns:
            MCPImage: Cropped image that the LLM can directly view
            str: Error message if no objects are detected
        """
        return await _zoom_to_object_impl(params)


def register_enlarge_image_tool(mcp: "FastMCP") -> None:
    """Register the enlarge_image tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_enlarge_image",
        annotations={
            "title": "Enlarge Image for Inspection",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def enlarge_image(params: ViewImageInput):
        """Load, optionally scale, and return an image for LLM to view directly.

        This tool loads an image, optionally scales it using high-quality
        interpolation, and returns it as MCPImage for LLM viewing.

        Useful for:
        - Viewing and enlarging cropped images from crop_detections
        - Viewing any image file without running detection
        - High-quality image scaling for better inspection

        Args:
            params (ViewImageInput): View parameters including:
                - image_path (str): Path to image file, URL, or base64 string
                - scale (float): Scale factor (2.0 = 2x larger). Default 1.0
                - interpolation (str): Method: 'lanczos4', 'cubic', 'linear', 'nearest', 'area'

        Returns:
            MCPImage: The (optionally scaled) image that the LLM can directly view
            str: Error message if image cannot be loaded
        """
        return await _view_image_impl(params)
