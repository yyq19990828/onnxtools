"""
MCP Tools package.

This package contains all MCP tool implementations for onnxtools.
"""

from .classification import register_classification_tools
from .detection import (
    register_detection_tools,
    register_detect_objects_tool,
    register_full_pipeline_tool,
)
from .ocr import register_ocr_tools
from .visualization import (
    register_visualization_tools,
    register_crop_detections_tool,
    register_annotate_image_tool,
    register_zoom_to_object_tool,
    register_enlarge_image_tool,
)

__all__ = [
    # Detection
    "register_detection_tools",
    "register_detect_objects_tool",
    "register_full_pipeline_tool",
    # OCR
    "register_ocr_tools",
    # Classification
    "register_classification_tools",
    # Visualization
    "register_visualization_tools",
    "register_crop_detections_tool",
    "register_annotate_image_tool",
    "register_zoom_to_object_tool",
    "register_enlarge_image_tool",
    # Batch registration
    "register_all_tools",
    "register_selected_tools",
]


def register_all_tools(mcp) -> None:
    """Register all tools with the MCP server.

    Args:
        mcp: FastMCP server instance
    """
    register_detection_tools(mcp)
    register_ocr_tools(mcp)
    register_classification_tools(mcp)
    register_visualization_tools(mcp)


def register_selected_tools(mcp) -> None:
    """Register only selected tools with the MCP server.

    Currently exposes:
    - onnxtools_detect_objects: Object detection
    - onnxtools_zoom_to_object: Zoom and return cropped image to LLM
    - onnxtools_enlarge_image: Enlarge and view image for inspection
    - onnxtools_crop_detections: Crop and save detected objects
    - onnxtools_server_status: Server status (registered in server.py)

    Args:
        mcp: FastMCP server instance
    """
    register_detect_objects_tool(mcp)
    register_zoom_to_object_tool(mcp)
    register_enlarge_image_tool(mcp)
    register_crop_detections_tool(mcp)
