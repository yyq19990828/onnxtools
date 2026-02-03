"""
Detection tools for MCP server.

Provides object detection and full pipeline tools.
"""

from typing import TYPE_CHECKING

from ..models import DetectObjectsInput, FullPipelineInput
from ..utils.error_handler import handle_inference_error
from ..utils.image_loader import load_image
from ..utils.model_manager import get_color_layer_classifier, get_detector, get_ocr_model
from ..utils.response_formatter import format_detection_response, format_pipeline_response

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


# ============================================================================
# Core Tool Implementations
# ============================================================================

async def _detect_objects_impl(params: DetectObjectsInput) -> str:
    """Core implementation for detect_objects tool."""
    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Get or create detector
        detector = get_detector(
            params.model_path,
            params.model_type.value,
            conf_thres=params.conf_threshold,
        )

        # Run detection
        result = detector(image, conf_thres=params.conf_threshold)

        # Filter by classes if specified
        if params.classes:
            result = result.filter(classes=params.classes)

        # Format detections
        detections = []
        for i in range(len(result)):
            detections.append(
                {
                    "box": result.boxes[i].tolist(),
                    "score": float(result.scores[i]),
                    "class_id": int(result.class_ids[i]),
                    "class_name": result.names.get(
                        int(result.class_ids[i]), "unknown"
                    ),
                }
            )

        return format_detection_response(
            detections=detections,
            image_shape=result.orig_shape,
            model_path=params.model_path,
            format_type=params.response_format.value,
        )

    except Exception as e:
        return handle_inference_error(e, "object detection")


async def _full_pipeline_impl(params: FullPipelineInput) -> str:
    """Core implementation for full_pipeline tool."""
    try:
        # Load image
        image, _ = await load_image(
            params.image_path, params.image_source.value
        )

        # Get models
        detector = get_detector(
            params.detection_model_path,
            params.detection_model_type.value,
            conf_thres=params.conf_threshold,
        )

        # Run detection
        result = detector(image, conf_thres=params.conf_threshold)

        vehicles = []
        plates = []

        # Process each detection
        for i in range(len(result)):
            box = result.boxes[i].tolist()
            class_name = result.names.get(int(result.class_ids[i]), "unknown")
            score = float(result.scores[i])

            if class_name == "plate":
                # Get plate crop
                single_result = result[i]
                crops = single_result.crop()

                if crops:
                    plate_img = crops[0]["image"]

                    # Classify color/layer
                    try:
                        color_classifier = get_color_layer_classifier(
                            params.color_model_path, params.conf_threshold
                        )
                        cls_result = color_classifier(plate_img)
                        color = cls_result.labels[0] if cls_result.labels else "unknown"
                        layer = cls_result.labels[1] if len(cls_result.labels) > 1 else "single"
                    except Exception:
                        color = "unknown"
                        layer = "single"

                    # OCR recognition
                    try:
                        ocr_model = get_ocr_model(
                            params.ocr_model_path, params.conf_threshold
                        )
                        is_double = layer == "double"
                        ocr_result = ocr_model(plate_img, is_double_layer=is_double)
                        text = ocr_result[0] if ocr_result else ""
                        ocr_conf = ocr_result[1] if ocr_result else 0.0
                    except Exception:
                        text = ""
                        ocr_conf = 0.0

                    plates.append(
                        {
                            "box": box,
                            "text": text,
                            "color": color,
                            "layer": layer,
                            "ocr_confidence": ocr_conf,
                            "detection_confidence": score,
                        }
                    )
            else:
                vehicles.append(
                    {
                        "box": box,
                        "class_name": class_name,
                        "confidence": score,
                    }
                )

        # Save annotated image if path provided
        output_info = None
        if params.output_path:
            result.save(
                params.output_path, annotator_preset=params.annotator_preset.value
            )
            output_info = params.output_path

        return format_pipeline_response(
            total_detections=len(result),
            vehicles=vehicles,
            plates=plates,
            output_path=output_info,
            format_type=params.response_format.value,
        )

    except Exception as e:
        return handle_inference_error(e, "full pipeline")


# ============================================================================
# Tool Registration Functions
# ============================================================================

def register_detection_tools(mcp: "FastMCP") -> None:
    """Register all detection tools with the MCP server.

    Args:
        mcp: FastMCP server instance
    """
    register_detect_objects_tool(mcp)
    register_full_pipeline_tool(mcp)


def register_detect_objects_tool(mcp: "FastMCP") -> None:
    """Register the detect_objects tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_detect_objects",
        annotations={
            "title": "Detect Objects in Image",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def detect_objects(params: DetectObjectsInput) -> str:
        """Detect vehicles and license plates in an image using ONNX models.

        This tool performs object detection on images using YOLO, RT-DETR, or RF-DETR
        architectures. It returns bounding boxes, confidence scores, and class labels
        for detected vehicles and plates.

        Args:
            params (DetectObjectsInput): Detection parameters including:
                - image_path (str): Path to image, URL, or base64 string
                - image_source (ImageSource): Source type (file/url/base64)
                - model_path (str): Path to ONNX detection model
                - model_type (ModelType): Architecture (yolo/rtdetr/rfdetr)
                - conf_threshold (float): Confidence threshold (0.0-1.0)
                - classes (Optional[List[str]]): Filter specific classes
                - response_format (ResponseFormat): Output format (json/markdown)

        Returns:
            str: Detection results in JSON or Markdown format.
        """
        return await _detect_objects_impl(params)


def register_full_pipeline_tool(mcp: "FastMCP") -> None:
    """Register the full_pipeline tool with the MCP server."""

    @mcp.tool(
        name="onnxtools_full_pipeline",
        annotations={
            "title": "Full Vehicle and Plate Detection Pipeline",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def full_pipeline(params: FullPipelineInput) -> str:
        """Execute complete vehicle and license plate detection, OCR, and visualization.

        This tool runs the full inference pipeline:
        1. Detect vehicles and plates in the image
        2. For each detected plate: classify color/layer and perform OCR
        3. Optionally annotate the image with all results
        4. Return comprehensive results

        Args:
            params (FullPipelineInput): Pipeline parameters

        Returns:
            str: Complete pipeline results in JSON or Markdown format.
        """
        return await _full_pipeline_impl(params)
