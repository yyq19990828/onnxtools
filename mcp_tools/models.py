"""
Pydantic models for MCP tools input/output validation.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import DEFAULT_COLOR_LAYER_MODEL, DEFAULT_CONF_THRESHOLD, DEFAULT_DETECTION_MODEL, DEFAULT_OCR_MODEL

# ============================================================================
# Enums
# ============================================================================


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    JSON = "json"
    MARKDOWN = "markdown"


class ImageSource(str, Enum):
    """Image input source type."""

    FILE = "file"
    URL = "url"
    BASE64 = "base64"


class ModelType(str, Enum):
    """Detection model architecture type."""

    YOLO = "yolo"
    RTDETR = "rtdetr"
    RFDETR = "rfdetr"


class AnnotatorPreset(str, Enum):
    """Visualization preset type."""

    STANDARD = "standard"
    DEBUG = "debug"
    LIGHTWEIGHT = "lightweight"
    PRIVACY = "privacy"
    HIGH_CONTRAST = "high_contrast"


# ============================================================================
# Input Models
# ============================================================================


class DetectObjectsInput(BaseModel):
    """Input model for object detection tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to image file, URL, or base64 encoded string",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type: 'file' for local path, 'url' for HTTP URL, 'base64' for encoded string",
    )
    model_path: str = Field(
        default=DEFAULT_DETECTION_MODEL,
        description="Path to ONNX detection model file",
    )
    model_type: ModelType = Field(
        default=ModelType.RTDETR,
        description="Detection model architecture: 'yolo', 'rtdetr', or 'rfdetr'",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for filtering detections (0.0-1.0)",
    )
    classes: Optional[List[str]] = Field(
        default=None,
        description="Filter specific classes by name (e.g., ['vehicle', 'plate']). None for all classes.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data, 'markdown' for human-readable",
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Image path cannot be empty")
        return v.strip()


class RecognizePlateInput(BaseModel):
    """Input model for plate OCR recognition tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to license plate image file, URL, or base64 string",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    model_path: str = Field(
        default=DEFAULT_OCR_MODEL,
        description="Path to OCR ONNX model file",
    )
    is_double_layer: bool = Field(
        default=False,
        description="Whether the plate is double-layer format (e.g., large vehicle plates)",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for OCR recognition",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format",
    )


class ClassifyPlateColorLayerInput(BaseModel):
    """Input model for plate color and layer classification."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to license plate image",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    model_path: str = Field(
        default=DEFAULT_COLOR_LAYER_MODEL,
        description="Path to color/layer classification ONNX model",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for classification",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format",
    )


class CropDetectionsInput(BaseModel):
    """Input model for cropping detected objects from image."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to source image",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    model_path: str = Field(
        default=DEFAULT_DETECTION_MODEL,
        description="Path to detection ONNX model",
    )
    model_type: ModelType = Field(
        default=ModelType.RTDETR,
        description="Detection model type",
    )
    classes: Optional[List[str]] = Field(
        default=None,
        description="Classes to crop (e.g., ['plate']). None for all classes.",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save cropped images. If None, crops are not saved.",
    )
    gain: float = Field(
        default=1.02,
        gt=0.0,
        le=2.0,
        description="Bounding box expansion gain multiplier",
    )
    pad: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Padding pixels to add around crop",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format",
    )


class AnnotateImageInput(BaseModel):
    """Input model for image annotation/visualization."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to input image",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    model_path: str = Field(
        default=DEFAULT_DETECTION_MODEL,
        description="Path to detection model",
    )
    model_type: ModelType = Field(
        default=ModelType.RTDETR,
        description="Detection model type",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold",
    )
    annotator_preset: AnnotatorPreset = Field(
        default=AnnotatorPreset.STANDARD,
        description="Visualization preset: 'standard', 'debug', 'lightweight', 'privacy', 'high_contrast'",
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path to save annotated image. If None, image is not saved.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format",
    )


class ZoomToObjectInput(BaseModel):
    """Input model for zooming to a detected object and returning the cropped image.

    This tool returns an MCPImage that the LLM can directly view.
    Can be used to zoom to any detected class (plate, vehicle, etc.).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to input image",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    model_path: str = Field(
        default=DEFAULT_DETECTION_MODEL,
        description="Path to detection ONNX model",
    )
    model_type: ModelType = Field(
        default=ModelType.RTDETR,
        description="Detection model type",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection",
    )
    target_class: str = Field(
        default="plate",
        description="Target class to zoom to (e.g., 'plate', 'car', 'truck')",
    )
    gain: float = Field(
        default=1.02,
        gt=0.0,
        le=2.0,
        description="Bounding box expansion gain multiplier",
    )
    pad: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Padding pixels to add around crop",
    )


class InterpolationMethod(str, Enum):
    """OpenCV interpolation method for image scaling."""

    NEAREST = "nearest"      # Fastest, lowest quality
    LINEAR = "linear"        # Bilinear interpolation
    CUBIC = "cubic"          # Bicubic interpolation (good quality)
    LANCZOS4 = "lanczos4"    # Lanczos interpolation (best quality)
    AREA = "area"            # Best for downscaling


class ViewImageInput(BaseModel):
    """Input model for viewing and optionally scaling an image.

    This tool returns an MCPImage that the LLM can directly view.
    Supports high-quality image scaling using OpenCV interpolation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to image file, URL, or base64 encoded string",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type: 'file' for local path, 'url' for HTTP URL, 'base64' for encoded string",
    )
    scale: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Scale factor for image (e.g., 2.0 = 2x larger, 0.5 = half size). Default 1.0 (no scaling).",
    )
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.LANCZOS4,
        description="Interpolation method: 'lanczos4' (best quality), 'cubic', 'linear', 'nearest', 'area' (for downscaling)",
    )


class FullPipelineInput(BaseModel):
    """Input model for complete vehicle/plate detection and recognition pipeline."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_path: str = Field(
        ...,
        description="Path to input image",
        min_length=1,
    )
    image_source: ImageSource = Field(
        default=ImageSource.FILE,
        description="Image source type",
    )
    detection_model_path: str = Field(
        default=DEFAULT_DETECTION_MODEL,
        description="Path to detection ONNX model",
    )
    detection_model_type: ModelType = Field(
        default=ModelType.RTDETR,
        description="Detection model architecture",
    )
    ocr_model_path: str = Field(
        default=DEFAULT_OCR_MODEL,
        description="Path to OCR ONNX model",
    )
    color_model_path: str = Field(
        default=DEFAULT_COLOR_LAYER_MODEL,
        description="Path to color/layer classification model",
    )
    conf_threshold: float = Field(
        default=DEFAULT_CONF_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
    )
    annotator_preset: AnnotatorPreset = Field(
        default=AnnotatorPreset.DEBUG,
        description="Visualization preset for annotated output",
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path to save annotated image",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format",
    )


# ============================================================================
# Output Models (for documentation and type hints)
# ============================================================================


class DetectionResult(BaseModel):
    """Single detection result."""

    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    score: float = Field(..., description="Confidence score")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class DetectionOutput(BaseModel):
    """Detection tool output schema."""

    total_detections: int = Field(..., description="Total number of detections")
    detections: List[DetectionResult] = Field(..., description="List of detections")
    image_shape: List[int] = Field(..., description="Original image shape [H, W, C]")
    model_path: str = Field(..., description="Path to model used")


class OCROutput(BaseModel):
    """OCR tool output schema."""

    text: str = Field(..., description="Recognized plate text")
    confidence: float = Field(..., description="Average confidence score")
    char_confidences: List[float] = Field(..., description="Per-character confidences")
    is_double_layer: bool = Field(..., description="Whether plate is double-layer")


class ClassificationOutput(BaseModel):
    """Classification tool output schema."""

    labels: List[str] = Field(..., description="Classification labels")
    confidences: List[float] = Field(..., description="Confidence for each label")
    avg_confidence: float = Field(..., description="Average confidence")


class CropOutput(BaseModel):
    """Crop tool output schema."""

    total_crops: int = Field(..., description="Number of crops")
    crops: List[dict] = Field(..., description="Crop information")
    saved_paths: Optional[List[str]] = Field(None, description="Saved file paths")


class PipelineOutput(BaseModel):
    """Full pipeline output schema."""

    total_detections: int = Field(..., description="Total detections")
    vehicles: List[dict] = Field(..., description="Vehicle detections")
    plates: List[dict] = Field(..., description="Plate detections with OCR")
    output_path: Optional[str] = Field(None, description="Saved annotated image path")
