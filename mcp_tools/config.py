"""
MCP server configuration.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default model paths
DEFAULT_DETECTION_MODEL = str(PROJECT_ROOT / "models" / "rtdetr-2024080100.onnx")
DEFAULT_OCR_MODEL = str(PROJECT_ROOT / "models" / "ocr.onnx")
DEFAULT_COLOR_LAYER_MODEL = str(PROJECT_ROOT / "models" / "color_layer.onnx")

# Default thresholds
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5

# Image loading
MAX_IMAGE_SIZE_MB = 50
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Response formatting
DEFAULT_RESPONSE_FORMAT = "json"

# Model cache settings
MAX_CACHED_MODELS = 10

# Timeouts
HTTP_TIMEOUT_SECONDS = 30.0

# Server name
SERVER_NAME = "onnxtools_mcp"

# Default output directory for crops and annotated images
DEFAULT_OUTPUT_DIR = "/tmp/onnxtools_mcp"
