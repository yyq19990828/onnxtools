"""
Error handling utilities for MCP tools.

Provides unified error handling with actionable messages.
"""

import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)


def handle_inference_error(e: Exception, operation: str) -> str:
    """Handle inference errors with actionable messages.

    Args:
        e: Exception that occurred
        operation: Description of the operation that failed

    Returns:
        User-friendly error message with suggestions
    """
    logger.error(f"Error during {operation}: {e}")
    logger.debug(traceback.format_exc())

    error_type = type(e).__name__
    error_msg = str(e)

    # File not found errors
    if isinstance(e, FileNotFoundError):
        return (
            f"Error: File not found - {error_msg}. "
            "Please verify the file path exists and is accessible."
        )

    # GPU/CUDA errors
    if "CUDA" in error_msg or "GPU" in error_msg.upper():
        return (
            f"Error: GPU error during {operation}. "
            "The GPU may be unavailable or out of memory. "
            "Try using CPU provider by ensuring CUDA is not required."
        )

    # Memory errors
    if "memory" in error_msg.lower() or "oom" in error_msg.lower():
        return (
            f"Error: Out of memory during {operation}. "
            "Try reducing image size, using a smaller model, or freeing GPU memory."
        )

    # Shape/dimension errors
    if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
        return (
            f"Error: Input shape mismatch during {operation}. "
            "Ensure the image format is correct (BGR format, HWC layout). "
            f"Details: {error_msg}"
        )

    # ONNX Runtime errors
    if "onnxruntime" in error_msg.lower() or "onnx" in error_msg.lower():
        return (
            f"Error: ONNX Runtime error during {operation}. "
            f"The model may be corrupted or incompatible. Details: {error_msg}"
        )

    # Value errors (input validation)
    if isinstance(e, ValueError):
        return f"Error: Invalid input - {error_msg}"

    # Type errors
    if isinstance(e, TypeError):
        return f"Error: Type error - {error_msg}. Check input parameter types."

    # Network errors
    if "timeout" in error_msg.lower():
        return (
            f"Error: Timeout during {operation}. "
            "The request took too long. Try again or check network connectivity."
        )

    if "connection" in error_msg.lower() or "network" in error_msg.lower():
        return (
            f"Error: Network error during {operation}. "
            "Check your internet connection and try again."
        )

    # Generic error with full details
    return (
        f"Error during {operation}: [{error_type}] {error_msg}. "
        "Check the logs for more details."
    )


def validate_model_path(model_path: str, model_name: str = "Model") -> Optional[str]:
    """Validate that a model path exists.

    Args:
        model_path: Path to the model file
        model_name: Name of the model for error messages

    Returns:
        None if valid, error message string if invalid
    """
    from pathlib import Path

    path = Path(model_path)

    if not path.exists():
        return f"{model_name} file not found: {model_path}"

    if not path.is_file():
        return f"{model_name} path is not a file: {model_path}"

    if path.suffix.lower() not in [".onnx", ".engine"]:
        return f"{model_name} has unsupported format: {path.suffix}. Expected .onnx or .engine"

    return None


def validate_image_path(image_path: str, source_type: str = "file") -> Optional[str]:
    """Validate image path or source.

    Args:
        image_path: Path/URL/base64 of the image
        source_type: Source type ('file', 'url', 'base64')

    Returns:
        None if valid, error message string if invalid
    """
    if source_type == "file":
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            return f"Image file not found: {image_path}"
        if not path.is_file():
            return f"Image path is not a file: {image_path}"

    elif source_type == "url":
        if not image_path.startswith(("http://", "https://")):
            return f"Invalid URL scheme. Expected http:// or https://, got: {image_path}"

    elif source_type == "base64":
        if not image_path:
            return "Empty base64 string provided"

    return None
