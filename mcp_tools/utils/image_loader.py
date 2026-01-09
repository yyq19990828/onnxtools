"""
Image loading utilities for MCP tools.

Supports loading images from file paths, URLs, and base64 strings.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple

import cv2
import httpx
import numpy as np

from ..config import HTTP_TIMEOUT_SECONDS, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


async def load_image(
    source: str,
    source_type: str = "file"
) -> Tuple[np.ndarray, str]:
    """Load image from various sources.

    Args:
        source: Image source (file path, URL, or base64 string)
        source_type: Source type ('file', 'url', 'base64')

    Returns:
        Tuple of (image_array in BGR format, source_info string)

    Raises:
        ValueError: If image cannot be loaded or decoded
        FileNotFoundError: If file path does not exist
    """
    if source_type == "file":
        return _load_from_file(source)
    elif source_type == "url":
        return await _load_from_url(source)
    elif source_type == "base64":
        return _load_from_base64(source)
    else:
        raise ValueError(f"Unknown source type: {source_type}. Use 'file', 'url', or 'base64'.")


def _load_from_file(path: str) -> Tuple[np.ndarray, str]:
    """Load image from local file path.

    Args:
        path: Path to image file

    Returns:
        Tuple of (image_array, absolute_path)
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Check file extension
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: {SUPPORTED_IMAGE_FORMATS}"
        )

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to read image file: {path}")

    logger.debug(f"Loaded image from file: {file_path.absolute()}")
    return image, str(file_path.absolute())


async def _load_from_url(url: str) -> Tuple[np.ndarray, str]:
    """Load image from URL.

    Args:
        url: HTTP/HTTPS URL to image

    Returns:
        Tuple of (image_array, url)
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL scheme. Expected http:// or https://, got: {url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
            response.raise_for_status()
        except httpx.TimeoutException:
            raise ValueError(f"Timeout loading image from URL: {url}")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"HTTP error {e.response.status_code} loading image from URL: {url}")
        except httpx.RequestError as e:
            raise ValueError(f"Request error loading image from URL: {url} - {e}")

        image_bytes = response.content
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image from URL: {url}")

        logger.debug(f"Loaded image from URL: {url}")
        return image, url


def _load_from_base64(data: str) -> Tuple[np.ndarray, str]:
    """Load image from base64 string.

    Args:
        data: Base64 encoded image string (with or without data URL prefix)

    Returns:
        Tuple of (image_array, "base64_input")
    """
    try:
        # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
        if data.startswith("data:image"):
            # Extract the base64 part after the comma
            if "," in data:
                data = data.split(",", 1)[1]
            else:
                raise ValueError("Invalid data URL format: missing comma separator")

        # Decode base64
        image_bytes = base64.b64decode(data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode base64 image data")

        logger.debug("Loaded image from base64 string")
        return image, "base64_input"

    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding: {e}")
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {e}")


def image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """Convert image array to base64 string.

    Args:
        image: Image array in BGR format
        format: Output format ('jpeg' or 'png')

    Returns:
        Base64 encoded string
    """
    ext = f".{format}"
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to {format}")

    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/{format};base64,{base64_str}"


def to_mcp_image(image: np.ndarray, format: str = "jpeg") -> "MCPImage":
    """Convert OpenCV image (numpy array) to MCP Image object.

    This allows tools to return images directly to the LLM for viewing.
    The MCP protocol handles base64 encoding and transmission automatically.

    Args:
        image: Image array in BGR format (OpenCV default)
        format: Output format ('jpeg' or 'png')

    Returns:
        MCPImage object that can be returned from MCP tools

    Example:
        @mcp.tool()
        async def get_image() -> MCPImage:
            image = cv2.imread("test.jpg")
            return to_mcp_image(image)
    """
    from mcp.server.fastmcp import Image as MCPImage

    # Encode image to bytes (OpenCV uses BGR, but imencode handles this correctly)
    ext = f".{format}"
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to {format}")

    image_bytes = buffer.tobytes()

    return MCPImage(data=image_bytes, format=format)
