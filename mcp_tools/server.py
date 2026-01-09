"""
FastMCP server for ONNX vehicle plate recognition.

This module provides the MCP server implementation that exposes
onnxtools inference capabilities to LLMs through the MCP protocol.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .config import SERVER_NAME
from .tools import register_selected_tools
from .utils.model_manager import clear_cache, get_cache_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Log to stderr for stdio transport
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage server lifecycle.

    This context manager handles initialization and cleanup of server resources.
    Models are loaded lazily on first use, so this mainly handles cleanup.

    Args:
        server: FastMCP server instance

    Yields:
        Dictionary with server state (can be accessed via context)
    """
    logger.info(f"Starting {SERVER_NAME} MCP server...")

    # State that persists across requests
    state: Dict[str, Any] = {
        "initialized": True,
    }

    try:
        yield state
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down MCP server...")
        clear_cache()
        logger.info("Model cache cleared")


# Create FastMCP server instance
mcp = FastMCP(
    SERVER_NAME,
    lifespan=app_lifespan,
)

# Register selected tools only (detect_objects, zoom_to_plate)
# server_status is registered below
register_selected_tools(mcp)


# Add a utility tool to check server status
@mcp.tool(
    name="onnxtools_server_status",
    annotations={
        "title": "Get Server Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def server_status() -> str:
    """Get the status of the onnxtools MCP server.

    This tool returns information about the server's current state,
    including loaded models and cache statistics.

    Returns:
        str: Server status information in JSON format.

        Example:
        {
            "server_name": "onnxtools_mcp",
            "status": "running",
            "cache": {
                "cached_models": 2,
                "max_cache_size": 10,
                "model_keys": ["detector:models/rtdetr.onnx:rtdetr", "ocr:models/ocr.onnx"]
            }
        }
    """
    import json

    cache_info = get_cache_info()

    return json.dumps(
        {
            "server_name": SERVER_NAME,
            "status": "running",
            "cache": cache_info,
        },
        indent=2,
    )


def main():
    """Entry point for the MCP server."""
    try:
        logger.info(f"Running {SERVER_NAME} with stdio transport...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
