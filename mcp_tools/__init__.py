"""
MCP Tools for ONNX Vehicle Plate Recognition.

This module provides MCP (Model Context Protocol) server implementation
that exposes onnxtools inference capabilities to LLMs.
"""

from .server import mcp

__all__ = ["mcp"]
__version__ = "0.1.0"
