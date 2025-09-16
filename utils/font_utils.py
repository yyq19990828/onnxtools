"""Font detection utilities for cross-platform Chinese font support."""

import os
from typing import Optional


def get_chinese_font_path() -> Optional[str]:
    """
    Get Chinese font path with cross-platform support.

    Returns:
        Path to Chinese font file or None if not found
    """
    candidates = [
        "SourceHanSans-VF.ttf",  # Project local font (highest priority)
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux CJK
        "C:/Windows/Fonts/simhei.ttf",  # Windows Simplified Chinese
        "C:/Windows/Fonts/msyh.ttc",  # Windows Microsoft YaHei
        "C:/Windows/Fonts/simsun.ttc",  # Windows SimSun
        "C:/Windows/Fonts/arial.ttf",  # Windows fallback
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path

    return None


def validate_font_path(font_path: str) -> bool:
    """
    Validate if a font path exists and is accessible.

    Args:
        font_path: Path to font file

    Returns:
        True if font is valid and accessible
    """
    if not font_path:
        return False

    try:
        return os.path.exists(font_path) and os.path.isfile(font_path)
    except (OSError, ValueError):
        return False


def get_fallback_font_path(preferred_path: Optional[str] = None) -> Optional[str]:
    """
    Get font path with fallback support.

    Args:
        preferred_path: Preferred font path to try first

    Returns:
        Valid font path or None if no fonts found
    """
    # Try preferred path first
    if preferred_path and validate_font_path(preferred_path):
        return preferred_path

    # Fall back to system fonts
    return get_chinese_font_path()