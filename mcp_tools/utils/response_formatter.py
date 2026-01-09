"""
Response formatting utilities for MCP tools.

Supports JSON and Markdown output formats.
"""

import json
from typing import Any, Dict, List, Optional


def format_detection_response(
    detections: List[Dict[str, Any]],
    image_shape: tuple,
    model_path: str,
    format_type: str = "json",
) -> str:
    """Format detection results.

    Args:
        detections: List of detection dictionaries
        image_shape: Original image shape (H, W) or (H, W, C)
        model_path: Path to the model used
        format_type: Output format ('json' or 'markdown')

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(
            {
                "total_detections": len(detections),
                "detections": detections,
                "image_shape": list(image_shape),
                "model_path": model_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    else:
        # Markdown format
        lines = ["# Detection Results", ""]
        lines.append(f"**Total Detections**: {len(detections)}")
        lines.append(f"**Image Shape**: {image_shape}")
        lines.append(f"**Model**: `{model_path}`")
        lines.append("")

        if detections:
            lines.append("## Detections")
            lines.append("")
            for i, det in enumerate(detections):
                lines.append(f"### Detection {i + 1}")
                lines.append(f"- **Class**: {det['class_name']} (ID: {det['class_id']})")
                lines.append(f"- **Confidence**: {det['score']:.4f}")
                box_str = ", ".join(f"{x:.1f}" for x in det["box"])
                lines.append(f"- **Box** [x1, y1, x2, y2]: [{box_str}]")
                lines.append("")
        else:
            lines.append("*No detections found*")

        return "\n".join(lines)


def format_ocr_response(
    text: str,
    confidence: float,
    char_confidences: List[float],
    is_double_layer: bool,
    format_type: str = "json",
) -> str:
    """Format OCR results.

    Args:
        text: Recognized text
        confidence: Average confidence score
        char_confidences: Per-character confidence scores
        is_double_layer: Whether the plate is double-layer
        format_type: Output format ('json' or 'markdown')

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(
            {
                "text": text,
                "confidence": confidence,
                "char_confidences": char_confidences,
                "is_double_layer": is_double_layer,
            },
            indent=2,
            ensure_ascii=False,
        )
    else:
        # Markdown format
        lines = ["# OCR Recognition Result", ""]
        lines.append(f"**Plate Text**: `{text}`")
        lines.append(f"**Confidence**: {confidence:.4f}")
        lines.append(f"**Layer Type**: {'Double-layer' if is_double_layer else 'Single-layer'}")
        lines.append("")

        if char_confidences and len(text) == len(char_confidences):
            lines.append("## Character Confidences")
            lines.append("")
            lines.append("| Character | Confidence |")
            lines.append("|:---------:|:----------:|")
            for char, conf in zip(text, char_confidences):
                lines.append(f"| `{char}` | {conf:.4f} |")

        return "\n".join(lines)


def format_classification_response(
    labels: List[str],
    confidences: List[float],
    avg_confidence: float,
    format_type: str = "json",
) -> str:
    """Format classification results.

    Args:
        labels: Classification labels
        confidences: Confidence scores for each label
        avg_confidence: Average confidence score
        format_type: Output format ('json' or 'markdown')

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(
            {
                "labels": labels,
                "confidences": confidences,
                "avg_confidence": avg_confidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    else:
        # Markdown format
        lines = ["# Classification Result", ""]

        # Create label-specific formatting
        label_names = ["Color", "Layer"] if len(labels) >= 2 else [f"Label {i+1}" for i in range(len(labels))]

        for name, label, conf in zip(label_names, labels, confidences):
            lines.append(f"- **{name}**: {label} ({conf:.4f})")

        lines.append("")
        lines.append(f"**Average Confidence**: {avg_confidence:.4f}")

        return "\n".join(lines)


def format_crop_response(
    crops: List[Dict[str, Any]],
    saved_paths: Optional[List[str]] = None,
    format_type: str = "json",
) -> str:
    """Format crop results.

    Args:
        crops: List of crop information dictionaries
        saved_paths: Optional list of saved file paths
        format_type: Output format ('json' or 'markdown')

    Returns:
        Formatted string
    """
    if format_type == "json":
        result = {
            "total_crops": len(crops),
            "crops": crops,
        }
        if saved_paths:
            result["saved_paths"] = saved_paths
        return json.dumps(result, indent=2, ensure_ascii=False)
    else:
        # Markdown format
        lines = ["# Crop Results", ""]
        lines.append(f"**Total Crops**: {len(crops)}")
        lines.append("")

        if crops:
            lines.append("## Cropped Objects")
            lines.append("")
            for i, crop in enumerate(crops):
                lines.append(f"### Crop {i + 1}")
                lines.append(f"- **Class**: {crop.get('class_name', 'unknown')}")
                lines.append(f"- **Confidence**: {crop.get('confidence', 0):.4f}")
                if "box" in crop:
                    box_str = ", ".join(f"{x:.1f}" for x in crop["box"])
                    lines.append(f"- **Original Box**: [{box_str}]")
                lines.append("")

        if saved_paths:
            lines.append("## Saved Files")
            lines.append("")
            for path in saved_paths:
                lines.append(f"- `{path}`")

        return "\n".join(lines)


def format_pipeline_response(
    total_detections: int,
    vehicles: List[Dict[str, Any]],
    plates: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    format_type: str = "json",
) -> str:
    """Format full pipeline results.

    Args:
        total_detections: Total number of detections
        vehicles: List of vehicle detection results
        plates: List of plate detection and OCR results
        output_path: Optional path to saved annotated image
        format_type: Output format ('json' or 'markdown')

    Returns:
        Formatted string
    """
    if format_type == "json":
        result = {
            "total_detections": total_detections,
            "vehicles": vehicles,
            "plates": plates,
        }
        if output_path:
            result["output_path"] = output_path
        return json.dumps(result, indent=2, ensure_ascii=False)
    else:
        # Markdown format
        lines = ["# Vehicle and Plate Recognition Results", ""]
        lines.append(f"**Total Detections**: {total_detections}")
        lines.append("")

        if vehicles:
            lines.append("## Vehicles")
            lines.append("")
            for i, v in enumerate(vehicles):
                lines.append(f"### Vehicle {i + 1}")
                lines.append(f"- **Type**: {v.get('class_name', 'vehicle')}")
                lines.append(f"- **Confidence**: {v.get('confidence', 0):.4f}")
                if "box" in v:
                    box_str = ", ".join(f"{x:.1f}" for x in v["box"])
                    lines.append(f"- **Box**: [{box_str}]")
                lines.append("")

        if plates:
            lines.append("## License Plates")
            lines.append("")
            for i, p in enumerate(plates):
                plate_text = p.get("text", "N/A")
                lines.append(f"### Plate {i + 1}: `{plate_text}`")
                lines.append(f"- **Color**: {p.get('color', 'unknown')}")
                lines.append(f"- **Layer**: {p.get('layer', 'unknown')}")
                lines.append(f"- **OCR Confidence**: {p.get('ocr_confidence', 0):.4f}")
                lines.append(f"- **Detection Confidence**: {p.get('detection_confidence', 0):.4f}")
                if "box" in p:
                    box_str = ", ".join(f"{x:.1f}" for x in p["box"])
                    lines.append(f"- **Box**: [{box_str}]")
                lines.append("")

        if output_path:
            lines.append("---")
            lines.append(f"**Annotated image saved to**: `{output_path}`")

        return "\n".join(lines)
