#!/usr/bin/env python3
"""Render a vertically stacked, frame-aligned comparison of two detectors.

The output deliberately contains only full rectangular detection boxes.  Model
names and inference settings are rendered in a header; detection labels are
not rendered.

Example:
    python examples/demo_compare_detect.py \\
        --input data/苏高速/video/example.mp4 \\
        --model-a-path models/vehicle_det_detr_batch1.onnx \\
        --model-a-type rtdetr \\
        --model-b-path models/rfdetr-medium_20260629_d_unified.onnx \\
        --model-b-type rfdetr_unified \\
        --output runs/model_compare.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from onnxtools import create_detector, setup_logger
from onnxtools.utils.supervision_annotator import AnnotatorPipeline, AnnotatorType


class BoxOnlyDetector:
    """Run one detector and draw only complete rectangular bounding boxes."""

    def __init__(self, model_type: str, model_path: str, conf_thres: float, iou_thres: float, thickness: int):
        self.detector = create_detector(
            model_type=model_type,
            onnx_path=model_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        self.pipeline = AnnotatorPipeline().add(AnnotatorType.BOX, {"thickness": thickness})

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and return the frame with full rectangular boxes only."""
        result = self.detector(frame)
        detections = result.to_supervision() if len(result) else sv.Detections.empty()
        return self.pipeline.annotate(frame, detections)


def _add_header(
    frame: np.ndarray,
    position: str,
    model_type: str,
    model_path: str,
    args: argparse.Namespace,
) -> np.ndarray:
    """Add model identity and shared inference settings above one comparison frame."""
    header_height = 76
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (0, 0, 0), thickness=-1)
    model_name = Path(model_path).name
    line_one = f"{position} | {model_type} | {model_name}"
    line_two = (
        f"conf={args.conf_thres:.2f} | iou={args.iou_thres:.2f} | "
        f"full boxes only | thickness={args.box_thickness} | CUDA preferred"
    )
    cv2.putText(frame, line_one, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line_two, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an MP4 writer, falling back to MPEG-4 Part 2 when H.264 is unavailable."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            logging.info("Writing %s with codec %s", output_path, codec)
            return writer
        writer.release()
    raise RuntimeError(f"Could not create video writer for {output_path}")


def compare(args: argparse.Namespace) -> None:
    """Run both models on every frame and write a vertically stacked comparison video."""
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector_a = BoxOnlyDetector(
        args.model_a_type, args.model_a_path, args.conf_thres, args.iou_thres, args.box_thickness
    )
    detector_b = BoxOnlyDetector(
        args.model_b_type, args.model_b_path, args.conf_thres, args.iou_thres, args.box_thickness
    )
    writer = _create_writer(output_path, fps, width, height * 2)

    logging.info("Processing %d frames at %.3f FPS", total_frames, fps)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            top = _add_header(detector_a.annotate(frame.copy()), "TOP", args.model_a_type, args.model_a_path, args)
            bottom = _add_header(
                detector_b.annotate(frame.copy()),
                "BOTTOM",
                args.model_b_type,
                args.model_b_path,
                args,
            )
            writer.write(np.vstack((top, bottom)))
            frame_index += 1
            if frame_index % 100 == 0:
                logging.info("Processed %d/%d frames", frame_index, total_frames)
    finally:
        cap.release()
        writer.release()

    logging.info("Finished %d frames: %s", frame_index, output_path)


def parse_args() -> argparse.Namespace:
    """Parse comparison-video command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare two ONNX detectors in a vertically stacked video.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--model-a-path", required=True, help="Top-model ONNX path")
    model_type_choices = ["rtdetr", "yolo", "rfdetr", "rfdetr_unified"]
    parser.add_argument("--model-a-type", required=True, choices=model_type_choices)
    parser.add_argument("--model-b-path", required=True, help="Bottom-model ONNX path")
    parser.add_argument("--model-b-type", required=True, choices=model_type_choices)
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--box-thickness", type=int, default=2)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    """Configure logging and run the comparison."""
    args = parse_args()
    setup_logger(args.log_level)
    compare(args)


if __name__ == "__main__":
    main()
