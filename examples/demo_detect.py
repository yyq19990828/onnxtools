#!/usr/bin/env python3
"""Detection-only demo: detector + optional 2D ByteTrack + visualization.

No OCR, no color/layer classifier — pure object detection. Use this when you
just want bounding boxes (and optionally persistent tracker_id) over an
image, folder, video, camera, or RTSP stream.

Example:
    # Image
    python examples/demo_detect.py --model-path models/rtdetr-2024080100.onnx \\
                                   --input data/sample.jpg

    # Video + tracking
    python examples/demo_detect.py --model-path models/rtdetr-2024080100.onnx \\
                                   --input video.mp4 --enable-tracking \\
                                   --annotator-preset tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import cv2
import numpy as np
import supervision as sv

from onnxtools import create_detector, setup_logger
from onnxtools.tracking import SUPPORTED_TRACKERS, BaseTracker, create_tracker
from onnxtools.tracking.class_voting import ClassVotingTracker
from onnxtools.utils.supervision_preset import VisualizationPreset

PRESETS_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "tracker_presets.yaml")


def _load_preset(name: str) -> dict:
    """Load a named tracker preset from configs/tracker_presets.yaml."""
    import yaml

    path = os.path.abspath(PRESETS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"preset file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if name not in data:
        raise KeyError(f"preset {name!r} not in {path}; available: {list(data.keys())}")
    return data[name]


def infer_source_type(path: str) -> str:
    p = path.lower()
    if os.path.isdir(path):
        return "folder"
    if any(p.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        return "image"
    if any(p.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
        return "video"
    if p.startswith("rtsp://"):
        return "rtsp"
    if path.isdigit():
        return "camera"
    return "unknown"


def _resolve_class_names(detector, det_config_arg) -> list[str]:
    """Read class names from ONNX metadata, then arg, with default fallback."""
    if isinstance(det_config_arg, dict):
        max_id = max(det_config_arg.keys()) if det_config_arg else -1
        return [det_config_arg.get(i, f"class_{i}") for i in range(max_id + 1)]
    if detector.class_names:
        max_id = max(detector.class_names.keys())
        return [detector.class_names.get(i, f"class_{i}") for i in range(max_id + 1)]
    # Last resort: numeric labels
    return []


class Detector:
    """Thin orchestrator: detector + optional ByteTrack + annotator pipeline."""

    def __init__(
        self,
        model_type: str,
        model_path: str,
        conf_thres: float,
        iou_thres: float,
        annotator_preset: str,
        det_config_arg,
        enable_tracking: bool,
        tracker_algo: str,
        tracker_kwargs: dict,
        class_voting: bool = False,
        class_voting_decay: float = 1.0,
    ):
        self.detector = create_detector(
            model_type=model_type,
            onnx_path=model_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        self.class_names = _resolve_class_names(self.detector, det_config_arg)

        preset = VisualizationPreset.from_yaml(annotator_preset)
        self.annotator_pipeline = preset.create_pipeline()
        self.label_type = preset.label_type

        if enable_tracking:
            inner = create_tracker(tracker_algo, **tracker_kwargs)
            self.tracker: BaseTracker | None = (
                ClassVotingTracker(inner, decay=class_voting_decay) if class_voting else inner
            )
        else:
            self.tracker = None

    def reset_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        result = self.detector(frame)

        # Build sv.Detections (use to_supervision so we get whatever the Result
        # class already takes care of, e.g. xyxy clipping).
        sv_dets = result.to_supervision() if len(result) > 0 else sv.Detections.empty()

        # Tracking (must run every frame so lost_track_buffer ages correctly)
        if self.tracker is not None:
            sv_dets = self.tracker.update(sv_dets, frame)

        labels = self._labels(sv_dets)
        annotated = self.annotator_pipeline.annotate(frame.copy(), sv_dets, labels=labels)

        output = self._to_json(sv_dets)
        return annotated, output

    def _labels(self, dets: sv.Detections) -> list[str]:
        n = len(dets)
        if n == 0:
            return []
        scores = dets.confidence if dets.confidence is not None else np.zeros(n)
        class_ids = dets.class_id if dets.class_id is not None else np.zeros(n, int)
        tracker_ids = getattr(dets, "tracker_id", None)

        out = []
        for i in range(n):
            cid = int(class_ids[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
            prefix = ""
            if tracker_ids is not None and tracker_ids[i] is not None:
                try:
                    prefix = f"#{int(tracker_ids[i])} "
                except (TypeError, ValueError):
                    pass
            if self.label_type == "confidence_only":
                out.append(f"{prefix}{float(scores[i]):.2f}")
            else:
                out.append(f"{prefix}{name} {float(scores[i]):.2f}")
        return out

    def _to_json(self, dets: sv.Detections) -> list[dict]:
        if len(dets) == 0:
            return []
        scores = dets.confidence
        class_ids = dets.class_id
        tracker_ids = getattr(dets, "tracker_id", None)
        items = []
        for i in range(len(dets)):
            cid = int(class_ids[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
            x1, y1, x2, y2 = map(float, dets.xyxy[i])
            entry = {
                "type": name,
                "box2d": [x1, y1, x2, y2],
                "confidence": float(scores[i]),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            }
            if tracker_ids is not None:
                try:
                    entry["tracker_id"] = int(tracker_ids[i])
                except (TypeError, ValueError):
                    entry["tracker_id"] = None
            items.append(entry)
        return items


def process_image(det, image_path, output_dir, output_mode, save_json):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image {image_path}")
        return
    annotated, output = det(img)
    logging.info(f"{os.path.basename(image_path)}: {len(output)} detections")
    if output_mode == "save":
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), annotated)
        if save_json:
            base = os.path.splitext(os.path.basename(image_path))[0]
            with open(os.path.join(output_dir, base + ".json"), "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_folder(det, folder_path, output_dir, output_mode, save_json):
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")))
    logging.info(f"Found {len(files)} images in {folder_path}")
    for i, name in enumerate(files):
        logging.info(f"[{i + 1}/{len(files)}] {name}")
        process_image(det, os.path.join(folder_path, name), output_dir, output_mode, save_json)


def process_video(det, source, output_dir, output_mode, frame_skip, save_json):
    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"Could not open video source {source}")
        return
    source_name = (
        "camera"
        if str(source).isdigit()
        else "rtsp"
        if str(source).startswith("rtsp://")
        else os.path.splitext(os.path.basename(source))[0]
    )

    writer = None
    json_dir = None
    if output_mode == "save":
        video_out = os.path.join(output_dir, source_name)
        os.makedirs(video_out, exist_ok=True)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(video_out, f"{source_name}_result.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened():
            logging.info(f"Saving result video to {out_path}")
        else:
            logging.error("Could not open video writer")
            writer = None
        if save_json:
            json_dir = os.path.join(video_out, "json")
            os.makedirs(json_dir, exist_ok=True)

    det.reset_tracker()  # restart tracker IDs from 1 for this video

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total:
        logging.info(f"Processing {total} frames")

    n = 0
    last_frame: np.ndarray | None = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if n % (frame_skip + 1) == 0:
            annotated, output = det(frame)
            last_frame = annotated
            if save_json and json_dir is not None:
                with open(os.path.join(json_dir, f"{source_name}_{n:06d}.json"), "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            annotated = last_frame if last_frame is not None else frame
        if n and n % 100 == 0:
            logging.info(f"Processed frame {n}/{total}" if total else f"Processed frame {n}")
        if output_mode == "save" and writer:
            writer.write(annotated)
        elif output_mode == "show":
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished — {n} frames processed")


def main(args):
    setup_logger(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)

    det_config = args.det_config
    if det_config == "coco80":
        from onnxtools.config import COCO_CLASSES

        det_config = COCO_CLASSES

    # Apply preset (overrides defaults; explicit CLI flags still win because
    # argparse already parsed them — but here we don't distinguish, so the
    # preset overwrites everything wholesale; re-pass flags on CLI to tweak).
    tracker_kwargs = dict(
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.minimum_matching_threshold,
        frame_rate=args.track_frame_rate,
    )
    tracker_algo = args.tracker_algo
    class_voting = args.class_voting
    class_voting_decay = args.class_voting_decay

    if args.preset:
        p = _load_preset(args.preset)
        logging.info(f"Applying tracker preset: {args.preset}")
        args.enable_tracking = True
        tracker_algo = p.get("algo", tracker_algo)
        # Translate native-kwargs to demo_detect's CLI names. Both name
        # styles are accepted by the native tracker constructors.
        k = p.get("kwargs", {})
        tracker_kwargs = {
            "track_activation_threshold": k.get("track_high_thresh", tracker_kwargs["track_activation_threshold"]),
            "lost_track_buffer": k.get("track_buffer", tracker_kwargs["lost_track_buffer"]),
            "minimum_matching_threshold": k.get("match_thresh", tracker_kwargs["minimum_matching_threshold"]),
            "frame_rate": k.get("frame_rate", tracker_kwargs["frame_rate"]),
        }
        # Native trackers also accept new_track_thresh / class_aware — forward.
        for extra in ("new_track_thresh", "class_aware", "delta_t", "inertia"):
            if extra in k:
                tracker_kwargs[extra] = k[extra]
        hints = p.get("pipeline_hints", {})
        args.conf_thres = hints.get("conf_thres", args.conf_thres)
        args.iou_thres = hints.get("iou_thres", args.iou_thres)
        args.annotator_preset = hints.get("annotator_preset", args.annotator_preset)
        args.frame_skip = hints.get("demo_frame_skip", args.frame_skip)
        cv = p.get("class_voting", {})
        if cv.get("enabled", False):
            class_voting = True
            class_voting_decay = float(cv.get("decay", class_voting_decay))

    det = Detector(
        model_type=args.model_type,
        model_path=args.model_path,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        annotator_preset=args.annotator_preset,
        det_config_arg=det_config,
        enable_tracking=args.enable_tracking,
        tracker_algo=tracker_algo,
        tracker_kwargs=tracker_kwargs,
        class_voting=class_voting,
        class_voting_decay=class_voting_decay,
    )

    src = infer_source_type(args.input)
    logging.info(f"Source type: {src}")
    if src == "image":
        process_image(det, args.input, args.output_dir, args.output_mode, args.save_json)
    elif src == "folder":
        process_folder(det, args.input, args.output_dir, args.output_mode, args.save_json)
    elif src in ("video", "rtsp", "camera"):
        process_video(det, args.input, args.output_dir, args.output_mode, args.frame_skip, args.save_json)
    else:
        logging.error(f"Unknown source type for input: {args.input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection-only demo (no OCR / no color classifier)")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, default="rtdetr", choices=["rtdetr", "yolo", "rfdetr", "rfdetr_unified"]
    )
    parser.add_argument(
        "--det-config",
        type=str,
        default=None,
        help='"coco80" or path to YAML class config (default: read from ONNX metadata)',
    )
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.5)

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-mode", type=str, default="save", choices=["save", "show"])
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--save-json", action="store_true")

    parser.add_argument("--frame-skip", type=int, default=0)

    parser.add_argument(
        "--annotator-preset",
        type=str,
        default="standard",
        choices=["standard", "lightweight", "privacy", "debug", "high_contrast", "box_only", "tracking"],
    )

    parser.add_argument("--enable-tracking", action="store_true", help="Enable 2D tracking (video / camera only)")
    parser.add_argument(
        "--tracker-algo",
        type=str,
        default="bytetrack",
        choices=list(SUPPORTED_TRACKERS),
        help='Tracking algorithm. Currently only "bytetrack" (supervision built-in) is shipped.',
    )
    parser.add_argument("--track-activation-threshold", type=float, default=0.45)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--minimum-matching-threshold", type=float, default=0.8)
    parser.add_argument("--track-frame-rate", type=int, default=30)

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Named tracker preset from configs/tracker_presets.yaml "
        "(e.g. 'intersection_3hz'). Applies algo, kwargs, conf_thres, "
        "annotator_preset, frame_skip, and class_voting.",
    )
    parser.add_argument(
        "--class-voting",
        action="store_true",
        help="Stabilise per-track class_id via confidence-weighted majority vote.",
    )
    parser.add_argument(
        "--class-voting-decay",
        type=float,
        default=1.0,
        help="Exponential decay applied to vote weights each frame. 1.0 = no decay.",
    )

    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    main(parser.parse_args())
