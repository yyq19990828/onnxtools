#!/usr/bin/env python3
"""Main inference script for vehicle and license plate detection.

This script uses the InferencePipeline class which provides a complete
end-to-end inference workflow including detection, OCR, and visualization.

Example usage:
    # Single image
    python main.py --model-path models/rtdetr.onnx --model-type rtdetr \\
                   --input data/sample.jpg --output-mode save

    # Video with frame skip
    python main.py --model-path models/yolo11n.onnx --model-type yolo \\
                   --input video.mp4 --source-type video \\
                   --frame-skip 2 --output-mode save

    # Camera stream
    python main.py --model-path models/rtdetr.onnx --model-type rtdetr \\
                   --input 0 --source-type camera --output-mode show
"""

import cv2
import json
import os
import argparse
import logging

from onnxtools import InferencePipeline, setup_logger


def infer_source_type(input_path):
    """Infer the source type based on the input path.

    Args:
        input_path: Path or identifier for the input source

    Returns:
        str: Source type ('image', 'video', 'folder', 'camera', 'rtsp', 'unknown')
    """
    input_path_lower = input_path.lower()
    if os.path.isdir(input_path):
        return 'folder'
    elif any(input_path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        return 'image'
    elif any(input_path_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        return 'video'
    elif input_path_lower.startswith('rtsp://'):
        return 'rtsp'
    elif input_path.isdigit():
        return 'camera'
    else:
        return 'unknown'


def process_single_image(pipeline, image_path, output_dir, output_mode, save_json=True):
    """Process a single image.

    Args:
        pipeline: InferencePipeline instance
        image_path: Path to input image
        output_dir: Directory for output files
        output_mode: 'save' or 'show'
        save_json: Whether to save JSON results
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image {image_path}")
        return

    # Run inference
    result_img, output_data = pipeline(img)
    logging.info(f"Detected {len(output_data)} objects")

    # Count plates
    plate_count = sum(1 for d in output_data if 'plate_name' in d and d['plate_name'])
    if plate_count > 0:
        logging.info(f"Recognized {plate_count} license plates")

    # Output
    if output_mode == 'save':
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, result_img)
        logging.info(f"Result image saved to {output_image_path}")

        if save_json:
            output_json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"JSON results saved to {output_json_path}")
    elif output_mode == 'show':
        cv2.imshow("Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_folder(pipeline, folder_path, output_dir, output_mode, save_json=True):
    """Process all images in a folder.

    Args:
        pipeline: InferencePipeline instance
        folder_path: Path to input folder
        output_dir: Directory for output files
        output_mode: 'save' or 'show'
        save_json: Whether to save JSON results
    """
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)
    logging.info(f"Found {total_images} images in folder '{folder_path}'")

    for i, image_file in enumerate(image_files):
        logging.info(f"Processing image {i + 1}/{total_images}: {image_file}")
        image_path = os.path.join(folder_path, image_file)

        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not read image {image_path}, skipping.")
            continue

        # Run inference
        result_img, output_data = pipeline(img)

        # Output
        if output_mode == 'save':
            output_image_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_image_path, result_img)

            if save_json:
                output_json_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
        elif output_mode == 'show':
            cv2.imshow(f"Result - {image_file}", result_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    logging.info(f"Finished processing all {total_images} images.")


def process_video(pipeline, video_source, output_dir, output_mode, frame_skip=0,
                 save_frame=False, save_json=False):
    """Process video stream (file, camera, or RTSP).

    Args:
        pipeline: InferencePipeline instance
        video_source: Video file path, camera ID, or RTSP URL
        output_dir: Directory for output files
        output_mode: 'save' or 'show'
        frame_skip: Number of frames to skip between processing
        save_frame: Whether to save individual frames
        save_json: Whether to save JSON results per frame
    """
    # Setup video capture
    if isinstance(video_source, int) or video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        source_name = 'camera'
    else:
        cap = cv2.VideoCapture(video_source)
        source_name = os.path.splitext(os.path.basename(video_source))[0] if not video_source.startswith('rtsp://') else 'rtsp'

    if not cap.isOpened():
        logging.error(f"Could not open video source {video_source}")
        return

    # Setup output directories and video writer
    writer = None
    if output_mode == 'save':
        video_output_dir = os.path.join(output_dir, source_name)
        os.makedirs(video_output_dir, exist_ok=True)

        if save_frame:
            frames_dir = os.path.join(video_output_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
        if save_json:
            json_dir = os.path.join(video_output_dir, 'json')
            os.makedirs(json_dir, exist_ok=True)

        # Setup video writer
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video_path = os.path.join(video_output_dir, f"{source_name}_result.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            logging.warning("H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if writer.isOpened():
            logging.info(f"Saving result video to {output_video_path}")
        else:
            logging.error("Could not open video writer")
            writer = None

    # Process frames
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total_frames = 0

    if total_frames > 0:
        logging.info(f"Processing video with {total_frames} frames")
    else:
        logging.info("Processing video stream (total frames unknown)")

    frame_count = 0
    last_result_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (frame_skip + 1) == 0:
            # Run inference
            result_frame, output_data = pipeline(frame)
            last_result_frame = result_frame.copy()

            # Save result frame with annotations if requested
            if save_frame and output_mode == 'save':
                frame_filename = f"{source_name}_{frame_count:06d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, result_frame)

            # Save JSON if requested
            if save_json and output_mode == 'save':
                json_filename = f"{source_name}_{frame_count:06d}.json"
                json_path = os.path.join(json_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
        else:
            # Use last result frame if skipping
            if last_result_frame is not None:
                result_frame = last_result_frame
            else:
                result_frame = frame

        # Log progress
        if frame_count > 0 and frame_count % 100 == 0:
            if total_frames > 0:
                logging.info(f"Processed frame {frame_count}/{total_frames}")
            else:
                logging.info(f"Processed frame {frame_count}")

        # Output
        if output_mode == 'save':
            if writer:
                writer.write(result_frame)
        elif output_mode == 'show':
            cv2.imshow("Result", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    logging.info("Finished processing video")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main(args):
    """Main entry point."""
    # Setup logger
    setup_logger(args.log_level)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    # Initialize pipeline
    logging.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(
        model_type=args.model_type,
        model_path=args.model_path,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        roi_top_ratio=args.roi_top_ratio,
        plate_conf_thres=args.plate_conf_thres,
        color_layer_model=args.color_layer_model,
        ocr_model=args.ocr_model,
        annotator_preset=args.annotator_preset,
        annotator_types=args.annotator_types,
        box_thickness=args.box_thickness,
        roundness=args.roundness,
        blur_kernel_size=args.blur_kernel_size
    )

    # Infer source type
    source_type = infer_source_type(args.input)
    logging.info(f"Detected source type: {source_type}")

    # Process based on source type
    if source_type == 'image':
        process_single_image(pipeline, args.input, args.output_dir, args.output_mode)

    elif source_type == 'folder':
        process_folder(pipeline, args.input, args.output_dir, args.output_mode)

    elif source_type in ['video', 'rtsp', 'camera']:
        process_video(
            pipeline, args.input, args.output_dir, args.output_mode,
            frame_skip=args.frame_skip,
            save_frame=args.save_frame,
            save_json=args.save_json
        )

    else:
        logging.error(f"Unknown source type for input: {args.input}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ONNX Vehicle and Plate Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  %(prog)s --model-path models/rtdetr.onnx --model-type rtdetr --input data/sample.jpg

  # Video with debug visualization
  %(prog)s --model-path models/yolo11n.onnx --model-type yolo \\
           --input video.mp4 --annotator-preset debug

  # Camera stream
  %(prog)s --model-path models/rtdetr.onnx --model-type rtdetr \\
           --input 0 --source-type camera --output-mode show
        """
    )

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the ONNX detection model')
    parser.add_argument('--model-type', type=str, default='rtdetr',
                        choices=['rtdetr', 'yolo', 'rfdetr'],
                        help='Model type (default: rtdetr)')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--plate-conf-thres', type=float, default=None,
                        help='Specific confidence threshold for plates (default: same as --conf-thres)')

    # Input/Output parameters
    parser.add_argument('--input', type=str, default='data/sample.jpg',
                        help='Path to input image/video/folder or camera ID (default: data/sample.jpg)')
    parser.add_argument('--output-mode', type=str, choices=['save', 'show'], default='save',
                        help='Output mode: save to file or show in window (default: save)')
    parser.add_argument('--output-dir', type=str, default='runs',
                        help='Directory to save output results (default: runs)')

    # Video processing parameters
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Number of frames to skip between processing (default: 0)')
    parser.add_argument('--save-frame', action='store_true',
                        help='Save individual frames for video input')
    parser.add_argument('--save-json', action='store_true',
                        help='Save JSON results for each frame')

    # ROI parameters
    parser.add_argument('--roi-top-ratio', type=float, default=0.5,
                        help='Top ratio of ROI for detection [0.0-1.0] (default: 0.5)')

    # OCR model parameters
    parser.add_argument('--color-layer-model', type=str, default='models/color_layer_20251222.onnx',
                        help='Path to color/layer ONNX model (default: models/color_layer.onnx)')
    parser.add_argument('--ocr-model', type=str, default='models/ocr_20251222.onnx',
                        help='Path to OCR ONNX model (default: models/ocr.onnx)')

    # Visualization parameters
    parser.add_argument('--annotator-preset', type=str, default='standard',
                        choices=['standard', 'lightweight', 'privacy', 'debug', 'high_contrast'],
                        help='Visualization preset (default: standard)')
    parser.add_argument('--annotator-types', type=str, nargs='+', default=None,
                        choices=['box', 'rich_label', 'round_box', 'box_corner', 'circle', 'triangle',
                                'ellipse', 'dot', 'color', 'background_overlay', 'halo',
                                'percentage_bar', 'blur', 'pixelate'],
                        help='Custom annotator types (overrides preset)')
    parser.add_argument('--box-thickness', type=int, default=2,
                        help='Thickness for box annotators (default: 2)')
    parser.add_argument('--roundness', type=float, default=0.3,
                        help='Roundness for round_box annotator [0.0-1.0] (default: 0.3)')
    parser.add_argument('--blur-kernel-size', type=int, default=15,
                        help='Kernel size for blur annotator (default: 15)')

    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')

    args = parser.parse_args()
    main(args)
