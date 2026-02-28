#!/usr/bin/env python
"""Classification Dataset Evaluation CLI Tool

Command-line interface for evaluating classification model performance on labeled datasets.
Supports CSV and ImageFolder formats, multiple output formats, and multi-branch models.

Usage:
    # Helmet model evaluation (single-branch, CSV format)
    python tools/eval_cls.py \
        --model-type helmet \
        --model-path models/helmet.onnx \
        --csv-path data/helmet_val.csv \
        --image-dir data/helmet_images/ \
        --branches helmet_missing:0:0=normal,1=helmet_missing \
        --output-format table

    # ColorLayer model evaluation (dual-branch, CSV format)
    python tools/eval_cls.py \
        --model-type color_layer \
        --model-path models/color_layer.onnx \
        --csv-path data/plate_val.csv \
        --image-dir data/plate_images/ \
        --branches color:0:0=black,1=blue,2=green,3=white,4=yellow \
                   layer:1:0=single,1=double \
        --output-format table

    # ImageFolder format evaluation
    python tools/eval_cls.py \
        --model-type helmet \
        --model-path models/helmet.onnx \
        --dataset-dir data/helmet_imagefolder/ \
        --output-format table
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from onnxtools import ClsDatasetEvaluator, setup_logger  # noqa: E402
from onnxtools.eval.eval_cls import BranchConfig  # noqa: E402

logging.getLogger("polygraphy").setLevel(logging.WARNING)


def parse_branch_spec(spec: str) -> BranchConfig:
    """Parse branch specification string into BranchConfig.

    Format: branch_name:branch_index:key1=val1,key2=val2,...
    Or:     column_name:branch_index:key1=val1,key2=val2,...

    Examples:
        "helmet_missing:0:0=normal,1=helmet_missing"
        "color:0:0=black,1=blue,2=green"

    Args:
        spec: Branch specification string

    Returns:
        BranchConfig instance
    """
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid branch spec '{spec}'. "
            f"Expected format: name:index:key1=val1,key2=val2"
        )

    branch_name = parts[0]
    branch_index = int(parts[1])

    # Parse label map
    label_map = {}
    for pair in parts[2].split(','):
        kv = pair.split('=')
        if len(kv) != 2:
            raise ValueError(f"Invalid label mapping '{pair}' in branch spec")
        try:
            key = int(kv[0])
        except ValueError:
            key = kv[0]
        label_map[key] = kv[1]

    return BranchConfig(
        branch_index=branch_index,
        column_name=branch_name,
        label_map=label_map,
        branch_name=branch_name,
    )


def create_model(model_type: str, model_path: str):
    """Create classification model based on type.

    Args:
        model_type: Model type string
        model_path: Path to ONNX model

    Returns:
        BaseClsORT instance
    """
    model_type = model_type.lower()

    if model_type == 'helmet':
        from onnxtools import HelmetORT
        return HelmetORT(model_path)
    elif model_type in ('color_layer', 'colorlayer'):
        from onnxtools import ColorLayerORT
        return ColorLayerORT(model_path)
    elif model_type in ('vehicle_attribute', 'vehicleattribute'):
        from onnxtools import VehicleAttributeORT
        return VehicleAttributeORT(model_path)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported: helmet, color_layer, vehicle_attribute"
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate classification model performance on labeled datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Helmet evaluation with CSV
  %(prog)s --model-type helmet --model-path models/helmet.onnx \\
      --csv-path data/val.csv --image-dir data/images/ \\
      --branches helmet_missing:0:0=normal,1=helmet_missing

  # ImageFolder evaluation
  %(prog)s --model-type helmet --model-path models/helmet.onnx \\
      --dataset-dir data/helmet_val/
        """
    )

    parser.add_argument(
        '--model-type', required=True, type=str,
        choices=['helmet', 'color_layer', 'vehicle_attribute'],
        help='Classification model type'
    )
    parser.add_argument(
        '--model-path', required=True, type=str,
        help='Path to ONNX model file'
    )

    # CSV dataset arguments
    csv_group = parser.add_argument_group('CSV dataset')
    csv_group.add_argument(
        '--csv-path', type=str, default=None,
        help='Path to CSV label file'
    )
    csv_group.add_argument(
        '--image-dir', type=str, default=None,
        help='Directory containing images'
    )
    csv_group.add_argument(
        '--image-column', type=str, default='img_name',
        help='CSV column name for image filename (default: img_name)'
    )
    csv_group.add_argument(
        '--branches', nargs='+', type=str, default=None,
        help='Branch specs: name:index:key1=val1,key2=val2 (multiple allowed)'
    )

    # ImageFolder arguments
    folder_group = parser.add_argument_group('ImageFolder dataset')
    folder_group.add_argument(
        '--dataset-dir', type=str, default=None,
        help='ImageFolder root directory (dataset_dir/class_name/image.jpg)'
    )

    # Common arguments
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Maximum images to evaluate (default: all)'
    )
    parser.add_argument(
        '--output-format', choices=['table', 'json'], default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """Main entry point for classification evaluation CLI."""
    args = parse_args()

    setup_logger(level=args.log_level)

    try:
        # Validate model path
        if not Path(args.model_path).exists():
            logging.error(f"Model file not found: {args.model_path}")
            sys.exit(1)

        # Create model
        logging.info(f"Loading {args.model_type} model: {args.model_path}")
        model = create_model(args.model_type, args.model_path)
        logging.info("Model loaded successfully")

        # Create evaluator
        evaluator = ClsDatasetEvaluator(model)

        # Determine dataset source and evaluate
        if args.csv_path is not None:
            if args.image_dir is None or args.branches is None:
                logging.error("CSV mode requires --image-dir and --branches")
                sys.exit(1)

            if not Path(args.csv_path).exists():
                logging.error(f"CSV file not found: {args.csv_path}")
                sys.exit(1)

            branch_configs = [parse_branch_spec(s) for s in args.branches]

            logging.info(f"CSV path: {args.csv_path}")
            logging.info(f"Image dir: {args.image_dir}")
            logging.info(f"Branches: {[b.branch_name for b in branch_configs]}")

            results = evaluator.evaluate_dataset(
                csv_path=args.csv_path,
                image_dir=args.image_dir,
                branches=branch_configs,
                image_column=args.image_column,
                conf_threshold=args.conf_threshold,
                max_images=args.max_images,
                output_format=args.output_format,
            )

        elif args.dataset_dir is not None:
            if not Path(args.dataset_dir).exists():
                logging.error(f"Dataset dir not found: {args.dataset_dir}")
                sys.exit(1)

            logging.info(f"Dataset dir: {args.dataset_dir}")

            results = evaluator.evaluate_dataset(
                dataset_dir=args.dataset_dir,
                conf_threshold=args.conf_threshold,
                max_images=args.max_images,
                output_format=args.output_format,
            )
        else:
            logging.error("Must provide either --csv-path or --dataset-dir")
            sys.exit(1)

        if not results or results.get('evaluated_samples', 0) == 0:
            logging.warning("No evaluation results generated")
            sys.exit(0)

        logging.info("Evaluation completed successfully")
        logging.info(f"Overall accuracy: {results['overall_accuracy']:.4f}")
        logging.info(f"Evaluated: {results['evaluated_samples']}/{results['total_samples']}")

        sys.exit(0)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid parameter: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
