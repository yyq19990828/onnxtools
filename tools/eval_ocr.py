#!/usr/bin/env python
"""OCR Dataset Evaluation CLI Tool

Command-line interface for evaluating OCR model performance on labeled datasets.
Supports multiple output formats, confidence thresholds, and detailed analysis.

Usage:
    python tools/eval_ocr.py \\
        --label-file data/ocr_rec_dataset_examples/val.txt \\
        --dataset-base data/ocr_rec_dataset_examples \\
        --ocr-model models/ocr.onnx \\
        --config configs/plate.yaml \\
        --conf-threshold 0.5 \\
        --output-format table

    # JSON export for model comparison
    python tools/eval_ocr.py \\
        --label-file data/val.txt \\
        --dataset-base data/ \\
        --ocr-model models/ocr_v2.onnx \\
        --config configs/plate.yaml \\
        --output-format json > results_v2.json
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infer_onnx import OCRDatasetEvaluator, OCRONNX, SampleEvaluation
from utils.logging_config import setup_logger
import logging
logging.getLogger("polygraphy").setLevel(logging.WARNING)


def analyze_errors(results: Dict[str, Any], top_n: int = 10) -> None:
    """Analyze and display error cases using SampleEvaluation objects

    Args:
        results: Evaluation results dictionary containing per_sample_results
        top_n: Number of worst cases to display
    """
    if 'per_sample_results' not in results:
        print("\n⚠️ No per-sample results available for error analysis")
        return

    # Convert dict results to SampleEvaluation objects for type safety
    per_sample_objs = [
        SampleEvaluation(**sample) for sample in results['per_sample_results']
    ]

    # Separate correct and incorrect samples
    correct_samples = [s for s in per_sample_objs if s.is_correct]
    incorrect_samples = [s for s in per_sample_objs if not s.is_correct]

    print("\n" + "="*80)
    print("📊 Error Analysis Report")
    print("="*80)

    # Overall statistics
    total = len(per_sample_objs)
    print(f"\n✅ Correct predictions: {len(correct_samples)}/{total} "
          f"({len(correct_samples)/total*100:.2f}%)")
    print(f"❌ Incorrect predictions: {len(incorrect_samples)}/{total} "
          f"({len(incorrect_samples)/total*100:.2f}%)")

    if not incorrect_samples:
        print("\n🎉 Perfect accuracy! No errors to analyze.")
        return

    # Sort by normalized edit distance (worst first)
    incorrect_samples.sort(key=lambda x: x.normalized_edit_distance, reverse=True)

    # Display top error cases
    print(f"\n🔍 Top {min(top_n, len(incorrect_samples))} Worst Error Cases:")
    print("-"*80)

    for i, sample in enumerate(incorrect_samples[:top_n], 1):
        print(f"\n[{i}] Image: {Path(sample.image_path).name}")
        print(f"    Ground Truth:  '{sample.ground_truth}'")
        print(f"    Predicted:     '{sample.predicted_text}'")
        print(f"    Confidence:    {sample.confidence:.3f}")
        print(f"    Edit Distance: {sample.edit_distance} "
              f"(normalized: {sample.normalized_edit_distance:.3f})")
        print(f"    Similarity:    {1.0 - sample.normalized_edit_distance:.3f}")

        # Highlight character differences
        if len(sample.ground_truth) == len(sample.predicted_text):
            diff_positions = [
                i for i, (gt_c, pred_c) in
                enumerate(zip(sample.ground_truth, sample.predicted_text))
                if gt_c != pred_c
            ]
            if diff_positions:
                print(f"    Diff positions: {diff_positions}")

    # Error type analysis
    print("\n" + "="*80)
    print("📈 Error Type Distribution")
    print("="*80)

    error_types = analyze_error_types(incorrect_samples)

    print(f"\n🔢 Length Errors: {error_types['length_errors']} cases")
    print(f"   - Too short: {error_types['too_short']}")
    print(f"   - Too long:  {error_types['too_long']}")

    print(f"\n📝 Character Errors: {error_types['char_errors']} cases")
    print(f"   - Single char:   {error_types['single_char_error']}")
    print(f"   - Multiple chars: {error_types['multi_char_error']}")

    print(f"\n⚡ Low Confidence: {error_types['low_confidence']} cases (< 0.7)")

    # Average metrics for errors
    avg_conf = sum(s.confidence for s in incorrect_samples) / len(incorrect_samples)
    avg_ed = sum(s.normalized_edit_distance for s in incorrect_samples) / len(incorrect_samples)

    print(f"\n📊 Error Statistics:")
    print(f"   - Average confidence: {avg_conf:.3f}")
    print(f"   - Average normalized edit distance: {avg_ed:.3f}")


def analyze_error_types(samples: List[SampleEvaluation]) -> Dict[str, int]:
    """Categorize error types using SampleEvaluation objects

    Args:
        samples: List of SampleEvaluation objects for incorrect predictions

    Returns:
        Dictionary with error type counts
    """
    error_types = {
        'length_errors': 0,
        'too_short': 0,
        'too_long': 0,
        'char_errors': 0,
        'single_char_error': 0,
        'multi_char_error': 0,
        'low_confidence': 0
    }

    for sample in samples:
        gt_len = len(sample.ground_truth)
        pred_len = len(sample.predicted_text)

        # Length errors
        if gt_len != pred_len:
            error_types['length_errors'] += 1
            if pred_len < gt_len:
                error_types['too_short'] += 1
            else:
                error_types['too_long'] += 1

        # Character errors
        if sample.edit_distance == 1:
            error_types['single_char_error'] += 1
        elif sample.edit_distance > 1:
            error_types['multi_char_error'] += 1
        error_types['char_errors'] += 1

        # Low confidence
        if sample.confidence < 0.7:
            error_types['low_confidence'] += 1

    return error_types


def analyze_confidence_distribution(results: Dict[str, Any]) -> None:
    """Analyze confidence score distribution using SampleEvaluation objects

    Args:
        results: Evaluation results dictionary
    """
    if 'per_sample_results' not in results:
        return

    # Convert to SampleEvaluation objects
    per_sample_objs = [
        SampleEvaluation(**sample) for sample in results['per_sample_results']
    ]

    print("\n" + "="*80)
    print("📊 Confidence Distribution Analysis")
    print("="*80)

    # Separate by correctness
    correct_samples = [s for s in per_sample_objs if s.is_correct]
    incorrect_samples = [s for s in per_sample_objs if not s.is_correct]

    if correct_samples:
        correct_confs = [s.confidence for s in correct_samples]
        print(f"\n✅ Correct predictions ({len(correct_samples)} samples):")
        print(f"   - Average confidence: {sum(correct_confs)/len(correct_confs):.3f}")
        print(f"   - Min confidence:     {min(correct_confs):.3f}")
        print(f"   - Max confidence:     {max(correct_confs):.3f}")

    if incorrect_samples:
        incorrect_confs = [s.confidence for s in incorrect_samples]
        print(f"\n❌ Incorrect predictions ({len(incorrect_samples)} samples):")
        print(f"   - Average confidence: {sum(incorrect_confs)/len(incorrect_confs):.3f}")
        print(f"   - Min confidence:     {min(incorrect_confs):.3f}")
        print(f"   - Max confidence:     {max(incorrect_confs):.3f}")

    # Confidence ranges
    ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\n📊 Confidence Range Distribution:")

    for low, high in ranges:
        in_range = [s for s in per_sample_objs if low <= s.confidence < high]
        correct_in_range = [s for s in in_range if s.is_correct]

        if in_range:
            accuracy = len(correct_in_range) / len(in_range) * 100
            print(f"   [{low:.1f}, {high:.1f}): {len(in_range):3d} samples, "
                  f"accuracy: {accuracy:5.1f}%")


def find_common_mistakes(results: Dict[str, Any], top_n: int = 5) -> None:
    """Find and display common mistake patterns using SampleEvaluation

    Args:
        results: Evaluation results dictionary
        top_n: Number of top patterns to show
    """
    if 'per_sample_results' not in results:
        return

    per_sample_objs = [
        SampleEvaluation(**sample) for sample in results['per_sample_results']
    ]
    incorrect_samples = [s for s in per_sample_objs if not s.is_correct]

    if not incorrect_samples:
        return

    print("\n" + "="*80)
    print("🔍 Common Mistake Patterns")
    print("="*80)

    # Collect character substitution patterns
    substitutions = {}
    for sample in incorrect_samples:
        if len(sample.ground_truth) == len(sample.predicted_text):
            for gt_c, pred_c in zip(sample.ground_truth, sample.predicted_text):
                if gt_c != pred_c:
                    key = f"{gt_c} → {pred_c}"
                    substitutions[key] = substitutions.get(key, 0) + 1

    if substitutions:
        print(f"\n📝 Top {top_n} Character Substitution Errors:")
        sorted_subs = sorted(substitutions.items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_subs[:top_n], 1):
            print(f"   {i}. '{pattern}': {count} occurrences")


def save_detailed_report(results: Dict[str, Any], output_path: str) -> None:
    """Save detailed error analysis report to JSON using SampleEvaluation

    Args:
        results: Evaluation results dictionary
        output_path: Output JSON file path
    """
    if 'per_sample_results' not in results:
        logging.warning("Cannot save report: no per-sample results available")
        return

    # Convert to SampleEvaluation objects
    per_sample_objs = [
        SampleEvaluation(**sample) for sample in results['per_sample_results']
    ]
    incorrect_samples = [s for s in per_sample_objs if not s.is_correct]

    # Convert back to dict for JSON serialization
    report = {
        'summary': {
            'total_samples': results['total_samples'],
            'evaluated_samples': results['evaluated_samples'],
            'filtered_samples': results['filtered_samples'],
            'skipped_samples': results['skipped_samples'],
            'accuracy': results['accuracy'],
            'normalized_edit_distance': results['normalized_edit_distance'],
            'edit_distance_similarity': results['edit_distance_similarity'],
            'error_count': len(incorrect_samples),
            'evaluation_time': results['evaluation_time'],
            'avg_inference_time_ms': results['avg_inference_time_ms']
        },
        'error_cases': [
            {
                'image_path': s.image_path,
                'ground_truth': s.ground_truth,
                'predicted_text': s.predicted_text,
                'confidence': s.confidence,
                'edit_distance': s.edit_distance,
                'normalized_edit_distance': s.normalized_edit_distance
            }
            for s in sorted(
                incorrect_samples,
                key=lambda x: x.normalized_edit_distance,
                reverse=True
            )
        ],
        'error_types': analyze_error_types(incorrect_samples) if incorrect_samples else {}
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"Detailed error report saved to: {output_path}")
    print(f"\n💾 Detailed error report saved to: {output_path}")


def load_character_dict(config_path: str) -> list:
    """Load character dictionary from configuration file

    Args:
        config_path: Path to plate.yaml configuration file

    Returns:
        List of characters for OCR model

    Raises:
        FileNotFoundError: If config file not found
        KeyError: If required keys missing in config
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        plate_yaml = yaml.safe_load(f)

    if 'ocr_dict' not in plate_yaml:
        raise KeyError("'ocr_dict' not found in config file")

    character = ["blank"] + plate_yaml["ocr_dict"] + [" "]
    logging.info(f"Loaded character dictionary: {len(character)} characters")
    return character


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate OCR model performance on labeled datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with table output
  %(prog)s --label-file data/val.txt --dataset-base data/ \\
      --ocr-model models/ocr.onnx --config configs/plate.yaml

  # High confidence threshold with JSON export
  %(prog)s --label-file data/val.txt --dataset-base data/ \\
      --ocr-model models/ocr.onnx --config configs/plate.yaml \\
      --conf-threshold 0.9 --output-format json

  # Quick test on first 100 images
  %(prog)s --label-file data/val.txt --dataset-base data/ \\
      --ocr-model models/ocr.onnx --config configs/plate.yaml \\
      --max-images 100

  # Save results to file
  %(prog)s --label-file data/val.txt --dataset-base data/ \\
      --ocr-model models/ocr.onnx --config configs/plate.yaml \\
      --output-format json > evaluation_results.json
        """
    )

    # Required arguments
    parser.add_argument(
        '--label-file',
        required=True,
        type=str,
        help='Path to label file (tab-separated format: image_path<TAB>ground_truth)'
    )

    parser.add_argument(
        '--dataset-base',
        required=True,
        type=str,
        help='Dataset root directory (for resolving relative image paths)'
    )

    parser.add_argument(
        '--ocr-model',
        required=True,
        type=str,
        help='Path to ONNX OCR model file'
    )

    parser.add_argument(
        '--config',
        default='configs/plate.yaml',
        type=str,
        help='Path to configuration file (plate.yaml) containing character dictionary'
    )

    # Optional arguments
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for filtering predictions (default: 0.5)'
    )

    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to evaluate (default: all)'
    )

    parser.add_argument(
        '--output-format',
        choices=['table', 'json'],
        default='table',
        help='Output format: table (human-readable) or json (machine-readable) (default: table)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    # Error analysis argument
    parser.add_argument(
        '--error-analysis',
        type=str,
        default=None,
        metavar='REPORT_FILE',
        help='Enable deep error analysis and save detailed report to JSON file (e.g., error_report.json)'
    )

    return parser.parse_args()


def main():
    """Main entry point for OCR evaluation CLI"""
    args = parse_args()

    # Setup logging
    setup_logger(level=args.log_level)

    try:
        # Validate paths
        if not Path(args.label_file).exists():
            logging.error(f"Label file not found: {args.label_file}")
            sys.exit(1)

        if not Path(args.dataset_base).exists():
            logging.error(f"Dataset base directory not found: {args.dataset_base}")
            sys.exit(1)

        if not Path(args.ocr_model).exists():
            logging.error(f"OCR model file not found: {args.ocr_model}")
            sys.exit(1)

        if not Path(args.config).exists():
            logging.error(f"Config file not found: {args.config}")
            sys.exit(1)

        # Load character dictionary
        logging.info("Loading configuration...")
        character = load_character_dict(args.config)

        # Load OCR model
        logging.info(f"Loading OCR model: {args.ocr_model}")
        ocr_model = OCRONNX(args.ocr_model, character=character)
        logging.info("OCR model loaded successfully")

        # Create evaluator
        evaluator = OCRDatasetEvaluator(ocr_model)

        # Run evaluation
        logging.info("Starting dataset evaluation...")
        logging.info(f"Label file: {args.label_file}")
        logging.info(f"Dataset base: {args.dataset_base}")
        logging.info(f"Confidence threshold: {args.conf_threshold}")
        logging.info(f"Max images: {args.max_images if args.max_images else 'all'}")
        logging.info(f"Output format: {args.output_format}")

        results = evaluator.evaluate_dataset(
            label_file=args.label_file,
            dataset_base_path=args.dataset_base,
            conf_threshold=args.conf_threshold,
            max_images=args.max_images,
            output_format=args.output_format
        )

        if not results:
            logging.warning("No evaluation results generated (empty dataset or all samples filtered)")
            sys.exit(0)

        # Success
        logging.info("Evaluation completed successfully")
        logging.info(f"Total samples: {results['total_samples']}")
        logging.info(f"Evaluated samples: {results['evaluated_samples']}")
        logging.info(f"Filtered samples: {results['filtered_samples']}")
        logging.info(f"Accuracy: {results['accuracy']:.3f}")

        # Deep error analysis if requested
        if args.error_analysis:
            logging.info("Running deep error analysis...")
            analyze_errors(results, top_n=15)
            analyze_confidence_distribution(results)
            find_common_mistakes(results, top_n=10)
            
            # Save detailed report
            logging.info(f"Saving detailed report to {args.error_analysis}...")
            save_detailed_report(results, args.error_analysis)

        sys.exit(0)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid parameter: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during evaluation: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
