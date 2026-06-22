#!/usr/bin/env python
"""MOT (多目标跟踪) 数据集评估 CLI。

在 MOTChallenge 格式数据集上评估跟踪结果，输出 HOTA / MOTA / IDF1 等指标。

两种预测来源二选一：

* ``--predictions <dir>``：已有跟踪器输出（每序列一个 ``<seq>.txt``，MOTChallenge
  result 格式）。
* ``--tracker <algo>``：把 GT 框当作"理想检测"现场跑指定跟踪后端（bytetrack /
  bytetrack_native / ocsort），度量纯关联质量，无需检测器。

Usage:
    # 评估已有跟踪结果目录
    python tools/eval/eval_mot.py \\
        --gt-root data/track/MOT_dataset \\
        --predictions runs/my_tracker_outputs

    # 用 GT 框现场跑某个跟踪后端并评估（对比三种后端的关联质量）
    python tools/eval/eval_mot.py \\
        --gt-root data/track/MOT_dataset \\
        --tracker bytetrack_native \\
        --frame-rate 5

    # 只算 HOTA，按类别 1 (pedestrian) 单独评估，导出 JSON
    python tools/eval/eval_mot.py \\
        --gt-root data/track/MOT_dataset \\
        --tracker ocsort \\
        --metrics hota \\
        --classes 1 \\
        --output runs/mot_eval/ocsort_ped.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from onnxtools.eval import MOTEvaluator, run_tracker_on_gt  # noqa: E402
from onnxtools.utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MOTChallenge 格式多目标跟踪评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gt-root",
        required=True,
        help="GT 数据集根目录（含 <SEQ>/gt/gt.txt、seqinfo.ini、可选 seqmap.txt）",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--predictions",
        help="预测结果目录（每序列一个 <seq>.txt，MOTChallenge result 格式）",
    )
    src.add_argument(
        "--tracker",
        choices=("bytetrack", "bytetrack_native", "ocsort"),
        help="用 GT 框作为理想检测，现场跑该跟踪后端生成预测",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["clear", "identity", "hota"],
        choices=("clear", "identity", "hota"),
        help="要计算的指标族",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="CLEAR/Identity 匹配的最小 IoU（HOTA 不受影响）",
    )
    p.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="仅评估这些类别 id；缺省则池化全部类别（标准单类 MOT）",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="过滤置信度低于该值的预测框（0 表示不过滤）",
    )
    p.add_argument(
        "--frame-rate",
        type=int,
        default=None,
        help="--tracker 模式下传给跟踪器的帧率；缺省用各序列 seqinfo 的 frameRate",
    )
    p.add_argument("--output", help="把结果（含 OVERALL）写为 JSON 文件")
    p.add_argument("--log-level", default="INFO", help="日志级别")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logger(args.log_level)
    logging.getLogger("polygraphy").setLevel(logging.WARNING)

    evaluator = MOTEvaluator(args.gt_root)

    if args.tracker:
        tracker_kwargs = {}
        if args.frame_rate is not None:
            tracker_kwargs["frame_rate"] = args.frame_rate
        predictions = run_tracker_on_gt(args.gt_root, args.tracker, **tracker_kwargs)
        source = f"tracker={args.tracker}"
    else:
        predictions = args.predictions
        source = f"predictions={args.predictions}"

    result = evaluator.evaluate(
        predictions,
        iou_threshold=args.iou_threshold,
        metrics=tuple(args.metrics),
        classes=args.classes,
        conf_threshold=args.conf_threshold,
    )

    print(f"\nMOT 评估结果 ({source}, gt={args.gt_root})\n")
    print(result.summary_table())
    print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
