#!/usr/bin/env python3
"""
图片分类和统计脚本
根据车牌名字符数分类图片，并统计尺寸分布
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['LXGW WenKai', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def extract_plate_name(filename: str) -> str:
    """从文件名中提取车牌名（第一个下划线之前的字符串）"""
    return filename.split("_")[0]


def classify_by_length(plate_name: str) -> str:
    """根据车牌名字符数分类"""
    length = len(plate_name)
    if length >= 7:
        return "7+chars"
    elif 5 <= length <= 6:
        return "5-6chars"
    else:  # length <= 4
        return "<=4chars"


def process_images(
    input_dir: Path, output_dir: Path, dry_run: bool = False
) -> Dict[str, List[Tuple[int, int]]]:
    """
    处理图片：分类并记录尺寸

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        dry_run: 是否只统计不移动文件

    Returns:
        各类别的尺寸列表字典
    """
    # 创建输出目录
    categories = ["7+chars", "5-6chars", "<=4chars"]
    category_dirs = {}
    if not dry_run:
        for category in categories:
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            category_dirs[category] = category_dir

    # 统计数据
    size_data: Dict[str, List[Tuple[int, int]]] = {cat: [] for cat in categories}
    stats = {cat: {"count": 0} for cat in categories}

    # 获取所有图片文件
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"找到 {len(image_files)} 张图片")

    # 处理每张图片
    for img_path in tqdm(image_files, desc="处理图片"):
        try:
            # 提取车牌名并分类
            plate_name = extract_plate_name(img_path.name)
            category = classify_by_length(plate_name)

            # 读取图片尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图片 {img_path.name}")
                continue

            height, width = img.shape[:2]
            size_data[category].append((width, height))
            stats[category]["count"] += 1

            # 移动文件（如果不是dry_run模式）
            if not dry_run:
                dest_path = category_dirs[category] / img_path.name
                shutil.copy2(img_path, dest_path)

        except Exception as e:
            print(f"处理图片 {img_path.name} 时出错: {e}")
            continue

    return size_data


def calculate_statistics(
    size_data: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, Dict[str, float]]:
    """计算统计数据"""
    stats = {}
    for category, sizes in size_data.items():
        if not sizes:
            stats[category] = {
                "count": 0,
                "avg_width": 0.0,
                "avg_height": 0.0,
                "std_width": 0.0,
                "std_height": 0.0,
            }
            continue

        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]

        stats[category] = {
            "count": len(sizes),
            "avg_width": np.mean(widths),
            "avg_height": np.mean(heights),
            "std_width": np.std(widths),
            "std_height": np.std(heights),
            "min_width": np.min(widths),
            "max_width": np.max(widths),
            "min_height": np.min(heights),
            "max_height": np.max(heights),
        }

    return stats


def print_statistics(stats: Dict[str, Dict[str, float]]):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    for category, data in stats.items():
        print(f"\n类别: {category}")
        print(f"  图片数量: {data['count']}")
        if data["count"] > 0:
            print(f"  平均宽度: {data['avg_width']:.2f} ± {data['std_width']:.2f} px")
            print(f"  平均高度: {data['avg_height']:.2f} ± {data['std_height']:.2f} px")
            print(
                f"  宽度范围: [{data['min_width']:.0f}, {data['max_width']:.0f}] px"
            )
            print(
                f"  高度范围: [{data['min_height']:.0f}, {data['max_height']:.0f}] px"
            )


def plot_distribution(
    size_data: Dict[str, List[Tuple[int, int]]], output_path: Path
):
    """绘制尺寸分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("车牌图片尺寸分布", fontsize=16)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    categories = ["7+chars", "5-6chars", "<=4chars"]
    category_names = ["≥7字符", "5-6字符", "≤4字符"]

    for idx, (category, name, color) in enumerate(
        zip(categories, category_names, colors)
    ):
        ax = axes[idx]
        sizes = size_data[category]

        if not sizes:
            ax.text(
                0.5,
                0.5,
                "无数据",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_title(f"{name} (0张)")
            continue

        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]

        # 绘制散点图
        ax.scatter(widths, heights, alpha=0.3, s=10, color=color, edgecolors="none")

        # 绘制平均值点
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        ax.scatter(
            avg_width,
            avg_height,
            color="red",
            s=200,
            marker="x",
            linewidths=3,
            label=f"平均值 ({avg_width:.1f}, {avg_height:.1f})",
        )

        # 设置标签和标题
        ax.set_xlabel("宽度 (px)")
        ax.set_ylabel("高度 (px)")
        ax.set_title(f"{name} ({len(sizes)}张)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 设置坐标轴范围
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n分布图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="车牌图片分类和统计工具")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="runs/test_color_fixed",
        help="输入图片目录 (默认: runs/test_color_fixed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/classified_plates",
        help="输出分类目录 (默认: runs/classified_plates)",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="runs/plate_size_distribution.png",
        help="分布图输出路径 (默认: runs/plate_size_distribution.png)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计不移动文件",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    plot_output = Path(args.plot_output)

    # 验证输入目录
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    # 处理图片
    print(f"输入目录: {input_dir}")
    if not args.dry_run:
        print(f"输出目录: {output_dir}")
    else:
        print("模式: 仅统计（不移动文件）")

    size_data = process_images(input_dir, output_dir, args.dry_run)

    # 计算统计数据
    stats = calculate_statistics(size_data)

    # 打印统计结果
    print_statistics(stats)

    # 绘制分布图
    plot_distribution(size_data, plot_output)

    print("\n处理完成!")


if __name__ == "__main__":
    main()
