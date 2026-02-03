#!/usr/bin/env python3
"""
使用onnx-graphsurgeon修改ONNX模型的输入输出名字
将models/ocr_mobile.onnx的输入输出名字改为input和output
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
except ImportError as e:
    print(f"错误: 缺少必要的依赖包: {e}")
    print("请安装: pip install onnx onnx-graphsurgeon")
    sys.exit(1)


def modify_onnx_io_names(input_path: str, output_path: str, input_name: str = "input", output_name: str = "output"):
    """
    修改ONNX模型的输入输出名字

    Args:
        input_path: 输入ONNX模型路径
        output_path: 输出ONNX模型路径
        input_name: 新的输入名字
        output_name: 新的输出名字
    """
    print(f"正在加载ONNX模型: {input_path}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 使用onnx-graphsurgeon加载模型
    graph = gs.import_onnx(onnx.load(input_path))

    print(f"模型加载成功!")
    print(f"原始输入节点数量: {len(graph.inputs)}")
    print(f"原始输出节点数量: {len(graph.outputs)}")

    # 显示原始输入输出信息
    print("\n原始输入信息:")
    for i, inp in enumerate(graph.inputs):
        print(f"  输入 {i}: 名字='{inp.name}', 形状={inp.shape}, 类型={inp.dtype}")

    print("\n原始输出信息:")
    for i, out in enumerate(graph.outputs):
        print(f"  输出 {i}: 名字='{out.name}', 形状={out.shape}, 类型={out.dtype}")

    # 修改输入名字
    if len(graph.inputs) > 0:
        old_input_name = graph.inputs[0].name
        graph.inputs[0].name = input_name
        print(f"\n修改输入名字: '{old_input_name}' -> '{input_name}'")
    else:
        print("\n警告: 模型没有输入节点")

    # 修改输出名字
    if len(graph.outputs) > 0:
        old_output_name = graph.outputs[0].name
        graph.outputs[0].name = output_name
        print(f"修改输出名字: '{old_output_name}' -> '{output_name}'")
    else:
        print("警告: 模型没有输出节点")

    # 如果有多个输入输出，给出提示
    if len(graph.inputs) > 1:
        print(f"\n注意: 模型有 {len(graph.inputs)} 个输入，只修改了第一个")
        print("其他输入:")
        for i, inp in enumerate(graph.inputs[1:], 1):
            print(f"  输入 {i}: 名字='{inp.name}'")

    if len(graph.outputs) > 1:
        print(f"\n注意: 模型有 {len(graph.outputs)} 个输出，只修改了第一个")
        print("其他输出:")
        for i, out in enumerate(graph.outputs[1:], 1):
            print(f"  输出 {i}: 名字='{out.name}'")

    # 清理图并导出
    graph.cleanup().toposort()

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 导出修改后的模型
    print(f"\n正在保存修改后的模型到: {output_path}")
    onnx.save(gs.export_onnx(graph), output_path)

    print("模型修改完成!")

    # 验证修改后的模型
    print("\n验证修改后的模型:")
    modified_model = onnx.load(output_path)

    print("修改后的输入信息:")
    for i, inp in enumerate(modified_model.graph.input):
        print(f"  输入 {i}: 名字='{inp.name}'")

    print("修改后的输出信息:")
    for i, out in enumerate(modified_model.graph.output):
        print(f"  输出 {i}: 名字='{out.name}'")


def main():
    parser = argparse.ArgumentParser(
        description="修改ONNX模型的输入输出名字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 修改ocr_mobile.onnx的输入输出名字为input和output
  python tools/modify_onnx_io_names.py

  # 指定自定义的输入输出文件
  python tools/modify_onnx_io_names.py -i models/my_model.onnx -o models/my_model_modified.onnx

  # 指定自定义的输入输出名字
  python tools/modify_onnx_io_names.py --input-name "data" --output-name "result"
        """
    )

    parser.add_argument(
        "-i", "--input",
        default="models/ocr_mobile.onnx",
        help="输入ONNX模型路径 (默认: models/ocr_mobile.onnx)"
    )

    parser.add_argument(
        "-o", "--output",
        default="models/ocr_mobile_modified.onnx",
        help="输出ONNX模型路径 (默认: models/ocr_mobile_modified.onnx)"
    )

    parser.add_argument(
        "--input-name",
        default="input",
        help="新的输入名字 (默认: input)"
    )

    parser.add_argument(
        "--output-name",
        default="output",
        help="新的输出名字 (默认: output)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，是否覆盖"
    )

    args = parser.parse_args()

    # 检查输出文件是否已存在
    if os.path.exists(args.output) and not args.overwrite:
        response = input(f"输出文件 '{args.output}' 已存在，是否覆盖? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return

    try:
        modify_onnx_io_names(
            input_path=args.input,
            output_path=args.output,
            input_name=args.input_name,
            output_name=args.output_name
        )

        print(f"\n✅ 成功修改模型输入输出名字!")
        print(f"   输入文件: {args.input}")
        print(f"   输出文件: {args.output}")
        print(f"   输入名字: {args.input_name}")
        print(f"   输出名字: {args.output_name}")

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
