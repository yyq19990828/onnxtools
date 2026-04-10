"""改造 RF-DETR ONNX 模型

每个改造步骤独立可控:
1. --rename          输入输出重命名为 input / output
2. --dynamic-batch   batch 维度改为动态
3. --normalize       烧入 ImageNet 归一化 (mean/std)，外部只需 /255.0
4. --resize          前插 Resize 节点 (input_size -> model_size)，需配合 --input-hw
5. --concat          两个输出 concat 成一个 [batch, 300, 4+C]
6. --fold            Polygraphy 折叠常量算子，精简计算图

默认全部开启 (等价于生成 Unified 版本)。
传入任意 bool 开关后切换为手动模式，只执行指定的步骤。

改造后数据流 (全部开启):
  input [batch,3,input_h,input_w] ∈ [0,1]
    → Sub(mean) → Div(std)         # --normalize
    → Resize(input→model)          # --resize (仅尺寸不同时)
    → 原始 RF-DETR 主干
    → Concat(pred_boxes, pred_logits)  # --concat
    → fold_constants()             # --fold
    → output [batch,300,4+C]

用法:
    # 全部改造 (默认)
    python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_unified.onnx

    # 仅重命名 + 动态 batch
    python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx --rename --dynamic-batch

    # 仅烧入归一化 (不 concat、不 resize)
    python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx --normalize

    # 仅折叠常量 (不做其他改造)
    python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_fold.onnx --fold

    # 跳过归一化，其余全做
    python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx \\
        --rename --dynamic-batch --resize --concat --fold --input-hw 640 640
"""

import argparse
from typing import Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def modify_model(
    input_path: str,
    output_path: str,
    input_hw: Tuple[int, int] = (640, 640),
    rename: bool = True,
    dynamic_batch: bool = True,
    normalize: bool = True,
    resize: bool = True,
    concat: bool = True,
    fold: bool = True,
):
    """改造 RF-DETR 模型

    Args:
        input_path: 输入 ONNX 模型路径
        output_path: 输出 ONNX 模型路径
        input_hw: 外部输入尺寸 (H, W)
        rename: 重命名输入输出为 input/output
        dynamic_batch: batch 维度改为动态
        normalize: 烧入 ImageNet 归一化
        resize: 前插 Resize 节点
        concat: 两个输出 concat 成一个
        fold: Polygraphy 折叠常量算子
    """
    model = onnx.load(input_path)

    # ========== 折叠常量 (先于改造) ==========
    if fold:
        from polygraphy.backend.onnx import fold_constants
        num_nodes_before = len(model.graph.node)
        model = fold_constants(model)
        num_nodes_after = len(model.graph.node)
        print(f"Fold constants: {num_nodes_before} -> {num_nodes_after} nodes "
              f"(removed {num_nodes_before - num_nodes_after})")

    graph = model.graph

    # ========== 读取原始模型信息 ==========
    orig_input_name = graph.input[0].name
    orig_shape = graph.input[0].type.tensor_type.shape.dim
    model_h, model_w = orig_shape[2].dim_value, orig_shape[3].dim_value
    input_h, input_w = input_hw

    need_resize = resize and (input_h, input_w) != (model_h, model_w)

    print(f"Model size: {model_h}x{model_w}, Input size: {input_h}x{input_w}")
    print(f"Options: rename={rename}, dynamic_batch={dynamic_batch}, "
          f"normalize={normalize}, resize={need_resize}, concat={concat}, fold={fold}")

    # 推断输出维度
    out_dims = {}
    for out in graph.output:
        last_dim = out.type.tensor_type.shape.dim[-1].dim_value
        out_dims[out.name] = last_dim
    total_out_dim = sum(out_dims.values())
    print(f"Output dims: {out_dims}" + (f" -> concat dim: {total_out_dim}" if concat else ""))

    # 确定输入节点名: 如果 rename 则用 "input"，否则保留原名
    ext_input_name = "input" if rename else orig_input_name

    # ========== 构建前置节点链 ==========
    # 数据流: ext_input_name -> [normalize] -> [resize] -> orig_input_name
    prepend_nodes = []
    current_output = ext_input_name

    if normalize:
        # ImageNet 归一化常量
        imagenet_mean = numpy_helper.from_array(
            np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1),
            name="imagenet_mean",
        )
        imagenet_std = numpy_helper.from_array(
            np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1),
            name="imagenet_std",
        )
        graph.initializer.append(imagenet_mean)
        graph.initializer.append(imagenet_std)

        sub_node = helper.make_node(
            "Sub",
            inputs=[current_output, "imagenet_mean"],
            outputs=["normalized_sub"],
            name="pre_imagenet_sub",
        )
        prepend_nodes.append(sub_node)

        # Div 的输出: 如果后面没有 resize 则直接连到原始输入
        div_out = "normalized_div" if need_resize else orig_input_name
        div_node = helper.make_node(
            "Div",
            inputs=["normalized_sub", "imagenet_std"],
            outputs=[div_out],
            name="pre_imagenet_div",
        )
        prepend_nodes.append(div_node)
        current_output = div_out

    if need_resize:
        roi_tensor = numpy_helper.from_array(
            np.array([], dtype=np.float32), name="resize_roi"
        )
        scales_tensor = numpy_helper.from_array(
            np.array([1.0, 1.0, model_h / input_h, model_w / input_w], dtype=np.float32),
            name="resize_scales",
        )
        graph.initializer.append(roi_tensor)
        graph.initializer.append(scales_tensor)

        resize_node = helper.make_node(
            "Resize",
            inputs=[current_output, "resize_roi", "resize_scales"],
            outputs=[orig_input_name],
            name="pre_resize",
            mode="linear",
            coordinate_transformation_mode="pytorch_half_pixel",
        )
        prepend_nodes.append(resize_node)
        current_output = orig_input_name

    # 如果有前置节点但最后的输出不是原始输入名 (比如仅 rename 没有 normalize/resize)
    if not normalize and not need_resize and ext_input_name != orig_input_name:
        identity_node = helper.make_node(
            "Identity",
            inputs=[ext_input_name],
            outputs=[orig_input_name],
            name="pre_rename_input",
        )
        prepend_nodes.append(identity_node)

    # 逆序插入前置节点到图最前面
    for node in reversed(prepend_nodes):
        graph.node.insert(0, node)

    # ========== 替换输入 ==========
    if rename or normalize or need_resize or dynamic_batch:
        actual_h = input_h if (normalize or need_resize) else model_h
        actual_w = input_w if (normalize or need_resize) else model_w
        batch_dim = "batch" if dynamic_batch else (orig_shape[0].dim_value or 1)

        new_input = helper.make_tensor_value_info(
            ext_input_name, TensorProto.FLOAT, [batch_dim, 3, actual_h, actual_w]
        )
        while len(graph.input) > 0:
            graph.input.pop()
        graph.input.append(new_input)

    # ========== Concat 输出 ==========
    if concat:
        boxes_name = "pred_boxes"
        logits_name = "pred_logits"

        concat_node = helper.make_node(
            "Concat",
            inputs=[boxes_name, logits_name],
            outputs=["output" if rename else "concat_output"],
            name="post_concat",
            axis=2,
        )
        graph.node.append(concat_node)

        out_name = "output" if rename else "concat_output"
        new_output = helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, ["batch", 300, total_out_dim]
        )
        while len(graph.output) > 0:
            graph.output.pop()
        graph.output.append(new_output)
    elif rename:
        # 不 concat 但要 rename 输出
        orig_outputs = list(graph.output)
        for i, orig_out in enumerate(orig_outputs):
            old_name = orig_out.name
            new_name = f"output_{i}" if len(orig_outputs) > 1 else "output"
            if old_name != new_name:
                rename_node = helper.make_node(
                    "Identity",
                    inputs=[old_name],
                    outputs=[new_name],
                    name=f"post_rename_output_{i}",
                )
                graph.node.append(rename_node)

                out_shape = [
                    d.dim_param if d.dim_param else d.dim_value
                    for d in orig_out.type.tensor_type.shape.dim
                ]
                new_out = helper.make_tensor_value_info(
                    new_name, TensorProto.FLOAT, out_shape
                )
                graph.output.remove(orig_out)
                graph.output.append(new_out)

    # ========== 动态 batch ==========
    if dynamic_batch:
        for out in model.graph.output:
            dim0 = out.type.tensor_type.shape.dim[0]
            dim0.ClearField("dim_value")
            dim0.dim_param = "batch"

    # ========== 验证并保存 ==========
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference warning: {e}")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved to {output_path}")
    _print_model_info(output_path)


def _print_model_info(path: str):
    """打印模型输入输出信息"""
    model = onnx.load(path)
    print("\n=== Modified Model ===")
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}: {dims}")
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}: {dims}")

    added = [n.name for n in model.graph.node
             if n.name.startswith("pre_") or n.name.startswith("post_")]
    if added:
        print(f"  Added nodes: {added}")


def main():
    parser = argparse.ArgumentParser(
        description="改造 RF-DETR ONNX 模型 (每步独立可控)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全部改造 (默认，等价于 --rename --dynamic-batch --normalize --resize --concat)
  python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_unified.onnx

  # 仅重命名 + 动态 batch
  python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx --rename --dynamic-batch

  # 仅烧入归一化
  python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx --normalize

  # Resize + Concat (不烧入归一化)
  python tools/modify_rfdetr.py -i rfdetr.onnx -o rfdetr_mod.onnx \\
      --resize --concat --input-hw 640 640
    """,
    )
    parser.add_argument("-i", "--input", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("-o", "--output", required=True, help="输出 ONNX 模型路径")
    parser.add_argument("--input-hw", type=int, nargs=2, default=[640, 640],
                        metavar=("H", "W"), help="外部输入尺寸 (默认: 640 640)")

    # 每个步骤独立的 bool 开关
    parser.add_argument("--rename", action="store_true",
                        help="重命名输入输出为 input/output")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="batch 维度改为动态")
    parser.add_argument("--normalize", action="store_true",
                        help="烧入 ImageNet 归一化 (mean/std)")
    parser.add_argument("--resize", action="store_true",
                        help="前插 Resize 节点 (input_hw -> model_hw)")
    parser.add_argument("--concat", action="store_true",
                        help="两个输出 Concat 成单输出 [batch,300,4+C]")
    parser.add_argument("--fold", action="store_true",
                        help="Polygraphy 折叠常量算子，精简计算图")

    args = parser.parse_args()

    # 如果没有指定任何 bool 开关，默认全部开启
    flags = [args.rename, args.dynamic_batch, args.normalize, args.resize, args.concat, args.fold]
    if not any(flags):
        args.rename = True
        args.dynamic_batch = True
        args.normalize = True
        args.resize = True
        args.concat = True
        args.fold = True

    modify_model(
        input_path=args.input,
        output_path=args.output,
        input_hw=tuple(args.input_hw),
        rename=args.rename,
        dynamic_batch=args.dynamic_batch,
        normalize=args.normalize,
        resize=args.resize,
        concat=args.concat,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()
