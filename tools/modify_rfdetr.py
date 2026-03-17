"""
改造 rfdetr ONNX 模型:
1. batch 维度改为动态
2. 两个输出 concat 成一个 [batch, 300, 9]
3. 输入输出重命名为 input / output
4. 前面加 Resize 节点: 640x640 -> 576x576
5. 烧入 ImageNet 归一化 (mean/std)，使外部只需 /255.0

改造后数据流:
  input [batch,3,640,640] ∈ [0,1]
    → Sub(mean) → Div(std)      # ImageNet 归一化
    → Resize(640→576)           # bilinear
    → 原始 RF-DETR 主干
    → Concat(pred_boxes, pred_logits)
    → output [batch,300,9]
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def modify_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
    graph = model.graph

    # 原始输入节点名
    orig_input_name = graph.input[0].name  # "images"

    # ========== 1. 新的外部输入: [batch, 3, 640, 640] ==========
    new_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 3, 640, 640]
    )

    # ========== 2. ImageNet 归一化常量 ==========
    # mean 和 std 形状为 [1, 3, 1, 1]，用于 NCHW 广播
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

    # Sub 节点: (input - mean)
    sub_node = helper.make_node(
        "Sub",
        inputs=["input", "imagenet_mean"],
        outputs=["normalized_sub"],
        name="pre_imagenet_sub",
    )

    # Div 节点: (input - mean) / std
    div_node = helper.make_node(
        "Div",
        inputs=["normalized_sub", "imagenet_std"],
        outputs=["normalized_div"],
        name="pre_imagenet_div",
    )

    # ========== 3. Resize 节点: 640x640 -> 576x576 ==========
    roi_tensor = numpy_helper.from_array(
        np.array([], dtype=np.float32), name="resize_roi"
    )
    scale_val = 576.0 / 640.0
    scales_tensor = numpy_helper.from_array(
        np.array([1.0, 1.0, scale_val, scale_val], dtype=np.float32),
        name="resize_scales",
    )
    graph.initializer.append(roi_tensor)
    graph.initializer.append(scales_tensor)

    resize_node = helper.make_node(
        "Resize",
        inputs=["normalized_div", "resize_roi", "resize_scales"],
        outputs=[orig_input_name],
        name="pre_resize",
        mode="linear",
        coordinate_transformation_mode="pytorch_half_pixel",
    )

    # 插入到图的最前面 (顺序: Sub → Div → Resize)
    graph.node.insert(0, resize_node)
    graph.node.insert(0, div_node)
    graph.node.insert(0, sub_node)

    # 替换图的输入
    while len(graph.input) > 0:
        graph.input.pop()
    graph.input.append(new_input)

    # ========== 4. Concat 两个输出 ==========
    boxes_name = "pred_boxes"    # [batch, 300, 4]
    logits_name = "pred_logits"  # [batch, 300, 5]

    concat_node = helper.make_node(
        "Concat",
        inputs=[boxes_name, logits_name],
        outputs=["output"],
        name="post_concat",
        axis=2,
    )
    graph.node.append(concat_node)

    # 替换图的输出
    new_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 300, 9]
    )
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.append(new_output)

    # ========== 5. 清理并验证 ==========
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference warning: {e}")

    # 强制设置输出的 batch 维度为动态
    for out in model.graph.output:
        dim0 = out.type.tensor_type.shape.dim[0]
        dim0.ClearField("dim_value")
        dim0.dim_param = "batch"

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved to {output_path}")

    # 打印验证信息
    model2 = onnx.load(output_path)
    print("\n=== Modified Model ===")
    for inp in model2.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}: {dims}")
    for out in model2.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}: {dims}")

    # 统计新增节点
    pre_nodes = [n.name for n in model2.graph.node if n.name.startswith("pre_") or n.name.startswith("post_")]
    print(f"  Added nodes: {pre_nodes}")


if __name__ == "__main__":
    src = "models/rfdetr-medium_上电室内_20260226_optimized_ir10.onnx"
    dst = "models/rfdetr-medium_上电室内_20260226_modified.onnx"
    modify_model(src, dst)
