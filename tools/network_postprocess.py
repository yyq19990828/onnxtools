#!/usr/bin/env python3
"""
TensorRT网络后处理脚本
针对YOLO/RTDETR模型的关键层进行精度优化
优化包括：sigmoid、softmax、normalization、attention、reduction等层
"""

import tensorrt as trt
from typing import Dict, List, Tuple

def _get_layer_operation_type(layer: 'trt.ILayer') -> str:
    """
    检测层的具体操作类型
    
    Args:
        layer: TensorRT层对象
        
    Returns:
        操作类型字符串
    """
    layer_type = layer.type
    layer_name = layer.name.lower()
    
    # Activation层的具体操作检测
    if layer_type == trt.LayerType.ACTIVATION:
        try:
            activation_type = layer.get_activation_type()
            if activation_type == trt.ActivationType.SIGMOID:
                return "sigmoid_activation"
            elif activation_type == trt.ActivationType.TANH:
                return "tanh_activation"
            elif activation_type == trt.ActivationType.RELU:
                return "relu_activation"
        except (AttributeError, TypeError):
            pass
            
        # 通过名称推断
        if any(name in layer_name for name in ['sigmoid', 'sig']):
            return "sigmoid_activation"
        elif any(name in layer_name for name in ['tanh']):
            return "tanh_activation"
        elif any(name in layer_name for name in ['relu', 'leaky']):
            return "relu_activation"
            
    # Unary层的具体操作检测
    elif layer_type == trt.LayerType.UNARY:
        try:
            unary_op = layer.get_unary_operation()
            if unary_op == trt.UnaryOperation.EXP:
                return "exp_unary"
            elif unary_op == trt.UnaryOperation.LOG:
                return "log_unary"
            elif unary_op == trt.UnaryOperation.SQRT:
                return "sqrt_unary"
        except (AttributeError, TypeError):
            pass
            
        # 通过名称推断
        if any(name in layer_name for name in ['exp', 'exponential']):
            return "exp_unary"
        elif any(name in layer_name for name in ['log', 'logarithm']):
            return "log_unary"
        elif any(name in layer_name for name in ['sqrt', 'square_root']):
            return "sqrt_unary"
    
    # ElementWise层的具体操作检测
    elif layer_type == trt.LayerType.ELEMENTWISE:
        try:
            elementwise_op = layer.get_elementwise_operation()
            if elementwise_op == trt.ElementWiseOperation.DIV:
                return "div_elementwise"
            elif elementwise_op == trt.ElementWiseOperation.POW:
                return "pow_elementwise"
        except (AttributeError, TypeError):
            pass
    
    # Reduce层的具体操作检测
    elif layer_type == trt.LayerType.REDUCE:
        try:
            reduce_op = layer.get_reduce_operation()
            if reduce_op == trt.ReduceOperation.MAX:
                return "reduce_max"
            elif reduce_op == trt.ReduceOperation.MIN:
                return "reduce_min"
            elif reduce_op == trt.ReduceOperation.SUM:
                return "reduce_sum"
            elif reduce_op == trt.ReduceOperation.AVG:
                return "reduce_avg"
            elif reduce_op == trt.ReduceOperation.PROD:
                return "reduce_prod"
        except (AttributeError, TypeError):
            pass
            
        # 通过名称推断
        if any(name in layer_name for name in ['reducemax', 'reduce_max']):
            return "reduce_max"
        elif any(name in layer_name for name in ['reducemin', 'reduce_min']):
            return "reduce_min"
    
    return str(layer_type).split('.')[-1].lower()


def _should_use_fp32_precision(layer: 'trt.ILayer', operation_type: str, layer_name: str) -> Tuple[bool, str]:
    """
    判断层是否应该使用FP32精度
    
    Args:
        layer: TensorRT层对象
        operation_type: 操作类型
        layer_name: 层名称
        
    Returns:
        (是否使用FP32, 原因)
    """
    layer_name_lower = layer_name.lower()
    
    # 总是需要FP32的层类型
    critical_operations = {
        trt.LayerType.SOFTMAX: "softmax数值敏感",
        trt.LayerType.NORMALIZATION: "normalization精度要求高",
        trt.LayerType.TOPK: "topk排序算子精度敏感",
        # trt.LayerType.REDUCE: "reduction操作累积误差",  # 暂时注释，太多了
    }
    
    if layer.type in critical_operations:
        return True, critical_operations[layer.type]
    
    # 特定操作类型需要FP32
    if operation_type in ["sigmoid_activation", "exp_unary", "log_unary", "div_elementwise", "reduce_max"]:
        return True, f"{operation_type}数值不稳定"
    
    # 特定路径下的节点强制使用FP32
    critical_path_patterns = [
        '/model.28/enc_score_head/',  # RTDETR encoder score head路径
    ]
    
    for pattern in critical_path_patterns:
        if pattern in layer_name:
            return True, f"关键路径节点: {pattern}"
    
    # 暂时注释掉attention层的FP32设置
    # attention_patterns = ['attention', 'attn', 'self_attn', 'cross_attn']
    # if any(pattern in layer_name_lower for pattern in attention_patterns):
    #     return True, "attention机制精度敏感"
    
    # YOLO/DETR特定层
    # detection_patterns = ['decode', 'postprocess', 'nms', 'score', 'class', 'bbox']
    # if any(pattern in layer_name_lower for pattern in detection_patterns):
    #     return True, "检测后处理精度要求"
    
    return False, ""


def postprocess(network: 'trt.INetworkDefinition') -> 'trt.INetworkDefinition':
    """
    对TensorRT网络进行智能后处理
    
    Args:
        network: TensorRT网络定义
        
    Returns:
        修改后的网络
    """
    print(f"=== TensorRT网络智能后处理开始 ===")
    print(f"网络层数: {network.num_layers}")
    print(f"网络输入数: {network.num_inputs}")
    print(f"网络输出数: {network.num_outputs}")
    
    # 统计信息
    fp32_stats: Dict[str, List[str]] = {
        'softmax': [],
        'normalization': [],
        'topk': [],
        'reduce': [],  # reduce操作(包含reducemax)
        'sigmoid': [],
        'critical_path': [],  # 关键路径节点
        'attention': [],
        'detection': [],
        'numerical_unstable': [],
        'failed': []
    }
    
    total_layers = network.num_layers
    processed_layers = 0
    
    # 输入输出信息
    print("\n=== 网络输入输出信息 ===")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"输入张量 {i}: {input_tensor.name} - 形状: {input_tensor.shape}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"输出张量 {i}: {output_tensor.name} - 形状: {output_tensor.shape}")
    
    print("\n=== 智能精度优化处理 ===")
    
    # 遍历所有层进行智能处理
    for i in range(total_layers):
        try:
            layer = network.get_layer(i)
            layer_name = layer.name
            operation_type = _get_layer_operation_type(layer)
            
            # 判断是否需要FP32精度
            should_use_fp32, reason = _should_use_fp32_precision(layer, operation_type, layer_name)
            
            if should_use_fp32:
                try:
                    # 设置层精度
                    layer.precision = trt.DataType.FLOAT
                    
                    # 尝试设置输出张量精度
                    for output_idx in range(layer.num_outputs):
                        try:
                            layer.set_output_type(output_idx, trt.DataType.FLOAT)
                        except (AttributeError, RuntimeError):
                            # 有些层不支持设置输出类型
                            pass
                    
                    # 分类统计
                    layer_type = layer.type
                    if '关键路径节点' in reason:
                        fp32_stats['critical_path'].append(layer_name)
                    elif layer_type == trt.LayerType.SOFTMAX:
                        fp32_stats['softmax'].append(layer_name)
                    elif layer_type == trt.LayerType.NORMALIZATION:
                        fp32_stats['normalization'].append(layer_name)
                    elif layer_type == trt.LayerType.TOPK:
                        fp32_stats['topk'].append(layer_name)
                    elif 'reduce' in operation_type:
                        fp32_stats['reduce'].append(layer_name)
                    elif 'sigmoid' in operation_type:
                        fp32_stats['sigmoid'].append(layer_name)
                    # elif 'attention' in reason:
                    #     fp32_stats['attention'].append(layer_name)
                    # elif '检测' in reason:
                    #     fp32_stats['detection'].append(layer_name)
                    else:
                        fp32_stats['numerical_unstable'].append(layer_name)
                    
                    print(f"  ✓ [{i+1}/{total_layers}] {layer_name} ({operation_type}) → FP32: {reason}")
                    
                except Exception as layer_error:
                    fp32_stats['failed'].append(f"{layer_name}: {str(layer_error)}")
                    print(f"  ✗ [{i+1}/{total_layers}] {layer_name} 精度设置失败: {layer_error}")
            
            processed_layers += 1
                    
        except Exception as general_error:
            fp32_stats['failed'].append(f"Layer {i}: {str(general_error)}")
            print(f"  ✗ [{i+1}/{total_layers}] 层 {i} 处理失败: {general_error}")
    
    # 输出详细统计
    print("\n=== 精度优化统计报告 ===")
    total_fp32_layers = sum(len(layers) for key, layers in fp32_stats.items() if key != 'failed')
    
    print(f"处理层数: {processed_layers}/{total_layers}")
    print(f"FP32层总数: {total_fp32_layers}")
    print(f"失败层数: {len(fp32_stats['failed'])}")
    
    # 分类统计
    for category, layers in fp32_stats.items():
        if layers and category != 'failed':
            print(f"  {category.capitalize()}: {len(layers)} 层")
            for layer_name in layers[:3]:  # 只显示前3个
                print(f"    - {layer_name}")
            if len(layers) > 3:
                print(f"    ... 还有 {len(layers) - 3} 层")
    
    # 失败统计
    if fp32_stats['failed']:
        print(f"\n=== 失败层详情 ===")
        for failed in fp32_stats['failed'][:5]:  # 只显示前5个失败
            print(f"  ✗ {failed}")
        if len(fp32_stats['failed']) > 5:
            print(f"  ... 还有 {len(fp32_stats['failed']) - 5} 个失败")
    
    print(f"\n=== TensorRT网络智能后处理完成 ===")
    return network