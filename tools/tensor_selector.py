#!/usr/bin/env python3
"""
张量选择器 - 简单函数库

提供简单的函数来选择ONNX模型中的指定张量，支持：
- 模式匹配 (pattern)
- 索引选择 ([1,2,3])  
- 前N层选择 (20)

可以被其他脚本导入使用。
"""

import re
from typing import List, Union
from polygraphy.backend.onnx import OnnxFromPath
from polygraphy.backend.onnx.util import all_tensor_names


def get_model_tensors(model_path: str) -> List[str]:
    """
    获取模型中所有可用的张量名称
    
    Args:
        model_path: ONNX模型文件路径
        
    Returns:
        张量名称列表
    """
    try:
        loader = OnnxFromPath(model_path)
        model = loader()
        return all_tensor_names(model, include_inputs=False)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")


def select_tensors(model_path: str, selector: Union[str, List[int], int]) -> List[str]:
    """
    根据不同方式选择张量
    
    Args:
        model_path: ONNX模型文件路径
        selector: 选择器，支持三种格式：
            - str: 模式匹配，如 "stem*", "*conv*", "*output*"
            - List[int]: 索引列表，如 [1,2,3,5] (1-based索引)
            - int: 前N个张量，如 20
            
    Returns:
        选择的张量名称列表
        
    Examples:
        >>> select_tensors("model.onnx", "stem*")        # 模式匹配
        >>> select_tensors("model.onnx", [1,2,3,5])     # 指定索引
        >>> select_tensors("model.onnx", 20)            # 前20个
    """
    tensor_names = get_model_tensors(model_path)
    
    if isinstance(selector, str):
        # 模式匹配
        return _select_by_pattern(tensor_names, selector)
    elif isinstance(selector, list):
        # 索引列表
        return _select_by_indices(tensor_names, selector)
    elif isinstance(selector, int):
        # 前N个
        return _select_first_n(tensor_names, selector)
    else:
        raise ValueError(f"不支持的选择器类型: {type(selector)}")


def _select_by_pattern(tensor_names: List[str], pattern: str) -> List[str]:
    """根据模式选择张量"""
    try:
        # 将通配符转换为正则表达式
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
        
        selected = []
        for name in tensor_names:
            if compiled_pattern.search(name):
                selected.append(name)
        
        return selected
    except Exception as e:
        raise ValueError(f"模式匹配失败: {e}")


def _select_by_indices(tensor_names: List[str], indices: List[int]) -> List[str]:
    """根据索引选择张量 (1-based)"""
    selected = []
    for idx in indices:
        if 1 <= idx <= len(tensor_names):
            selected.append(tensor_names[idx - 1])  # 转换为0-based
        else:
            print(f"警告: 索引 {idx} 超出范围 (1-{len(tensor_names)})")
    
    return selected


def _select_first_n(tensor_names: List[str], n: int) -> List[str]:
    """选择前N个张量"""
    if n <= 0:
        return []
    return tensor_names[:n]


def generate_polygraphy_command(model_path: str, selected_tensors: List[str], 
                               backend: str = "onnx") -> str:
    """
    生成Polygraphy命令行
    
    Args:
        model_path: 模型路径
        selected_tensors: 选择的张量列表
        backend: 后端类型 ("onnx" 或 "trt")
        
    Returns:
        Polygraphy命令字符串
    """
    if not selected_tensors:
        return ""
    
    backend_flag = "--onnx-outputs" if backend == "onnx" else "--trt-outputs"
    tensor_args = ' \\\n    '.join(f'"{name}"' for name in selected_tensors)
    
    return f"""polygraphy run {model_path} \\
  {backend_flag} \\
    {tensor_args}"""


def print_selection_summary(model_path: str, selector, selected_tensors: List[str]):
    """打印选择摘要"""
    print(f"\n📊 张量选择摘要")
    print(f"模型: {model_path}")
    print(f"选择器: {selector}")
    print(f"选中张量数: {len(selected_tensors)}")
    
    if selected_tensors:
        print(f"\n选中的张量:")
        for i, name in enumerate(selected_tensors, 1):
            print(f"  {i:2d}. {name}")


# 示例和测试函数
def main():
    """命令行测试接口"""
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python tensor_selector.py <model_path> <selector>")
        print("示例:")
        print("  python tensor_selector.py model.onnx 'stem*'")
        print("  python tensor_selector.py model.onnx '[1,2,3,5]'")  
        print("  python tensor_selector.py model.onnx '20'")
        return
    
    model_path = sys.argv[1]
    selector_str = sys.argv[2]
    
    try:
        # 解析选择器
        if selector_str.startswith('[') and selector_str.endswith(']'):
            # 索引列表: [1,2,3,5]
            indices_str = selector_str[1:-1]
            selector = [int(x.strip()) for x in indices_str.split(',')]
        elif selector_str.isdigit():
            # 数字: 20
            selector = int(selector_str)
        else:
            # 模式: stem*
            selector = selector_str
        
        # 选择张量
        selected = select_tensors(model_path, selector)
        
        # 打印结果
        print_selection_summary(model_path, selector, selected)
        
        # 生成命令
        if selected:
            print(f"\n🎯 生成的Polygraphy命令:")
            print(generate_polygraphy_command(model_path, selected, "onnx"))
    
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()