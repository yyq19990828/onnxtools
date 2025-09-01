#!/usr/bin/env python3
"""
优化的TensorRT与ONNX Runtime比较脚本
支持灵活的参数配置，可用于任意ONNX模型的TensorRT引擎构建和对比

特性:
- 智能网络后处理，自动识别关键层并设置FP32精度
- 优化的FP16构建配置，减少权重转换警告
- 详细的性能和精度对比报告
- 自动保存引擎文件以供复用
- 支持命令行参数配置

Usage:
    python tools/build_engine.py --onnx-path models/model.onnx [--engine-path models/model.engine] [--compare]
"""

import argparse
import os
import sys
from pathlib import Path

from polygraphy import mod
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineBytesFromNetwork, EngineFromBytes, NetworkFromOnnxPath, TrtRunner, CreateConfig, PostprocessNetwork, SaveEngine
from polygraphy.backend.common import InvokeFromScript
from polygraphy.comparator import Comparator

trt = mod.lazy_import('tensorrt>=8.5')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='构建TensorRT引擎并可选地与ONNX Runtime进行比较',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--onnx-path',
        type=str,
        required=True,
        help='输入的ONNX模型路径'
    )
    
    parser.add_argument(
        '--engine-path',
        type=str,
        default=None,
        help='输出的TensorRT引擎路径（默认与ONNX同名，扩展名为.engine）'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='是否开启ONNX Runtime和TensorRT的精度对比'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='启用FP16精度（默认启用）'
    )
    
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='禁用FP16精度，使用FP32'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help='TensorRT构建优化级别（0-5）'
    )
    
    return parser.parse_args()


def get_engine_path(onnx_path, engine_path):
    """生成引擎路径"""
    if engine_path is None:
        onnx_file = Path(onnx_path)
        engine_path = str(onnx_file.with_suffix('.engine'))
    return engine_path


def validate_paths(onnx_path, engine_path):
    """验证路径有效性"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
    
    engine_dir = os.path.dirname(engine_path)
    if engine_dir and not os.path.exists(engine_dir):
        os.makedirs(engine_dir, exist_ok=True)
        print(f"创建输出目录: {engine_dir}")


def main():
    """主函数"""
    args = parse_args()
    
    # 处理FP16/FP32选项
    if args.no_fp16:
        args.fp16 = False
    
    # 获取引擎路径
    engine_path = get_engine_path(args.onnx_path, args.engine_path)
    
    # 验证路径
    try:
        validate_paths(args.onnx_path, engine_path)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    print(f"ONNX模型路径: {args.onnx_path}")
    print(f"引擎输出路径: {engine_path}")
    print(f"精度对比: {'是' if args.compare else '否'}")
    
    # 构建TensorRT引擎配置
    print("配置优化的TensorRT构建参数...")
    create_config = CreateConfig(
        fp16=args.fp16,
        precision_constraints='prefer',
        profiles=None,
        builder_optimization_level=args.optimization_level
    )
    
    print(f"TensorRT构建配置: FP16={create_config.fp16}, 优化级别={args.optimization_level}")
    
    # 解析ONNX网络
    parse_network_from_onnx = NetworkFromOnnxPath(args.onnx_path)
    
    # 网络后处理仅在FP16开启时进行
    if args.fp16:
        postprocess_script = 'tools/network_postprocess.py'
        if os.path.exists(postprocess_script):
            print(f"FP16模式已启用，加载智能网络后处理脚本: {postprocess_script}")
            postprocess_func = InvokeFromScript(postprocess_script, name='postprocess')
            network = PostprocessNetwork(parse_network_from_onnx, func=postprocess_func)
        else:
            print("FP16模式已启用，但未找到网络后处理脚本，使用默认配置")
            network = parse_network_from_onnx
    else:
        print("FP32模式，跳过网络后处理")
        network = parse_network_from_onnx
    
    # 构建引擎
    print("开始构建TensorRT引擎...")
    try:
        build_engine = EngineBytesFromNetwork(network, config=create_config)
        
        # 保存引擎
        print(f"保存引擎到: {engine_path}")
        deserialize_engine = EngineFromBytes(build_engine)
        save_engine = SaveEngine(deserialize_engine, path=engine_path)
        
        # 触发实际的引擎构建和保存
        _ = save_engine()
        
        # 验证引擎文件是否成功生成
        if os.path.exists(engine_path):
            file_size = os.path.getsize(engine_path) / 1024 / 1024  # MB
            print(f"✓ 引擎文件已生成: {engine_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ 引擎文件未生成: {engine_path}")
            sys.exit(1)
    
    except Exception as e:
        print(f"构建引擎时发生错误: {e}")
        sys.exit(1)
    
    # 如果需要比较，运行对比测试
    if args.compare:
        run_comparison(args.onnx_path, save_engine, engine_path)
    else:
        print(f"✓ TensorRT引擎构建完成: {engine_path}")
        print("提示: 使用 --compare 参数可进行精度对比测试")


def run_comparison(onnx_path, save_engine, engine_path):
    """运行ONNX Runtime和TensorRT对比"""
    print("\n=== 开始模型推理对比 ===")
    
    # 构建ONNX Runtime会话
    build_onnxrt_session = SessionFromOnnx(onnx_path)
    
    # 创建runners
    runners = [
        OnnxrtRunner(build_onnxrt_session),
        TrtRunner(save_engine),
    ]
    
    print("正在运行ONNX Runtime和TensorRT推理...")
    try:
        results = Comparator.run(runners)
        
        print("\n=== 推理结果分析 ===")
        success = True
        
        # 精度对比
        print("进行精度对比分析...")
        accuracy_result = Comparator.compare_accuracy(results)
        success &= bool(accuracy_result)
        
        # 性能统计
        print_performance_stats(results)
        
        # 输出统计
        print_output_stats(results)
        
        # 最终报告
        print_final_report(success, engine_path)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"比较过程中发生错误: {e}")
        sys.exit(1)

def print_performance_stats(results):
    """打印性能统计"""
    print("\n=== 性能统计 ===")
    for runner_name, result in results.items():
        if hasattr(result, 'runtime'):
            print(f"{runner_name} 推理时间: {result.runtime:.4f}s")
        else:
            print(f"{runner_name}: 推理完成")


def print_output_stats(results):
    """打印输出统计"""
    print("\n=== 输出统计 ===")
    try:
        import numpy as np
        if isinstance(results, list):
            # 处理Comparator.run返回的列表格式
            for i, result in enumerate(results):
                print(f"\nRunner {i} 输出:")
                if hasattr(result, 'items'):
                    for output_name, output_data in result.items():
                        if isinstance(output_data, np.ndarray):
                            print(f"  {output_name}: 形状={output_data.shape}, "
                                  f"数据类型={output_data.dtype}, "
                                  f"范围=[{output_data.min():.6f}, {output_data.max():.6f}], "
                                  f"均值={output_data.mean():.6f}")
                        else:
                            print(f"  {output_name}: {type(output_data)}")
                else:
                    print(f"  结果类型: {type(result)}")
        elif hasattr(results, 'items'):
            # 处理字典格式
            for runner_name, result in results.items():
                print(f"\n{runner_name} 输出:")
                if hasattr(result, 'items'):
                    for output_name, output_data in result.items():
                        if isinstance(output_data, np.ndarray):
                            print(f"  {output_name}: 形状={output_data.shape}, "
                                  f"数据类型={output_data.dtype}, "
                                  f"范围=[{output_data.min():.6f}, {output_data.max():.6f}], "
                                  f"均值={output_data.mean():.6f}")
                        else:
                            print(f"  {output_name}: {type(output_data)}")
                else:
                    print(f"  结果类型: {type(result)}")
        else:
            print(f"结果类型: {type(results)}")
    except Exception as e:
        print(f"输出统计处理错误: {e}")


def print_final_report(success, engine_path):
    """打印最终报告"""
    print("\n=== 最终结果 ===")
    if success:
        print("✓ 精度对比通过！TensorRT引擎与ONNX Runtime结果一致")
        print(f"✓ 优化的TensorRT引擎已保存到: {engine_path}")
        print("✓ 建议在生产环境中使用此优化引擎")
    else:
        print("✗ 精度对比失败！请检查模型和后处理配置")
        print("建议调整容差或检查网络后处理脚本")


if __name__ == '__main__':
    main()
