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
import time
from pathlib import Path

from polygraphy import mod, config
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineBytesFromNetwork, EngineFromBytes, EngineFromPath, NetworkFromOnnxPath, TrtRunner, CreateConfig, PostprocessNetwork, SaveEngine
from polygraphy.backend.common import InvokeFromScript
from polygraphy.comparator import Comparator

# 启用RTX加速，规避TensorRT 8.6.1兼容性问题
config.USE_TENSORRT_RTX = True

# 添加项目路径到系统路径
project_root = Path(__file__).parent.parent  # 获取父目录作为项目根目录
sys.path.insert(0, str(project_root))

from infer_onnx import RUN

trt = mod.lazy_import('tensorrt>=8.5')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='构建TensorRT引擎并可选地与ONNX Runtime进行比较',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主要参数
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
    
    # 构建参数
    build_group = parser.add_argument_group('构建参数')
    build_group.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='启用FP16精度（默认启用）'
    )
    
    build_group.add_argument(
        '--no-fp16',
        action='store_true',
        help='禁用FP16精度，使用FP32'
    )
    
    build_group.add_argument(
        '--optimization-level',
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help='TensorRT构建优化级别（0-5）'
    )
    
    # 比较功能参数
    compare_group = parser.add_argument_group('比较功能参数')
    compare_group.add_argument(
        '--compare',
        action='store_true',
        help='是否开启ONNX Runtime和TensorRT的精度对比'
    )
    
    compare_group.add_argument(
        '--rtol',
        type=float,
        default=1e-3,
        help='相对容差，用于精度比较（默认1e-3）'
    )
    
    compare_group.add_argument(
        '--atol',
        type=float,
        default=1e-3,
        help='绝对容差，用于精度比较（默认1e-3）'
    )
    
    compare_group.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='性能测试迭代次数（默认10次）'
    )
    
    compare_group.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='性能测试预热次数（默认3次）'
    )
    
    compare_group.add_argument(
        '--save-outputs',
        action='store_true',
        help='保存推理结果到文件进行详细分析'
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


def check_engine_exists_and_prompt(engine_path, compare_enabled):
    """检查引擎文件是否存在，如果存在则提供选项"""
    if os.path.exists(engine_path):
        file_size = os.path.getsize(engine_path) / 1024 / 1024  # MB
        print(f"✓ 发现已存在的引擎文件: {engine_path} ({file_size:.1f} MB)")
        
        # 如果compare未启用，询问用户是否想要启用
        if not compare_enabled:
            print("💡 建议启用比较功能来验证已存在引擎的精度和性能")
            print("   使用方式: python tools/build_engine.py --onnx-path <path> --engine-path <path> --compare")
            print("   高级选项: --rtol 1e-2 --atol 1e-3 --iterations 5 --warmup 2")
            
            try:
                response = input("是否现在启用比较功能？(y/n): ").strip().lower()
                if response in ['y', 'yes', '是', 'Y']:
                    return True, True  # skip_build=True, enable_compare=True
            except KeyboardInterrupt:
                print("\n用户取消操作")
                sys.exit(0)
        
        # 询问是否跳过构建
        try:
            response = input("引擎文件已存在，是否跳过构建？(Y/n): ").strip().lower()
            if response in ['', 'y', 'yes', '是', 'Y']:
                return True, compare_enabled  # skip_build=True, compare_enabled=原值
            else:
                print("将重新构建引擎文件...")
                return False, compare_enabled  # skip_build=False
        except KeyboardInterrupt:
            print("\n用户取消操作")
            sys.exit(0)
    
    return False, compare_enabled  # 引擎不存在，不跳过构建


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
    
    # 确定是否启用比较功能
    compare_enabled = args.compare
    
    print(f"ONNX模型路径: {args.onnx_path}")
    print(f"引擎输出路径: {engine_path}")
    
    # 检查引擎是否已存在并获取用户选择
    skip_build, compare_enabled = check_engine_exists_and_prompt(engine_path, compare_enabled)
    
    save_engine = None
    deserialize_engine = None
    saved_engine_for_comparison = None
    
    if not skip_build:
        print(f"精度对比: {'是' if compare_enabled else '否'}")
        
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
            
            # 构建完成后，创建引擎加载器用于比较（避免重复构建）
            saved_engine_for_comparison = EngineFromPath(engine_path)
            
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
    else:
        print("⏭️  跳过引擎构建，使用现有文件")
        # 如果跳过构建但需要比较，需要加载现有引擎
        if compare_enabled:
            try:
                saved_engine_for_comparison = EngineFromPath(engine_path)
                print(f"✓ 已加载现有引擎文件用于比较: {engine_path}")
            except Exception as e:
                print(f"加载现有引擎文件时发生错误: {e}")
                sys.exit(1)
    
    # 如果需要比较，运行对比测试
    if compare_enabled and saved_engine_for_comparison is not None:
        # 获取比较参数
        compare_args = {
            'rtol': args.rtol,
            'atol': args.atol,
            'iterations': args.iterations,
            'warmup': args.warmup,
            'save_outputs': args.save_outputs
        }
        run_comparison(args.onnx_path, saved_engine_for_comparison, engine_path, **compare_args)
    elif compare_enabled:
        print("⚠️  无法进行比较：引擎加载失败")
    elif not skip_build:
        print(f"✓ TensorRT引擎构建完成: {engine_path}")
        print("💡 提示: 使用 --compare 参数可进行精度对比测试")
        print("   示例: python tools/build_engine.py --onnx-path <path> --engine-path <path> --compare --rtol 1e-2 --atol 1e-3")


def run_comparison(onnx_path, save_engine, engine_path, rtol=1e-3, atol=1e-3, iterations=10, warmup=3, save_outputs=False):
    """运行ONNX Runtime和TensorRT对比"""
    print("\n=== 开始模型推理对比 ===")
    print(f"比较参数: 相对容差={rtol}, 绝对容差={atol}, 迭代次数={iterations}, 预热次数={warmup}")
    
    # 构建ONNX Runtime会话
    build_onnxrt_session = SessionFromOnnx(onnx_path)
    
    # 创建runners
    runners = [
        OnnxrtRunner(build_onnxrt_session),
        TrtRunner(save_engine),
    ]
    
    print("正在运行ONNX Runtime和TensorRT推理...")
    try:
        # 预热阶段
        if warmup > 0:
            print(f"预热阶段: 运行 {warmup} 次...")
            for i in range(warmup):
                _ = Comparator.run(runners)
                print(f"预热进度: {i+1}/{warmup}")
        
        # 正式测试阶段
        print(f"正式测试: 运行 {iterations} 次...")
        all_results = []
        import time
        start_time = time.time()
        
        for i in range(iterations):
            results = Comparator.run(runners)
            all_results.append(results)
            print(f"测试进度: {i+1}/{iterations}")
        
        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / iterations
        
        print(f"\n=== 性能统计 ===")
        print(f"总测试时间: {total_time:.4f}s")
        print(f"平均每次推理: {avg_time_per_iteration:.4f}s")
        
        print("\n=== 推理结果分析 ===")
        success = True
        
        # 使用最后一次结果进行精度对比（所有次结果应该一致）
        results = all_results[-1]
        
        # 精度对比
        print("进行精度对比分析...")
        
        from polygraphy.comparator import CompareFunc
        # 创建自定义比较函数，使用指定的相对和绝对容差
        
        # 设置保存路径：使用 infer_onnx.RUN/trt文件名
        if save_outputs:
            # 获取引擎文件名（不包含扩展名）
            engine_name = Path(engine_path).stem
            output_dir = Path(RUN) / engine_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建带输出文件路径的比较函数
            heatmap_path = str(output_dir / "heatmap")
            error_plot_path = str(output_dir / "error_metrics")
            
            print(f"保存输出文件到: {output_dir}")
            
            compare_func = CompareFunc.simple(
                rtol=rtol, 
                atol=atol,
                save_heatmaps=heatmap_path,
                save_error_metrics_plot=error_plot_path
            )
        else:
            compare_func = CompareFunc.simple(rtol=rtol, atol=atol, fail_fast=True)
        
        accuracy_result = Comparator.compare_accuracy(results, compare_func=compare_func, fail_fast=True)
        
        if save_outputs:
            # 保存原始运行结果 (RunResults 有 save 方法)
            results_path = str(output_dir / "run_results.json")
            results.save(results_path)
            print(f"✓ 运行结果已保存到: {results_path}")
            
            # # 保存精度统计信息 (AccuracyResult 的 stats 可以序列化)
            # if hasattr(accuracy_result, 'stats'):
            #     stats_path = str(output_dir / "accuracy_stats.json")
            #     import json
            #     with open(stats_path, 'w') as f:
            #         json.dump(accuracy_result.stats(), f, indent=2)
            #     print(f"✓ 精度统计已保存到: {stats_path}")
            
            # # 保存精度概况
            # summary_path = str(output_dir / "accuracy_summary.json")
            # import json
            # summary = {
            #     'overall_match': bool(accuracy_result),
            #     'percentage': accuracy_result.percentage() if accuracy_result else 0.0,
            #     'rtol': rtol,
            #     'atol': atol,
            #     'timestamp': time.time()
            # }
            # with open(summary_path, 'w') as f:
            #     json.dump(summary, f, indent=2)
            # print(f"✓ 精度概况已保存到: {summary_path}")

        
        success &= bool(accuracy_result)
        print(f"精度对比结果: {'✓ 通过' if accuracy_result else '✗ 失败'}")
        
        
        # 最终报告
        print_final_report(success, engine_path, rtol, atol, avg_time_per_iteration)
        
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

def print_final_report(success, engine_path, rtol=1e-3, atol=1e-3, avg_time=None):
    """打印最终报告"""
    print("\n=== 最终结果 ===")
    if success:
        print("✓ 精度对比通过！TensorRT引擎与ONNX Runtime结果一致")
        print(f"✓ 优化的TensorRT引擎已保存到: {engine_path}")
        print(f"✓ 使用容差: 相对容差={rtol}, 绝对容差={atol}")
        if avg_time:
            print(f"✓ 平均推理时间: {avg_time:.4f}s")
        print("✓ 建议在生产环境中使用此优化引擎")
    else:
        print("✗ 精度对比失败！请检查模型和后处理配置")
        print(f"✗ 当前容差: 相对容差={rtol}, 绝对容差={atol}")
        print("建议:")
        print("  1. 调整容差参数: --rtol <相对容差> --atol <绝对容差>")
        print("  2. 检查网络后处理脚本")
        print("  3. 验证模型输入数据范围")


if __name__ == '__main__':
    main()
