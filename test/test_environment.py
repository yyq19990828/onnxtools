#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测脚本 - 检测TensorRT、CUDA、GPU等环境信息
"""

import sys
import platform
import subprocess
import os

def check_python_info():
    print("=" * 50)
    print("Python 环境信息")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print()

def check_cuda():
    print("=" * 50)
    print("CUDA 环境检测")
    print("=" * 50)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA 编译器版本:")
            print(result.stdout)
        else:
            print("CUDA 编译器未找到")
    except FileNotFoundError:
        print("CUDA 编译器未安装")
    
    # 检查CUDA运行时
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA 驱动信息:")
            print(result.stdout)
        else:
            print("nvidia-smi 命令失败")
    except FileNotFoundError:
        print("nvidia-smi 未找到，可能没有NVIDIA GPU或驱动未安装")
    print()

def check_gpu_libraries():
    print("=" * 50)
    print("GPU 相关库检测")
    print("=" * 50)
    
    # 检查 PyTorch CUDA
    torch_available = False
    try:
        import torch
        torch_available = True
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本 (PyTorch): {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
    except ImportError:
        print("PyTorch 未安装")
    
    # 检查 TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT 版本: {trt.__version__}")
        
        # 检测GPU类型
        if torch_available and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            if 'rtx' in device_name or 'geforce rtx' in device_name:
                print(f"检测到RTX GPU: {torch.cuda.get_device_name(0)}")
                print("建议使用 USE_TENSORRT_RTX = True")
            else:
                print(f"检测到非RTX GPU: {torch.cuda.get_device_name(0)}")
                print("建议使用 USE_TENSORRT_RTX = False")
        else:
            # 从nvidia-smi输出中获取GPU信息
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().lower()
                    print(f"从nvidia-smi检测到GPU: {result.stdout.strip()}")
                    if 'rtx' in gpu_name or 'geforce rtx' in gpu_name:
                        print("建议使用 USE_TENSORRT_RTX = True")
                    else:
                        print("建议使用 USE_TENSORRT_RTX = False")
            except:
                print("无法检测GPU类型")
    except ImportError:
        print("TensorRT 未安装")
    
    # 检查 ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime 版本: {ort.__version__}")
        print(f"可用提供者: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime 未安装")
    
    print()

def check_polygraphy_config():
    print("=" * 50)
    print("Polygraphy 配置检测")
    print("=" * 50)
    
    try:
        import polygraphy
        print(f"Polygraphy 版本: {polygraphy.__version__}")
        
        # 检查配置
        from polygraphy import config
        if hasattr(config, 'USE_TENSORRT_RTX'):
            print(f"当前 USE_TENSORRT_RTX 设置: {config.USE_TENSORRT_RTX}")
        else:
            print("USE_TENSORRT_RTX 配置未找到")
            
    except ImportError:
        print("Polygraphy 未安装")
    
    print()

def check_system_info():
    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    print(f"架构: {platform.machine()}")
    
    # 检查内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"总内存: {mem.total / (1024**3):.1f} GB")
        print(f"可用内存: {mem.available / (1024**3):.1f} GB")
    except ImportError:
        print("psutil 未安装，无法获取内存信息")
    
    print()

def main():
    print("环境检测开始...")
    print()
    
    check_python_info()
    check_system_info()
    check_cuda()
    check_gpu_libraries()
    check_polygraphy_config()
    
    print("=" * 50)
    print("检测完成")
    print("=" * 50)

if __name__ == "__main__":
    main()