#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TensorRT引擎加载方式对比
"""

import os
import time
import tensorrt as trt
from polygraphy import config
from polygraphy.backend.trt import EngineFromPath

def test_engine_loading(engine_path, use_rtx_mode=False):
    """测试引擎加载"""
    print(f"测试模式: USE_TENSORRT_RTX = {use_rtx_mode}")
    print(f"引擎路径: {engine_path}")
    
    if not os.path.exists(engine_path):
        print(f"❌ 引擎文件不存在: {engine_path}")
        return None
    
    # 设置配置
    config.USE_TENSORRT_RTX = use_rtx_mode
    
    try:
        start_time = time.time()
        
        # 使用polygraphy加载引擎
        engine_loader = EngineFromPath(engine_path)
        engine = engine_loader()
        
        load_time = time.time() - start_time
        
        if engine:
            print(f"✅ 引擎加载成功")
            print(f"⏱️ 加载耗时: {load_time:.4f} 秒")
            print(f"📊 引擎信息:")
            print(f"   - 输入数量: {engine.num_bindings}")
            print(f"   - 最大batch大小: {engine.max_batch_size}")
            
            # 获取输入输出信息
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                binding_type = "输入" if is_input else "输出"
                print(f"   - {binding_type} {i}: {name}, 形状: {shape}, 类型: {dtype}")
            
            return load_time
        else:
            print(f"❌ 引擎加载失败")
            return None
            
    except Exception as e:
        print(f"❌ 加载出错: {str(e)}")
        return None

def find_engine_files():
    """查找项目中的引擎文件"""
    engine_files = []
    
    # 检查常见目录
    search_dirs = [
        "models/",
        "engines/",
        ".",
        "runs/"
    ]
    
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.engine') or file.endswith('.trt'):
                        engine_files.append(os.path.join(root, file))
    
    return engine_files

def main():
    print("=" * 60)
    print("TensorRT引擎加载方式对比测试")
    print("=" * 60)
    
    # 查找引擎文件
    engine_files = find_engine_files()
    
    if not engine_files:
        print("❌ 未找到引擎文件")
        print("请确保项目中有.engine或.trt文件")
        print("常见位置: models/, engines/, runs/")
        return
    
    print(f"📁 找到 {len(engine_files)} 个引擎文件:")
    for i, engine_file in enumerate(engine_files):
        print(f"   {i+1}. {engine_file}")
    
    print("\n" + "=" * 60)
    
    for engine_file in engine_files:
        print(f"\n🔧 测试引擎: {os.path.basename(engine_file)}")
        print("-" * 40)
        
        # 测试方式1: USE_TENSORRT_RTX = False
        print("\n📤 方式1: 文件流读取 (USE_TENSORRT_RTX = False)")
        time1 = test_engine_loading(engine_file, use_rtx_mode=False)
        
        print("\n" + "-" * 40)
        
        # 测试方式2: USE_TENSORRT_RTX = True  
        print("\n📥 方式2: 缓冲区读取 (USE_TENSORRT_RTX = True)")
        time2 = test_engine_loading(engine_file, use_rtx_mode=True)
        
        # 性能对比
        if time1 is not None and time2 is not None:
            print(f"\n📊 性能对比:")
            print(f"   文件流方式: {time1:.4f} 秒")
            print(f"   缓冲区方式: {time2:.4f} 秒")
            
            if time2 < time1:
                speedup = time1 / time2
                print(f"   🚀 缓冲区方式快 {speedup:.2f}x")
            elif time1 < time2:
                speedup = time2 / time1
                print(f"   🐌 文件流方式快 {speedup:.2f}x")
            else:
                print(f"   ⚖️ 两种方式性能相近")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()