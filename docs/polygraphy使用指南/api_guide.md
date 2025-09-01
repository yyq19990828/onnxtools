# Polygraphy Python API 使用指南

## CLI `mark all` 参数对应的 Python API

### 概述

在 Polygraphy 的 `run` CLI 命令中，`--trt-outputs mark all` 和 `--onnx-outputs mark all` 参数用于将模型中的所有张量标记为输出。本指南展示了如何在 Python 代码中实现相同的功能。

### 核心常量

```python
from polygraphy import constants

# 特殊值，用于标记所有张量为输出
constants.MARK_ALL  # 值为 "mark-all"
```

## ONNX 模型

### 1. 使用 ModifyOutputs 加载器

```python
from polygraphy.backend.onnx import ModifyOutputs, OnnxFromPath
from polygraphy import constants

# 基础用法 - 标记所有张量为输出
def load_onnx_with_all_outputs(model_path):
    """加载 ONNX 模型并将所有张量标记为输出"""
    
    # 创建 ONNX 加载器
    onnx_loader = OnnxFromPath(model_path)
    
    # 使用 ModifyOutputs 标记所有张量为输出
    modify_outputs_loader = ModifyOutputs(
        onnx_loader, 
        outputs=constants.MARK_ALL  # 等价于 CLI 中的 "mark all"
    )
    
    # 加载修改后的模型
    model = modify_outputs_loader()
    return model

# 高级用法 - 标记所有张量但排除某些输出
def load_onnx_with_filtered_outputs(model_path, exclude_outputs=None):
    """加载 ONNX 模型，标记所有张量为输出但排除指定的张量"""
    
    onnx_loader = OnnxFromPath(model_path)
    
    modify_outputs_loader = ModifyOutputs(
        onnx_loader,
        outputs=constants.MARK_ALL,
        exclude_outputs=exclude_outputs  # 要排除的输出列表
    )
    
    model = modify_outputs_loader()
    return model

# 示例使用
if __name__ == "__main__":
    model_path = "your_model.onnx"
    
    # 标记所有张量为输出
    model = load_onnx_with_all_outputs(model_path)
    
    # 标记所有张量但排除某些输出
    model_filtered = load_onnx_with_filtered_outputs(
        model_path, 
        exclude_outputs=["unwanted_output1", "unwanted_output2"]
    )
```

### 2. 结合其他 ONNX 操作

```python
from polygraphy.backend.onnx import (
    ModifyOutputs, 
    OnnxFromPath, 
    InferShapes, 
    FoldConstants
)
from polygraphy import constants

def create_onnx_processing_pipeline(model_path):
    """创建完整的 ONNX 处理管道"""
    
    # 1. 加载模型
    onnx_loader = OnnxFromPath(model_path)
    
    # 2. 形状推理
    shape_infer_loader = InferShapes(onnx_loader)
    
    # 3. 常量折叠
    fold_constants_loader = FoldConstants(shape_infer_loader)
    
    # 4. 标记所有张量为输出
    all_outputs_loader = ModifyOutputs(
        fold_constants_loader,
        outputs=constants.MARK_ALL
    )
    
    return all_outputs_loader()
```

## TensorRT 网络

### 1. 使用 ModifyNetworkOutputs 加载器

```python
from polygraphy.backend.trt import (
    ModifyNetworkOutputs,
    NetworkFromOnnxPath
)
from polygraphy import constants

def create_trt_network_with_all_outputs(model_path):
    """创建 TensorRT 网络并将所有张量标记为输出"""
    
    # 1. 从 ONNX 创建网络
    network_loader = NetworkFromOnnxPath(model_path)
    
    # 2. 修改输出 - 标记所有张量
    modify_outputs_loader = ModifyNetworkOutputs(
        network_loader,
        outputs=constants.MARK_ALL  # 等价于 CLI 中的 "mark all"
    )
    
    # 3. 获取修改后的网络
    builder, network, parser = modify_outputs_loader()
    return builder, network, parser

def create_trt_network_with_filtered_outputs(model_path, exclude_outputs=None):
    """创建 TensorRT 网络，标记所有张量为输出但排除指定的张量"""
    
    network_loader = NetworkFromOnnxPath(model_path)
    
    modify_outputs_loader = ModifyNetworkOutputs(
        network_loader,
        outputs=constants.MARK_ALL,
        exclude_outputs=exclude_outputs
    )
    
    builder, network, parser = modify_outputs_loader()
    return builder, network, parser
```

### 2. 完整的 TensorRT 引擎构建示例

```python
from polygraphy.backend.trt import (
    ModifyNetworkOutputs,
    NetworkFromOnnxPath,
    CreateConfig,
    EngineBytesFromNetwork,
    EngineFromBytes,
    TrtRunner
)
from polygraphy import constants

def build_trt_engine_with_all_outputs(model_path):
    """构建包含所有输出的 TensorRT 引擎"""
    
    # 1. 创建网络加载器
    network_loader = NetworkFromOnnxPath(model_path)
    
    # 2. 标记所有张量为输出
    all_outputs_loader = ModifyNetworkOutputs(
        network_loader,
        outputs=constants.MARK_ALL
    )
    
    # 3. 创建配置
    config_loader = CreateConfig()
    
    # 4. 构建引擎
    engine_bytes_loader = EngineBytesFromNetwork(
        all_outputs_loader,
        config=config_loader
    )
    
    # 5. 创建引擎对象
    engine_loader = EngineFromBytes(engine_bytes_loader)
    
    # 6. 创建运行器
    runner = TrtRunner(engine_loader)
    
    return runner
```

## CLI 命令对应关系

### 原始 CLI 命令
```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-outputs mark all \
    --onnx-outputs mark all
```

### 等价的 Python 代码

```python
from polygraphy.backend.onnx import ModifyOutputs, OnnxFromPath
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import (
    ModifyNetworkOutputs, NetworkFromOnnxPath, 
    CreateConfig, EngineBytesFromNetwork, 
    EngineFromBytes, TrtRunner
)
from polygraphy.comparator import Comparator
from polygraphy import constants

def equivalent_python_code(model_path):
    """与 CLI 命令等价的 Python 代码"""
    
    # 1. ONNX-Runtime 运行器 (--onnxrt --onnx-outputs mark all)
    onnx_loader = OnnxFromPath(model_path)
    onnx_all_outputs = ModifyOutputs(onnx_loader, outputs=constants.MARK_ALL)
    onnxrt_runner = OnnxrtRunner(SessionFromOnnx(onnx_all_outputs))
    
    # 2. TensorRT 运行器 (--trt --trt-outputs mark all)
    network_loader = NetworkFromOnnxPath(model_path)
    trt_all_outputs = ModifyNetworkOutputs(network_loader, outputs=constants.MARK_ALL)
    config = CreateConfig()
    engine_bytes = EngineBytesFromNetwork(trt_all_outputs, config=config)
    engine = EngineFromBytes(engine_bytes)
    trt_runner = TrtRunner(engine)
    
    # 3. 比较两个运行器的输出
    comparator = Comparator([onnxrt_runner, trt_runner])
    
    # 运行比较
    results = comparator.run()
    
    # 检查结果
    success = comparator.compare_accuracy(results)
    
    return success, results
```

## 最佳实践

### 1. 内存管理
```python
# 对于大模型，考虑使用 copy=False 以节省内存
modify_outputs = ModifyOutputs(
    onnx_loader, 
    outputs=constants.MARK_ALL,
    copy=False  # 不创建模型副本
)
```

### 2. 错误处理
```python
def safe_load_with_all_outputs(model_path):
    """安全地加载模型并处理可能的错误"""
    try:
        onnx_loader = OnnxFromPath(model_path)
        modify_outputs = ModifyOutputs(onnx_loader, outputs=constants.MARK_ALL)
        model = modify_outputs()
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
```

## 总结

**CLI `mark all` 对应的核心 Python API**：

1. **常量**: `constants.MARK_ALL`（值为 `"mark-all"`）

2. **ONNX 模型**: 
   ```python
   ModifyOutputs(model, outputs=constants.MARK_ALL)
   ```

3. **TensorRT 网络**: 
   ```python
   ModifyNetworkOutputs(network, outputs=constants.MARK_ALL)
   ```

4. **参数解析逻辑**:
   - CLI 中的 `mark all` 被解析为列表 `["mark", "all"]`
   - `args_util.get_outputs()` 函数检测到该模式后转换为 `constants.MARK_ALL`
   - 最终传递给相应的 Loader 类进行处理

这些 API 提供了与 CLI 命令完全相同的功能，允许在 Python 代码中灵活地控制模型输出的标记行为。