# Polygraphy Python API 使用指南

## CLI `mark all` 标记对应的 Python API

### 概述

在 Polygraphy 的 `run` CLI 命令中，`--trt-outputs mark all` 和 `--onnx-outputs mark all` 参数用于将模型中的所有张量标记为输出。本指南展示了如何在 Python 代码中实现相同的功能。

### 核心常量

```python
from polygraphy import constants

# 特殊值，用于标记所有张量为输出
constants.MARK_ALL  # 值为 "mark-all"
```

### ONNX 模型

#### 1. 使用 ModifyOutputs 加载器

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

#### 2. 结合其他 ONNX 操作

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

### TensorRT 网络

#### 1. 使用 ModifyNetworkOutputs 加载器

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

#### 2. 完整的 TensorRT 引擎构建示例

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

### CLI 命令对应关系

#### 原始 CLI 命令
```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-outputs mark all \
    --onnx-outputs mark all
```

#### 等价的 Python 代码

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

### 最佳实践

#### 1. 内存管理
```python
# 对于大模型，考虑使用 copy=False 以节省内存
modify_outputs = ModifyOutputs(
    onnx_loader, 
    outputs=constants.MARK_ALL,
    copy=False  # 不创建模型副本
)
```

#### 2. 错误处理
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

### 总结

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

## CLI `polygraphy debug` 命令对应的 Python API

### 概述

`polygraphy debug` 是一个用于调试各种模型问题的实验性工具套件。它提供了多个子工具，用于迭代调试、模型简化、精度分析等功能。本节展示如何在 Python 代码中直接使用这些调试功能。

### 重要概念：节点(Node) vs 层(Layer)

在 Polygraphy 的调试工具中，**节点(Node)** 和 **层(Layer)** 是两个不同的概念，适用于不同的模型表示：

#### 节点(Node) - 用于 ONNX 模型
- **定义**: ONNX 图中的基本计算单元，对应 ONNX 模型中的操作符
- **获取方式**: `len(model.graph.node)` 或 `len(GRAPH.nodes)` (使用 onnx-graphsurgeon)
- **使用场景**: 
  - `debug reduce` - 简化 ONNX 模型时按节点数计算迭代次数
  - ONNX 模型分析和修改
- **示例**: Conv、Relu、BatchNorm、Add 等 ONNX 操作符

```python
import onnx
from polygraphy.backend.onnx import gs_from_onnx

# 加载 ONNX 模型
model = onnx.load("model.onnx")
num_nodes = len(model.graph.node)  # ONNX 节点数
print(f"ONNX 模型包含 {num_nodes} 个节点")

# 使用 onnx-graphsurgeon
graph = gs_from_onnx(model)
num_gs_nodes = len(graph.nodes)  # GraphSurgeon 节点数
print(f"GraphSurgeon 表示包含 {num_gs_nodes} 个节点")
```

#### 层(Layer) - 用于 TensorRT 网络
- **定义**: TensorRT 网络中的计算层，对应 TensorRT 的 ILayer 对象
- **获取方式**: `len(network)` 或 `network.num_layers`
- **使用场景**:
  - `debug precision` - 精度调试时按层数计算迭代次数  
  - TensorRT 网络优化和分析
- **示例**: IConvolutionLayer、IActivationLayer、IElementWiseLayer 等 TensorRT 层

```python
from polygraphy.backend.trt import NetworkFromOnnxPath

# 创建 TensorRT 网络
network_loader = NetworkFromOnnxPath("model.onnx")
builder, network, parser = network_loader()

num_layers = len(network)  # 等同于 network.num_layers
print(f"TensorRT 网络包含 {num_layers} 层")

# 遍历所有层
for i in range(num_layers):
    layer = network.get_layer(i)
    print(f"层 {i}: {layer.name}, 类型: {layer.type}")
```

#### 关键区别总结

| 特性 | 节点(Node) | 层(Layer) |
|------|-----------|-----------|
| **模型格式** | ONNX 模型 | TensorRT 网络 |
| **计数方式** | `len(model.graph.node)` | `len(network)` |
| **调试工具** | `debug reduce` | `debug precision` |
| **迭代计算** | 基于节点数 | 基于层数 |
| **操作对象** | ONNX 操作符 | TensorRT ILayer |

#### 数量关系说明

**重要**: ONNX 节点数和 TensorRT 层数通常不相等，因为：

1. **融合优化**: TensorRT 会将多个 ONNX 节点融合为一个层
   ```
   ONNX: Conv -> BatchNorm -> ReLU (3个节点)
   TensorRT: ConvBNReLU (1层)
   ```

2. **插入操作**: TensorRT 可能插入额外的层(如格式转换层)

3. **优化重构**: TensorRT 优化器可能重组网络结构

```python
# 示例：比较同一模型的节点数和层数
def compare_node_layer_count(model_path):
    """比较 ONNX 节点数和 TensorRT 层数"""
    
    # ONNX 节点数
    import onnx
    model = onnx.load(model_path)
    node_count = len(model.graph.node)
    
    # TensorRT 层数
    from polygraphy.backend.trt import NetworkFromOnnxPath
    network_loader = NetworkFromOnnxPath(model_path)
    builder, network, parser = network_loader()
    layer_count = len(network)
    
    print(f"ONNX 节点数: {node_count}")
    print(f"TensorRT 层数: {layer_count}")
    print(f"融合比例: {node_count/layer_count:.2f}:1")
    
    return node_count, layer_count

# 使用示例
node_count, layer_count = compare_node_layer_count("model.onnx")
```

### 核心架构

所有 debug 工具都基于相同的迭代调试框架：

```python
from polygraphy.tools.debug.debug import Debug
from polygraphy.tools.debug.subtool import Build, Precision, Reduce, Repeat
```

### 1. Debug Build - 重复构建引擎以隔离有问题的策略

#### CLI 命令
```bash
polygraphy debug build model.onnx \
    --save-tactics replay.json \
    --artifacts replay.json \
    --until 10 \
    --check "polygraphy run model.onnx --trt --validate"
```

#### Python API 实现

```python
from polygraphy.tools.debug.subtool.build import Build
from polygraphy.tools.debug.subtool.iterative_debug_args import (
    CheckCmdArgs, ArtifactSortArgs, IterativeDebugArgs
)
from polygraphy.tools.args import ModelArgs, TrtConfigArgs
import argparse

def debug_build_engine(model_path, num_iterations=10, check_command=None, artifacts=None, save_tactics=None):
    """
    重复构建 TensorRT 引擎以调试非确定性行为
    
    Args:
        model_path: ONNX 模型路径
        num_iterations: 迭代次数
        check_command: 检查命令列表，如 ['polygraphy', 'run', 'model.onnx', '--trt']
        artifacts: 要跟踪和排序的工件列表
        save_tactics: 保存策略文件的路径
    
    Returns:
        调试结果信息
    """
    
    # 创建 Build 工具实例
    build_tool = Build()
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            # 模型相关
            self.model_file = model_path
            
            # 迭代控制
            self.until = num_iterations - 1  # until 参数是 0 索引
            
            # 检查命令
            self.check = check_command
            self.fail_codes = None
            self.ignore_fail_codes = None
            self.fail_regex = None
            self.show_output = False
            self.hide_fail_output = False
            
            # 工件管理
            self.artifacts = artifacts or []
            self.artifacts_dir = "polygraphy_artifacts"
            
            # 中间工件
            self.iter_artifact_path = "polygraphy_debug.engine"
            self.remove_intermediate = True
            self.iteration_info_path = None
            
            # 调试重放
            self.load_debug_replay = None
            self.save_debug_replay = "polygraphy_debug_replay.json"
            
            # TensorRT 配置
            self.save_tactics = save_tactics
            # 其他 TRT 相关参数...
    
    args = Args()
    
    # 运行调试构建
    try:
        result = build_tool.run_impl(args)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 使用示例
if __name__ == "__main__":
    result = debug_build_engine(
        model_path="model.onnx",
        num_iterations=5,
        check_command=["polygraphy", "run", "model.onnx", "--trt", "--validate"],
        artifacts=["polygraphy_debug.engine"],
        save_tactics="debug_tactics.json"
    )
    print(f"调试结果: {result}")
```

### 2. Debug Precision - 精度调试（标记层以在更高精度下运行）

#### CLI 命令
```bash
polygraphy debug precision model.onnx \
    --fp16 --int8 \
    --mode bisect \
    --direction forward \
    --precision float32 \
    --check "polygraphy run model.onnx --trt --validate"
```

#### Python API 实现

```python
from polygraphy.tools.debug.subtool.precision import Precision, BisectMarker, LinearMarker

class PrecisionDebugger:
    def __init__(self, model_path):
        self.model_path = model_path
        self.precision_tool = Precision()
        
    def debug_precision(self, mode="bisect", direction="forward", 
                       target_precision="float32", check_command=None):
        """
        迭代标记层以在更高精度下运行，找到性能和质量的平衡点
        
        Args:
            mode: 选择模式 ("bisect" 或 "linear")
                - "bisect": 二分搜索，迭代次数约为 log₂(网络层数)，速度快
                - "linear": 线性搜索，迭代次数等于网络层数，更彻底但较慢
            direction: 方向 ("forward" 或 "reverse") 
            target_precision: 目标精度 ("float32" 或 "float16")
            check_command: 检查命令
        
        Returns:
            调试结果
        """
        
        class Args:
            def __init__(self):
                self.model_file = self.model_path
                self.mode = mode
                self.direction = direction  
                self.precision = target_precision
                
                # TRT 配置 - 必须启用低精度才能进行精度调试
                self.fp16 = True
                self.int8 = True if target_precision == "float32" else False
                self.tf32 = False
                
                # 检查命令
                self.check = check_command
                self.fail_codes = None
                self.ignore_fail_codes = None
                self.fail_regex = None
                self.show_output = False
                self.hide_fail_output = False
                
                # 工件管理
                self.artifacts = []
                self.artifacts_dir = "polygraphy_artifacts"
                
                # 中间工件
                self.iter_artifact_path = "polygraphy_debug.engine"
                self.remove_intermediate = True
                self.iteration_info_path = None
                
                # 调试重放
                self.load_debug_replay = None
                self.save_debug_replay = "polygraphy_debug_replay.json"
        
        args = Args()
        
        try:
            result = self.precision_tool.run_impl(args)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

# 使用示例
debugger = PrecisionDebugger("model.onnx")
result = debugger.debug_precision(
    mode="bisect",
    direction="forward", 
    target_precision="float32",
    check_command=["polygraphy", "run", "model.onnx", "--trt", "--validate"]
)
print(f"精度调试结果: {result}")
```

### 3. Debug Reduce - 模型简化（减少失败模型到最小节点集）

#### CLI 命令
```bash
polygraphy debug reduce failing_model.onnx \
    --output reduced_model.onnx \
    --min-good minimal_good.onnx \
    --mode bisect \
    --check "polygraphy run --trt --validate"
```

#### Python API 实现

```python
from polygraphy.tools.debug.subtool.reduce import Reduce, BisectMarker, LinearMarker

class ModelReducer:
    def __init__(self):
        self.reduce_tool = Reduce()
        
    def reduce_failing_model(self, input_model_path, output_model_path, 
                           min_good_path=None, mode="bisect", 
                           check_command=None, model_input_shapes=None):
        """
        将失败的 ONNX 模型减少到导致失败的最小节点集
        
        Args:
            input_model_path: 输入模型路径
            output_model_path: 输出简化模型路径
            min_good_path: 最小良好模型保存路径（可选）
            mode: 简化模式 ("bisect" 或 "linear")
                - "bisect": 二分搜索，迭代次数约为 log₂(节点数)，快速找到问题区域
                - "linear": 线性搜索，迭代次数等于节点数，更精确但耗时更长
            check_command: 用于验证的检查命令
            model_input_shapes: 模型输入形状（用于动态形状模型）
            
        Returns:
            简化结果
        """
        
        class Args:
            def __init__(self):
                # 模型路径
                self.model_file = input_model_path
                
                # 输出设置
                self.output = output_model_path
                self.min_good = min_good_path
                
                # 简化设置
                self.mode = mode
                self.reduce_inputs = True
                self.reduce_outputs = True
                
                # 模型输入形状
                self.model_inputs = model_input_shapes
                
                # 检查命令
                self.check = check_command
                self.fail_codes = None
                self.ignore_fail_codes = None  
                self.fail_regex = None
                self.show_output = False
                self.hide_fail_output = False
                
                # 工件管理
                self.artifacts = []
                self.artifacts_dir = "polygraphy_artifacts"
                
                # 中间工件
                self.iter_artifact_path = "polygraphy_debug.onnx"
                self.remove_intermediate = True
                self.iteration_info_path = None
                
                # 调试重放
                self.load_debug_replay = None
                self.save_debug_replay = "polygraphy_debug_replay.json"
                
                # ONNX 设置
                self.do_shape_inference = True
                self.force_fallback = False
        
        args = Args()
        
        try:
            result = self.reduce_tool.run_impl(args)
            return {
                "success": True, 
                "result": result,
                "reduced_model": output_model_path,
                "min_good_model": min_good_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# 使用示例
reducer = ModelReducer()
result = reducer.reduce_failing_model(
    input_model_path="failing_model.onnx",
    output_model_path="reduced_model.onnx", 
    min_good_path="minimal_good.onnx",
    mode="bisect",
    check_command=["polygraphy", "run", "--trt", "--validate"],
    model_input_shapes={"input": [1, 3, 224, 224]}
)
print(f"模型简化结果: {result}")
```

### 4. Debug Repeat - 重复执行命令并排序工件

#### CLI 命令
```bash
polygraphy debug repeat \
    --until 10 \
    --artifacts engine.plan log.txt \
    --check "your_test_command"
```

#### Python API 实现

```python
from polygraphy.tools.debug.subtool.repeat import Repeat

def debug_repeat_command(num_iterations=10, check_command=None, artifacts=None):
    """
    重复执行命令并将生成的工件分类到 good/bad 目录
    
    Args:
        num_iterations: 重复执行次数
        check_command: 要执行的检查命令
        artifacts: 要跟踪的工件列表
        
    Returns:
        执行结果
    """
    
    repeat_tool = Repeat()
    
    class Args:
        def __init__(self):
            # 迭代控制
            self.until = num_iterations - 1
            
            # 检查命令
            self.check = check_command
            self.fail_codes = None
            self.ignore_fail_codes = None
            self.fail_regex = None
            self.show_output = False
            self.hide_fail_output = False
            
            # 工件管理
            self.artifacts = artifacts or []
            self.artifacts_dir = "polygraphy_artifacts"
            
            # 调试重放
            self.load_debug_replay = None
            self.save_debug_replay = "polygraphy_debug_replay.json"
    
    args = Args()
    
    try:
        result = repeat_tool.run_impl(args)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 使用示例
result = debug_repeat_command(
    num_iterations=5,
    check_command=["python", "test_script.py"],
    artifacts=["output.log", "result.json"]
)
print(f"重复执行结果: {result}")
```

### 5. 底层调试组件 - 自定义调试工具

#### 迭代调试基础框架

```python
from polygraphy.tools.debug.subtool.iterative_debug_args import (
    IterativeDebugArgs, CheckCmdArgs, ArtifactSortArgs, IterationContext
)

class CustomDebugTool:
    """自定义调试工具示例"""
    
    def __init__(self):
        self.check_args = CheckCmdArgs()
        self.artifact_args = ArtifactSortArgs()
        self.iter_args = IterativeDebugArgs()
        
    def custom_debug_iterate(self, make_artifact_func, advance_func, 
                           get_remaining_func=None, max_iterations=100):
        """
        自定义迭代调试框架
        
        Args:
            make_artifact_func: 生成每次迭代工件的函数
            advance_func: 处理成功/失败并推进迭代的函数
            get_remaining_func: 估算剩余迭代次数的函数
            max_iterations: 最大迭代次数
        """
        
        def make_iter_art(context):
            """生成迭代工件的包装函数"""
            try:
                make_artifact_func(context)
            except Exception as e:
                self.iter_args.skip_iteration(success=False)
                
        def advance(context):
            """推进迭代的包装函数"""
            should_stop = advance_func(context)
            if should_stop:
                self.iter_args.stop_iteration()
        
        # 执行迭代调试
        debug_replay = self.iter_args.iterate(
            make_iter_art_func=make_iter_art,
            advance_func=advance,
            get_remaining_func=get_remaining_func
        )
        
        return debug_replay

# 使用示例：创建自定义精度标记工具
class CustomPrecisionMarker:
    def __init__(self, network):
        self.network = network
        self.current_layer_index = 0
        
    def make_artifact(self, context):
        """为当前迭代生成工件"""
        # 标记当前层为 FP32
        if self.current_layer_index < len(self.network):
            layer = self.network.get_layer(self.current_layer_index)
            layer.precision = trt.float32
            context.state["marked_layer"] = self.current_layer_index
            
        # 构建引擎并保存...
        
    def advance_iteration(self, context):
        """根据结果推进到下一层"""
        if context.success:
            # 如果成功，记录最后一个成功的层
            self.last_good_layer = context.state["marked_layer"]
            return True  # 停止迭代
        else:
            # 如果失败，继续下一层
            self.current_layer_index += 1
            return self.current_layer_index >= len(self.network)
            
    def get_remaining(self):
        """估算剩余迭代次数"""
        return max(0, len(self.network) - self.current_layer_index)

# 使用自定义工具
custom_tool = CustomDebugTool()
precision_marker = CustomPrecisionMarker(network)

debug_replay = custom_tool.custom_debug_iterate(
    make_artifact_func=precision_marker.make_artifact,
    advance_func=precision_marker.advance_iteration,
    get_remaining_func=precision_marker.get_remaining
)
```

### 6. 调试工件管理

#### 工件排序和管理

```python
from polygraphy.tools.debug.subtool.iterative_debug_args import ArtifactSortArgs
import time

class DebugArtifactManager:
    """调试工件管理器"""
    
    def __init__(self, artifacts_list, output_dir="polygraphy_artifacts"):
        self.artifact_sorter = ArtifactSortArgs()
        self.artifact_sorter.artifacts = artifacts_list
        self.artifact_sorter.output_dir = output_dir
        self.artifact_sorter.start_date = time.strftime("%x").replace("/", "-")
        self.artifact_sorter.start_time = time.strftime("%X").replace(":", "-")
        
    def sort_iteration_artifacts(self, success, iteration_id):
        """
        根据成功/失败状态排序迭代工件
        
        Args:
            success: 迭代是否成功
            iteration_id: 迭代ID（用作后缀）
        """
        suffix = f"_iter_{iteration_id}"
        self.artifact_sorter.sort_artifacts(success, suffix)
        
    def cleanup_artifacts(self):
        """清理临时工件"""
        import os
        for artifact in self.artifact_sorter.artifacts:
            if os.path.exists(artifact):
                os.remove(artifact)

# 使用示例
artifact_manager = DebugArtifactManager(
    artifacts_list=["debug.engine", "debug.log", "tactics.json"],
    output_dir="my_debug_artifacts"
)

# 在调试循环中
for i in range(10):
    # ... 生成工件 ...
    
    # 运行检查
    success = run_some_check()
    
    # 排序工件
    artifact_manager.sort_iteration_artifacts(success, i)
    
# 清理
artifact_manager.cleanup_artifacts()
```

### 7. 检查命令集成

#### 自定义检查逻辑

```python
from polygraphy.tools.debug.subtool.iterative_debug_args import CheckCmdArgs
import subprocess
import re

class CustomChecker:
    """自定义检查器"""
    
    def __init__(self):
        self.check_args = CheckCmdArgs()
        
    def setup_check(self, check_command, fail_codes=None, fail_regexes=None):
        """
        设置检查参数
        
        Args:
            check_command: 检查命令列表
            fail_codes: 被视为失败的返回码
            fail_regexes: 失败的正则表达式模式
        """
        self.check_args.check = check_command
        self.check_args.fail_codes = fail_codes
        self.check_args.ignore_fail_codes = None
        
        if fail_regexes:
            self.check_args.fail_regexes = [re.compile(regex) for regex in fail_regexes]
        else:
            self.check_args.fail_regexes = None
            
        self.check_args.show_output = False
        self.check_args.hide_fail_output = False
    
    def run_check(self, artifact_path):
        """运行检查并返回结果"""
        return self.check_args.run_check(artifact_path)
    
    def interactive_check(self, artifact_path):
        """交互式检查（用户手动判断）"""
        # 设置为无检查命令以启用交互模式
        self.check_args.check = None
        return self.check_args.run_check(artifact_path)

# 使用示例
checker = CustomChecker()

# 设置自动检查
checker.setup_check(
    check_command=["python", "validate_model.py"],
    fail_codes=[1, 2],  # 返回码 1 或 2 表示失败
    fail_regexes=[r"ERROR.*accuracy", r"FAILED.*precision"]  # 匹配这些模式表示失败
)

# 运行检查
success = checker.run_check("debug_model.onnx")
print(f"检查结果: {'通过' if success else '失败'}")

# 或者使用交互模式
interactive_checker = CustomChecker()
success = interactive_checker.interactive_check("debug_model.onnx")
```

### 8. 标记器迭代次数详细说明

#### BisectMarker（二分搜索标记器）

**迭代次数计算**：
- **Precision 调试**: `⌈log₂(TensorRT网络层数)⌉` 次 - 基于 `len(network)`
- **Reduce 调试**: `⌈log₂(ONNX模型节点数)⌉` 次 - 基于 `len(model.graph.node)`

**工作原理**：
```python
# 伪代码展示二分搜索的迭代次数
def bisect_iterations(total_items):
    """计算二分搜索所需的迭代次数"""
    import math
    return math.ceil(math.log2(total_items))

# 示例
# Precision 调试 - 基于 TensorRT 网络层数
network_layers = 100  # len(network)
iterations = bisect_iterations(network_layers)  # ≈ 7 次迭代
print(f"100层TensorRT网络的精度调试预计需要 {iterations} 次迭代")

# Reduce 调试 - 基于 ONNX 模型节点数
model_nodes = 500  # len(model.graph.node)
iterations = bisect_iterations(model_nodes)  # ≈ 9 次迭代
print(f"500节点ONNX模型的简化预计需要 {iterations} 次迭代")
```

**优势**：
- 迭代次数少，调试速度快
- 适合初步定位问题区域
- 大型模型的首选方案

**限制**：
- 可能错过某些边缘情况
- 对于有复杂分支的模型可能不够精确

#### LinearMarker（线性搜索标记器）

**迭代次数计算**：
- **Precision 调试**: `TensorRT网络层数` 次（最坏情况） - 基于 `len(network)`
- **Reduce 调试**: `ONNX模型节点数` 次（最坏情况） - 基于 `len(model.graph.node)`

**工作原理**：
```python
# 伪代码展示线性搜索的迭代次数
def linear_iterations(total_items):
    """计算线性搜索的迭代次数（最坏情况）"""
    return total_items

# 示例  
# Precision 调试 - 基于 TensorRT 网络层数
network_layers = 100  # len(network)
iterations = linear_iterations(network_layers)  # 100 次迭代
print(f"100层TensorRT网络的精度调试最多需要 {iterations} 次迭代")

# Reduce 调试 - 基于 ONNX 模型节点数
model_nodes = 500  # len(model.graph.node)
iterations = linear_iterations(model_nodes)  # 500 次迭代  
print(f"500节点ONNX模型的简化最多需要 {iterations} 次迭代")
```

**优势**：
- 搜索彻底，不会遗漏问题
- 能找到最精确的边界
- 适合复杂分支模型

**限制**：
- 迭代次数多，调试时间长
- 对于大型模型可能不实用

#### 选择建议

```python
def recommend_marker_mode(total_items, time_budget="medium"):
    """
    根据模型大小和时间预算推荐标记器模式
    
    Args:
        total_items: 总项目数（层数或节点数）
        time_budget: 时间预算 ("low", "medium", "high")
    
    Returns:
        推荐的模式和预期迭代次数
    """
    import math
    
    bisect_iterations = math.ceil(math.log2(total_items))
    linear_iterations = total_items
    
    if time_budget == "low" or total_items > 200:
        return "bisect", bisect_iterations, "快速定位，适合初步调试"
    elif time_budget == "high" or total_items < 50:
        return "linear", linear_iterations, "彻底搜索，获得最精确结果"
    else:
        # 中等时间预算：先 bisect 后 linear
        return "bisect+linear", bisect_iterations + 20, "先快速定位，再精确搜索"

# 使用示例
mode, iterations, reason = recommend_marker_mode(150, "medium")
print(f"推荐模式: {mode}, 预计迭代次数: {iterations}, 理由: {reason}")
```

#### 实际迭代次数示例

| 模型规模 | BisectMarker | LinearMarker | 推荐策略 |
|---------|-------------|-------------|---------|
| 小模型 (< 50层/节点) | ~6次 | ~50次 | Linear (彻底搜索) |
| 中模型 (50-200层/节点) | ~8次 | ~200次 | Bisect (快速定位) |
| 大模型 (200-1000层/节点) | ~10次 | ~1000次 | Bisect (必须选择) |
| 超大模型 (> 1000层/节点) | ~11次 | > 1000次 | Bisect only |

**注意**: 
- 对于 **Precision 调试**，规模基于 TensorRT 网络层数 (`len(network)`)
- 对于 **Reduce 调试**，规模基于 ONNX 模型节点数 (`len(model.graph.node)`)
- 同一模型的 ONNX 节点数通常比 TensorRT 层数多（因融合优化）

#### 组合策略

```python
def two_stage_debugging(model_path, total_items):
    """
    两阶段调试策略：先用 bisect 快速定位，再用 linear 精确搜索
    """
    
    # 阶段1：使用 bisect 快速定位问题区域
    print("阶段1: 使用 BisectMarker 快速定位...")
    bisect_result = debug_with_bisect(model_path)
    
    if bisect_result["success"]:
        print(f"BisectMarker 找到了解决方案，总计 {bisect_result['iterations']} 次迭代")
        return bisect_result
    
    # 阶段2：使用 linear 进行精确搜索（仅搜索问题区域）
    print("阶段2: 使用 LinearMarker 进行精确搜索...")
    linear_result = debug_with_linear(model_path, narrow_range=True)
    
    total_iterations = bisect_result['iterations'] + linear_result['iterations']
    print(f"两阶段调试完成，总计 {total_iterations} 次迭代")
    
    return linear_result
```

### 总结

**Polygraphy Debug Python API 核心组件**：

1. **主工具类**:
   - `Debug`: 主调试工具入口
   - `Build`: 引擎构建调试
   - `Precision`: 精度调试  
   - `Reduce`: 模型简化
   - `Repeat`: 重复执行

2. **基础框架**:
   - `IterativeDebugArgs`: 迭代调试参数管理
   - `CheckCmdArgs`: 检查命令参数管理
   - `ArtifactSortArgs`: 工件排序参数管理
   - `IterationContext`: 迭代上下文信息

3. **标记器**:
   - `BisectMarker`: 二分搜索标记器
     - **默认迭代次数**: 
       - Precision 调试: `⌈log₂(TensorRT层数)⌉` 次
       - Reduce 调试: `⌈log₂(ONNX节点数)⌉` 次
     - **适用场景**: 快速收敛，适合大型模型的初步调试
   - `LinearMarker`: 线性搜索标记器  
     - **默认迭代次数**: 
       - Precision 调试: `TensorRT层数` 次（最坏情况）
       - Reduce 调试: `ONNX节点数` 次（最坏情况）
     - **适用场景**: 更彻底的搜索，适合有分支的模型或需要精确结果的场景

4. **使用模式**:
   - 继承现有工具类并配置参数
   - 使用底层框架组件构建自定义调试工具
   - 集成检查命令和工件管理

这些 API 提供了与 CLI 调试命令完全相同的功能，同时允许更细粒度的控制和自定义扩展。

## CLI `CompareFunc` 比较函数类详解

### 概述

`CompareFunc` 类是 Polygraphy 比较器系统的核心组件，提供了多种不同的比较策略用于评估两个 `IterationResult` 对象的差异。每种比较方法针对不同的使用场景和精度要求，从传统的数值误差比较到先进的感知相似性评估。

### 核心架构

```python
from polygraphy.comparator.compare import (
    CompareFunc, 
    OutputCompareResult, 
    DistanceMetricsResult, 
    QualityMetricsResult, 
    PerceptualMetricsResult
)
from polygraphy.comparator import Comparator
```

### 比较结果类型

#### 1. OutputCompareResult - 传统数值比较结果

```python
class OutputCompareResult:
    """传统数值比较的结果，包含多种统计误差指标"""
    
    def __init__(self, passed, max_absdiff, max_reldiff, mean_absdiff, 
                 mean_reldiff, median_absdiff, median_reldiff, 
                 quantile_absdiff, quantile_reldiff):
        self.passed = passed                    # 是否通过比较
        self.max_absdiff = max_absdiff         # 最大绝对误差
        self.max_reldiff = max_reldiff         # 最大相对误差
        self.mean_absdiff = mean_absdiff       # 平均绝对误差
        self.mean_reldiff = mean_reldiff       # 平均相对误差
        self.median_absdiff = median_absdiff   # 中位数绝对误差
        self.median_reldiff = median_reldiff   # 中位数相对误差
        self.quantile_absdiff = quantile_absdiff  # 分位数绝对误差
        self.quantile_reldiff = quantile_reldiff  # 分位数相对误差

# 使用示例
def analyze_compare_results(compare_result):
    """分析比较结果中的统计信息"""
    if isinstance(compare_result, OutputCompareResult):
        print(f"通过状态: {compare_result.passed}")
        print(f"最大误差: abs={compare_result.max_absdiff:.6f}, rel={compare_result.max_reldiff:.6f}")
        print(f"平均误差: abs={compare_result.mean_absdiff:.6f}, rel={compare_result.mean_reldiff:.6f}")
        print(f"中位数误差: abs={compare_result.median_absdiff:.6f}, rel={compare_result.median_reldiff:.6f}")
```

#### 2. DistanceMetricsResult - 距离度量比较结果

```python
class DistanceMetricsResult:
    """基于距离度量的比较结果"""
    
    def __init__(self, passed, l2_norm, cosine_similarity):
        self.passed = passed                           # 是否通过比较
        self.l2_norm = l2_norm                        # L2范数（欧几里得距离）
        self.cosine_similarity = cosine_similarity     # 余弦相似度

# 使用示例
def analyze_distance_results(distance_result):
    """分析距离度量结果"""
    if isinstance(distance_result, DistanceMetricsResult):
        print(f"通过状态: {distance_result.passed}")
        print(f"L2范数: {distance_result.l2_norm:.6f}")
        print(f"余弦相似度: {distance_result.cosine_similarity:.6f}")
```

#### 3. QualityMetricsResult - 质量度量比较结果

```python
class QualityMetricsResult:
    """基于质量度量的比较结果"""
    
    def __init__(self, passed, psnr=None, snr=None):
        self.passed = passed    # 是否通过比较
        self.psnr = psnr       # 峰值信噪比(Peak Signal-to-Noise Ratio)
        self.snr = snr         # 信噪比(Signal-to-Noise Ratio)

# 使用示例
def analyze_quality_results(quality_result):
    """分析质量度量结果"""
    if isinstance(quality_result, QualityMetricsResult):
        print(f"通过状态: {quality_result.passed}")
        if quality_result.psnr is not None:
            print(f"PSNR: {quality_result.psnr:.2f} dB")
        if quality_result.snr is not None:
            print(f"SNR: {quality_result.snr:.2f} dB")
```

#### 4. PerceptualMetricsResult - 感知度量比较结果

```python
class PerceptualMetricsResult:
    """基于感知度量的比较结果"""
    
    def __init__(self, passed, lpips=None):
        self.passed = passed    # 是否通过比较
        self.lpips = lpips     # LPIPS(Learned Perceptual Image Patch Similarity)

# 使用示例
def analyze_perceptual_results(perceptual_result):
    """分析感知度量结果"""
    if isinstance(perceptual_result, PerceptualMetricsResult):
        print(f"通过状态: {perceptual_result.passed}")
        if perceptual_result.lpips is not None:
            print(f"LPIPS: {perceptual_result.lpips:.6f}")
```

### 比较函数详解

#### 1. CompareFunc.simple() - 传统数值比较

**功能**: 基于绝对误差和相对误差的传统数值比较，是最常用的比较方法。

**核心参数**:

```python
def create_simple_comparator():
    """创建传统数值比较器"""
    
    # 基础配置
    simple_compare = CompareFunc.simple(
        check_shapes=True,           # 严格检查形状匹配
        rtol=1e-5,                  # 相对容忍度 (1%)
        atol=1e-5,                  # 绝对容忍度
        fail_fast=False,            # 不在首次失败时停止
        check_error_stat="elemwise", # 错误统计类型
        error_quantile=0.99,        # 分位数设置
        infinities_compare_equal=False,  # 无穷大值处理
        
        # 可视化选项 (实验性)
        save_heatmaps=None,         # 保存热图路径
        show_heatmaps=False,        # 显示热图
        save_error_metrics_plot=None,  # 保存误差图路径
        show_error_metrics_plot=False  # 显示误差图
    )
    return simple_compare

# 错误统计类型详解
ERROR_STAT_TYPES = {
    "elemwise": "逐元素检查 - 检查每个元素是否超出容忍度",
    "max": "最大值检查 - 检查最大绝对/相对误差，最严格",
    "mean": "均值检查 - 检查平均绝对/相对误差",
    "median": "中位数检查 - 检查中位数绝对/相对误差", 
    "quantile": "分位数检查 - 检查指定分位数的误差"
}
```

**按输出自定义配置**:

```python
def create_per_output_comparator():
    """为不同输出创建个性化比较配置"""
    
    # 为不同输出设置不同的容忍度
    per_output_rtol = {
        "detection_boxes": 1e-3,     # 检测框相对宽松
        "detection_scores": 1e-5,    # 分数需要精确
        "feature_maps": 1e-4,        # 特征图中等精度
        "": 1e-5                     # 默认值（空字符串键）
    }
    
    per_output_atol = {
        "detection_boxes": 1e-3,
        "detection_scores": 1e-6,
        "feature_maps": 1e-4,
        "": 1e-5
    }
    
    per_output_error_stat = {
        "detection_boxes": "mean",      # 检测框使用均值检查
        "detection_scores": "max",      # 分数使用最严格检查
        "feature_maps": "quantile",     # 特征图使用分位数检查
        "": "elemwise"                  # 默认逐元素检查
    }
    
    simple_compare = CompareFunc.simple(
        rtol=per_output_rtol,
        atol=per_output_atol,
        check_error_stat=per_output_error_stat,
        error_quantile=0.95  # 分位数设置为95%
    )
    return simple_compare
```

**高级可视化功能**:

```python
def create_visual_comparator(output_dir="comparison_results"):
    """创建带可视化功能的比较器"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visual_compare = CompareFunc.simple(
        rtol=1e-4,
        atol=1e-4,
        
        # 启用可视化功能
        save_heatmaps=os.path.join(output_dir, "heatmaps"),
        show_heatmaps=True,
        save_error_metrics_plot=os.path.join(output_dir, "error_plots"),
        show_error_metrics_plot=True
    )
    return visual_compare
```

#### 2. CompareFunc.indices() - 索引比较

**功能**: 专门用于比较索引类输出（如Top-K操作的结果），支持索引容忍度。

```python
def create_indices_comparator():
    """创建索引比较器"""
    
    # 基础索引比较
    indices_compare = CompareFunc.indices(
        index_tolerance=0,    # 索引必须完全匹配
        fail_fast=False
    )
    
    # 宽松索引比较 - 允许一定位置偏差
    tolerant_indices_compare = CompareFunc.indices(
        index_tolerance=2,    # 允许索引相差最多2个位置
        fail_fast=False
    )
    
    return indices_compare, tolerant_indices_compare

# 索引容忍度示例
def demonstrate_index_tolerance():
    """演示索引容忍度的工作原理"""
    
    # 示例输出
    output0 = [0, 1, 2, 3, 4]  # 基准输出
    output1 = [1, 0, 2, 4, 3]  # 待比较输出
    
    tolerance_examples = {
        0: "完全匹配 - 失败(0和1位置互换)",
        1: "容忍度1 - 通过(0和1只相差1位)",
        2: "容忍度2 - 通过(3和4相差1位，在容忍范围内)"
    }
    
    return tolerance_examples
```

#### 3. CompareFunc.distance_metrics() - 距离度量比较

**功能**: 使用L2范数和余弦相似度等距离度量来比较输出。

```python
def create_distance_comparator():
    """创建距离度量比较器"""
    
    distance_compare = CompareFunc.distance_metrics(
        l2_tolerance=1e-5,                    # L2范数容忍度
        cosine_similarity_threshold=0.997,    # 余弦相似度阈值
        check_shapes=True,
        fail_fast=False
    )
    return distance_compare

# 距离度量原理
def understand_distance_metrics():
    """理解距离度量的工作原理"""
    
    concepts = {
        "L2范数": {
            "定义": "欧几里得距离，计算两个向量之间的直线距离",
            "公式": "√(Σ(a_i - b_i)²)",
            "特点": "对大的差异敏感，适合检测显著变化",
            "典型值": "越小越好，0表示完全相同"
        },
        
        "余弦相似度": {
            "定义": "度量两个向量方向的相似性",
            "公式": "(A·B) / (||A|| × ||B||)",
            "范围": "[-1, 1]，1表示完全相同方向",
            "特点": "不受向量长度影响，只关注方向",
            "用途": "适合特征向量、嵌入向量比较"
        }
    }
    return concepts
```

#### 4. CompareFunc.quality_metrics() - 质量度量比较

**功能**: 使用PSNR和SNR等质量度量来比较输出，特别适合图像和信号处理。

```python
def create_quality_comparator():
    """创建质量度量比较器"""
    
    quality_compare = CompareFunc.quality_metrics(
        psnr_tolerance=30.0,    # PSNR最小值(dB)，高于30dB认为质量良好
        snr_tolerance=20.0,     # SNR最小值(dB)
        check_shapes=True,
        fail_fast=False
    )
    return quality_compare

# 质量度量概念
def understand_quality_metrics():
    """理解质量度量的概念和应用"""
    
    metrics_info = {
        "PSNR": {
            "全名": "Peak Signal-to-Noise Ratio - 峰值信噪比",
            "单位": "dB (分贝)",
            "计算": "20*log10(MAX) - 10*log10(MSE)",
            "典型值": {
                "> 40 dB": "优秀质量",
                "30-40 dB": "良好质量", 
                "20-30 dB": "可接受质量",
                "< 20 dB": "质量较差"
            },
            "应用": "图像质量评估、视频编码质量评估"
        },
        
        "SNR": {
            "全名": "Signal-to-Noise Ratio - 信噪比", 
            "单位": "dB (分贝)",
            "计算": "10*log10(信号功率/噪声功率)",
            "特点": "值越高，信号质量越好",
            "应用": "音频质量评估、通信质量评估"
        }
    }
    return metrics_info
```

#### 5. CompareFunc.perceptual_metrics() - 感知度量比较

**功能**: 使用LPIPS等感知度量来比较输出，更符合人类视觉感知。

```python
def create_perceptual_comparator():
    """创建感知度量比较器"""
    
    perceptual_compare = CompareFunc.perceptual_metrics(
        lpips_threshold=0.1,    # LPIPS阈值，越小越相似
        check_shapes=True,
        fail_fast=False
    )
    return perceptual_compare

# 感知度量概念
def understand_perceptual_metrics():
    """理解感知度量的概念"""
    
    lpips_info = {
        "LPIPS": {
            "全名": "Learned Perceptual Image Patch Similarity",
            "特点": "基于深度学习的感知相似性度量",
            "优势": "更符合人类视觉感知，比传统度量更准确",
            "范围": "[0, +∞)，0表示完全相同",
            "典型阈值": {
                "< 0.1": "非常相似",
                "0.1 - 0.3": "较相似", 
                "0.3 - 0.6": "中等相似",
                "> 0.6": "差异较大"
            },
            "依赖": "需要安装 torch 和 lpips 包"
        }
    }
    return lpips_info
```

### 综合应用示例

#### 完整的比较分析流程

```python
def comprehensive_model_comparison(runner1, runner2, test_inputs):
    """执行全面的模型比较分析"""
    
    from polygraphy.comparator import Comparator
    
    # 1. 基础数值比较
    print("=== 基础数值比较 ===")
    simple_compare = CompareFunc.simple(rtol=1e-5, atol=1e-5)
    comparator1 = Comparator([runner1, runner2], compare_func=simple_compare)
    
    results1 = comparator1.run(test_inputs)
    accuracy1 = comparator1.compare_accuracy(results1)
    
    # 2. 距离度量比较
    print("=== 距离度量比较 ===")
    distance_compare = CompareFunc.distance_metrics()
    comparator2 = Comparator([runner1, runner2], compare_func=distance_compare)
    
    results2 = comparator2.run(test_inputs)
    accuracy2 = comparator2.compare_accuracy(results2)
    
    # 3. 质量度量比较 (适用于图像输出)
    print("=== 质量度量比较 ===")
    quality_compare = CompareFunc.quality_metrics(psnr_tolerance=25.0)
    comparator3 = Comparator([runner1, runner2], compare_func=quality_compare)
    
    results3 = comparator3.run(test_inputs)
    accuracy3 = comparator3.compare_accuracy(results3)
    
    # 整合结果分析
    analysis_result = {
        "simple": analyze_results(accuracy1),
        "distance": analyze_results(accuracy2), 
        "quality": analyze_results(accuracy3)
    }
    
    return analysis_result

def analyze_results(accuracy_results):
    """分析比较结果"""
    passed_count = 0
    total_count = 0
    
    for output_name, result in accuracy_results.items():
        if isinstance(result, dict):  # 多次迭代结果
            for iter_result in result.values():
                if isinstance(iter_result, dict):  # 多个输出
                    for out_result in iter_result.values():
                        total_count += 1
                        if out_result:
                            passed_count += 1
                else:
                    total_count += 1
                    if iter_result:
                        passed_count += 1
        else:
            total_count += 1
            if result:
                passed_count += 1
    
    return {
        "passed": passed_count,
        "total": total_count,
        "pass_rate": passed_count / total_count if total_count > 0 else 0
    }
```

### 总结

**CompareFunc 类核心特性**:

1. **多种比较策略**:
   - `simple()`: 传统数值比较，基于绝对/相对误差
   - `indices()`: 索引比较，支持位置容忍度  
   - `distance_metrics()`: 距离度量比较，L2范数+余弦相似度
   - `quality_metrics()`: 质量度量比较，PSNR+SNR
   - `perceptual_metrics()`: 感知度量比较，LPIPS

2. **灵活的配置选项**:
   - 按输出名称个性化配置
   - 多种误差统计方式
   - 可视化和分析功能
   - 形状检查和快速失败选项

3. **丰富的结果类型**:
   - `OutputCompareResult`: 详细数值统计
   - `DistanceMetricsResult`: 距离度量信息
   - `QualityMetricsResult`: 质量指标信息  
   - `PerceptualMetricsResult`: 感知相似性信息

4. **适用场景**:
   - 模型验证和回归测试
   - 精度分析和误差诊断
   - 不同推理后端比较
   - 模型优化效果评估

这些比较函数为 Polygraphy 提供了强大而灵活的模型输出比较能力，能够适应从基本数值验证到高级感知相似性分析的各种需求。