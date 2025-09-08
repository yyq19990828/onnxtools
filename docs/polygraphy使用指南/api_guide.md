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