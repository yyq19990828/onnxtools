# 层统计工具说明

## 概述

`layer_statistics.py` 是一个基于 Polygraphy 库现成函数的工具，用于分析 ONNX 模型和 TensorRT 网络中的所有层和张量信息。该工具模拟了 Polygraphy 命令行中 `--onnx-outputs mark all` 和 `--trt-outputs mark all` 的行为。

## 核心原理

### ONNX 分析 (`--onnx-outputs mark all`)

使用 Polygraphy 的核心函数：

1. **`polygraphy.backend.onnx.util.all_tensor_names(model, include_inputs=False)`**
   - 获取所有非常量节点的输出张量名称
   - 这就是 `mark all` 实际标记的张量列表

2. **`polygraphy.backend.onnx.ModifyOutputs(loader, outputs=MARK_ALL)`**
   - 模拟命令行中的 `mark all` 行为
   - 将所有非常量张量标记为模型输出

### TensorRT 分析 (`--trt-outputs mark all`)

使用 Polygraphy 的核心函数：

1. **`polygraphy.backend.trt.util.get_all_tensors(network)`**
   - 遍历网络中所有层的输入和输出张量
   - 返回张量名称到张量对象的映射字典

2. **`polygraphy.backend.trt.ModifyNetworkOutputs(loader, outputs=MARK_ALL)`**
   - 模拟命令行中的 `mark all` 行为
   - 将所有张量标记为网络输出

## 使用方法

### 基本用法

```bash
# 仅分析 ONNX 模型
python tools/layer_statistics.py --model models/your_model.onnx

# 同时分析 ONNX 和 TensorRT
python tools/layer_statistics.py --model models/your_model.onnx --build-trt

# 保存结果到 JSON 文件
python tools/layer_statistics.py --model models/your_model.onnx --save-json stats.json

# 完整分析并保存
python tools/layer_statistics.py --model models/your_model.onnx --build-trt --save-json stats.json
```

### 命令行参数

- `--model, -m`: ONNX 模型文件路径（必需）
- `--build-trt`: 同时构建 TensorRT 网络进行分析
- `--save-json`: 将统计结果保存为 JSON 文件

## 输出说明

### 控制台输出

工具会显示以下关键信息：

1. **模型基本信息**
   - 节点数量、输入输出数量
   - 使用 Polygraphy 函数验证的节点计数

2. **mark all 效果对比**
   - 原始输出数量 vs mark all 后的输出数量
   - ONNX: 从 2 个输出 → 1200 个输出
   - TensorRT: 从 2 个输出 → 1770 个输出

3. **张量统计**
   - 显示前 10 个会被 mark all 标记的张量名称
   - 最常见的层类型及数量

### JSON 输出格式

保存的 JSON 文件包含完整的分析结果：

```json
{
  "analysis_method": "Polygraphy 现成函数",
  "onnx_analysis": {
    "model_info": {...},
    "graph_info": {...},
    "tensor_analysis": {
      "mark_all_tensor_count": 1200,
      "mark_all_tensor_names": [...],
      ...
    },
    "mark_all_analysis": {...},
    "layer_types": {...}
  },
  "tensorrt_analysis": {
    "network_info": {...},
    "tensor_analysis": {...},
    "mark_all_analysis": {...}
  }
}
```

## 示例结果

### RFDETR 模型分析

```
🔹 ONNX 模型分析 (使用 Polygraphy ONNX 后端):
   模型路径: models/rfdetr-20250811slim.onnx
   总节点数: 1189
   原始输出数量: 2
   📌 --onnx-outputs mark all 效果:
      原始输出数量: 2
      mark all 后输出数量: 1200

🔹 TensorRT 网络分析 (使用 Polygraphy TensorRT 后端):
   总层数: 2197
   📌 --trt-outputs mark all 效果:
      原始输出数量: 2
      mark all 后输出数量: 1770

💡 关键发现:
   • ONNX mark all 会标记 1200 个张量为输出
   • TensorRT mark all 会标记 2199 个张量为输出
```

## 核心发现

1. **ONNX vs TensorRT 差异**
   - ONNX: 1189 个节点 → 1200 个可标记张量
   - TensorRT: 2197 个层 → 2199 个可标记张量
   - TensorRT 在优化过程中会创建更多中间张量

2. **mark all 的实际行为**
   - ONNX: 排除常量节点，标记所有计算节点的输出
   - TensorRT: 标记网络中所有可用张量（包括层的输入和输出）

3. **层类型分布**
   - ONNX: Reshape, Gemm, Transpose 为主
   - TensorRT: SHUFFLE, ELEMENTWISE, CONSTANT 为主

## 技术实现

### 关键 Polygraphy 函数调用

```python
# ONNX 张量获取
from polygraphy.backend.onnx.util import all_tensor_names
tensor_names = all_tensor_names(model, include_inputs=False)

# TensorRT 张量获取
from polygraphy.backend.trt.util import get_all_tensors
tensor_dict = get_all_tensors(network)

# mark all 行为模拟
from polygraphy.constants import MARK_ALL
from polygraphy.backend.onnx import ModifyOutputs
modified_model = ModifyOutputs(loader, outputs=MARK_ALL)()
```

## 依赖要求

- `polygraphy`
- `onnx`
- `tensorrt` (可选，用于 TensorRT 分析)
- `numpy`

## 注意事项

1. TensorRT 分析需要系统安装 TensorRT 库
2. 某些模型可能因为精度或算子支持问题导致 TensorRT 构建失败
3. 大模型的 mark all 可能会生成大量输出张量，影响推理性能
4. JSON 输出文件可能较大，包含所有张量的详细信息