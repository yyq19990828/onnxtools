[根目录](../CLAUDE.md) > **mcp_tools**

# MCP 工具模块 (mcp_tools)

## 模块职责

提供 MCP (Model Context Protocol) 服务器实现，使 Claude 等 LLM 能够通过工具接口使用 onnxtools 的 ONNX 推理功能。支持车辆/车牌检测、OCR 识别、颜色/层级分类和可视化标注。

## 快速开始

### 安装依赖

```bash
# 安装 MCP 支持（可选依赖）
uv pip install -e ".[mcp]"

# 或单独安装
pip install mcp httpx
```

### 启动 MCP 服务器

```bash
# 方式 1: 使用入口脚本
onnxtools-mcp

# 方式 2: 使用 Python 模块
python -m mcp_tools.server
```

### Claude Code 集成配置

在 `~/.claude/settings.json` 或项目 `.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "onnxtools": {
      "command": "python",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "/path/to/onnx_vehicle_plate_recognition"
    }
  }
}
```

## 工具列表

当前暴露的工具（精简版，3个）：

| 工具名称 | 功能 | 只读 | 返回类型 |
|---------|------|:----:|---------|
| `onnxtools_detect_objects` | 车辆/车牌目标检测 | ✓ | JSON/Markdown |
| `onnxtools_zoom_to_plate` | **放大车牌返回图像** | ✓ | **MCPImage** |
| `onnxtools_server_status` | 获取服务器状态 | ✓ | JSON |

**如需启用所有工具**，修改 `server.py`：
```python
from .tools import register_all_tools
register_all_tools(mcp)  # 替换 register_selected_tools
```

<details>
<summary>完整工具列表（未暴露）</summary>

| 工具名称 | 功能 | 只读 | 返回类型 |
|---------|------|:----:|---------|
| `onnxtools_recognize_plate` | 车牌 OCR 文字识别 | ✓ | JSON/Markdown |
| `onnxtools_classify_plate_color_layer` | 车牌颜色/层级分类 | ✓ | JSON/Markdown |
| `onnxtools_crop_detections` | 裁剪检测目标区域 | ✗ | JSON/Markdown |
| `onnxtools_annotate_image` | 可视化标注图像 | ✗ | JSON/Markdown |
| `onnxtools_full_pipeline` | 完整检测+OCR流水线 | ✗ | JSON/Markdown |

</details>

## 目录结构

```
mcp_tools/
├── __init__.py           # 包初始化
├── server.py             # FastMCP 服务器定义
├── models.py             # Pydantic 输入/输出模型
├── config.py             # 配置常量
├── tools/
│   ├── __init__.py       # 工具包初始化
│   ├── detection.py      # 检测和流水线工具
│   ├── ocr.py            # OCR 识别工具
│   ├── classification.py # 分类工具
│   └── visualization.py  # 可视化和裁剪工具
├── utils/
│   ├── __init__.py       # 工具函数初始化
│   ├── image_loader.py   # 图像加载（文件/URL/base64）
│   ├── model_manager.py  # 模型懒加载和缓存
│   ├── response_formatter.py  # 响应格式化
│   └── error_handler.py  # 错误处理
└── CLAUDE.md             # 本文档
```

## 工具详细说明

### 1. onnxtools_detect_objects

检测图像中的车辆和车牌。

**输入参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | str | 必需 | 图像路径/URL/base64 |
| `image_source` | enum | "file" | 来源类型: file/url/base64 |
| `model_path` | str | models/rtdetr.onnx | 检测模型路径 |
| `model_type` | enum | "rtdetr" | 模型类型: yolo/rtdetr/rfdetr |
| `conf_threshold` | float | 0.5 | 置信度阈值 |
| `classes` | list | None | 过滤类别（如 ["plate"]） |
| `response_format` | enum | "json" | 输出格式: json/markdown |

**输出示例** (JSON):
```json
{
  "total_detections": 2,
  "detections": [
    {"box": [100, 200, 300, 400], "score": 0.95, "class_id": 0, "class_name": "vehicle"},
    {"box": [150, 350, 250, 400], "score": 0.88, "class_id": 1, "class_name": "plate"}
  ],
  "image_shape": [1080, 1920, 3],
  "model_path": "models/rtdetr.onnx"
}
```

### 2. onnxtools_recognize_plate

识别车牌文字。

**输入参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | str | 必需 | 车牌图像路径 |
| `image_source` | enum | "file" | 来源类型 |
| `model_path` | str | models/ocr.onnx | OCR 模型路径 |
| `is_double_layer` | bool | false | 是否双层车牌 |
| `conf_threshold` | float | 0.5 | 置信度阈值 |
| `response_format` | enum | "json" | 输出格式 |

**输出示例** (JSON):
```json
{
  "text": "京A12345",
  "confidence": 0.95,
  "char_confidences": [0.99, 0.98, 0.95, 0.94, 0.96, 0.93, 0.92],
  "is_double_layer": false
}
```

### 3. onnxtools_classify_plate_color_layer

分类车牌颜色和层级。

**支持的颜色**：blue, yellow, white, black, green
**支持的层级**：single, double

**输出示例** (JSON):
```json
{
  "labels": ["blue", "single"],
  "confidences": [0.95, 0.88],
  "avg_confidence": 0.915
}
```

### 4. onnxtools_crop_detections

裁剪检测到的目标区域。

**特殊参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | None | 保存目录（None 则不保存） |
| `gain` | float | 1.02 | 边框扩展系数 |
| `pad` | int | 10 | 额外填充像素 |

### 5. onnxtools_annotate_image

使用可视化预设标注图像。

**可视化预设**：
| 预设 | 说明 |
|------|------|
| `standard` | 角点边框 + 简单标签 |
| `debug` | 圆角边框 + 置信度条 + 详细标签 |
| `lightweight` | 点标记 + 小标签 |
| `privacy` | 边框 + 模糊车牌 |
| `high_contrast` | 填充区域 + 背景变暗 |

### 6. onnxtools_full_pipeline

完整的检测、分类、OCR 流水线。

**流程**：
1. 检测车辆和车牌
2. 对每个车牌：分类颜色/层级 + OCR 识别
3. 可选：生成标注图像
4. 返回完整结果

**输出示例** (JSON):
```json
{
  "total_detections": 3,
  "vehicles": [
    {"box": [...], "class_name": "vehicle", "confidence": 0.95}
  ],
  "plates": [
    {
      "box": [...],
      "text": "京A12345",
      "color": "blue",
      "layer": "single",
      "ocr_confidence": 0.92,
      "detection_confidence": 0.88
    }
  ],
  "output_path": "output/annotated.jpg"
}
```

### 7. onnxtools_zoom_to_plate

放大并返回检测到的车牌图像，直接返回给大模型查看（不保存文件）。

**特点**：
- 返回 `MCPImage` 类型，大模型可直接"看到"图像
- 自动选择置信度最高的目标
- 与 `crop_detections` 不同，无需保存文件即可查看裁剪结果

**输入参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | str | 必需 | 图像路径/URL/base64 |
| `image_source` | enum | "file" | 来源类型 |
| `model_path` | str | models/rtdetr.onnx | 检测模型路径 |
| `model_type` | enum | "rtdetr" | 模型类型 |
| `conf_threshold` | float | 0.5 | 置信度阈值 |
| `target_class` | str | "plate" | 目标类别: plate/vehicle |
| `gain` | float | 1.02 | 边框扩展系数 |
| `pad` | int | 10 | 额外填充像素 |

**返回值**：
- 成功：`MCPImage` - JPEG 格式的裁剪图像，LLM 可直接查看
- 失败：`str` - 错误消息（如未检测到目标）

**使用示例**：
```
# Claude 可以这样使用此工具：
"请放大这张图片中的车牌，让我看看车牌号"
→ 调用 zoom_to_plate(image_path="车辆图片.jpg", target_class="plate")
→ 返回裁剪后的车牌图像，Claude 可直接查看并识别
```

**与 crop_detections 的区别**：
| 特性 | zoom_to_plate | crop_detections |
|------|---------------|-----------------|
| 返回类型 | MCPImage（可直接查看） | JSON/Markdown |
| 文件保存 | 不保存 | 可选保存 |
| 多目标处理 | 返回最高置信度的一个 | 返回所有匹配目标 |
| 使用场景 | LLM 查看分析 | 批量处理或存档 |

## 模型懒加载

所有模型在首次使用时才加载，后续调用使用缓存：

```python
# 第一次调用：加载模型（~2-5秒）
# 后续调用：使用缓存（<100ms）
```

缓存管理：
- 最大缓存模型数：10
- 超出时自动移除最旧的模型
- 服务器关闭时自动清理

## 错误处理

所有工具返回用户友好的错误消息：

```
Error: File not found - path/to/image.jpg. Please verify the file path exists.
Error: GPU error during detection. Try using CPU provider.
Error: Out of memory. Try reducing image size or using a smaller model.
```

## 测试验证

```bash
# 语法检查
python -c "import mcp_tools"

# 服务器启动测试
python -m mcp_tools.server --help

# 使用 MCP Inspector 交互测试
npx @modelcontextprotocol/inspector
```

## 与 onnxtools 的关系

MCP 工具模块是 onnxtools 的扩展接口层：

```
Claude/LLM
    ↓ MCP Protocol (stdio)
mcp_tools (FastMCP Server)
    ↓ Python API
onnxtools (Core Library)
    ↓ ONNX Runtime
Models (.onnx/.engine)
```

## FAQ

**Q: 为什么使用 stdio 传输？**
A: stdio 适合本地集成，无需网络配置，与 Claude Code 兼容性最好。

**Q: 如何添加新工具？**
A: 在 `tools/` 目录下创建新文件，定义工具函数并在 `__init__.py` 中注册。

**Q: 图像支持哪些格式？**
A: 支持 JPG、JPEG、PNG、BMP、WebP，以及 URL 和 base64 编码。

**Q: 如何调试 MCP 工具？**
A: 使用 `npx @modelcontextprotocol/inspector` 进行交互测试。

---

最后更新: 2025-01-09
