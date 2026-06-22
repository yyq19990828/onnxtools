[根目录](../CLAUDE.md) > **mcp_tools**

# MCP 工具模块 (mcp_tools)

## 模块职责

基于 FastMCP 的 MCP (Model Context Protocol) 服务器，把 onnxtools 的 ONNX 推理能力（车辆/车牌检测、OCR、颜色/层级分类、可视化）暴露为 LLM 可调用的工具。作为 onnxtools 核心库之上的接口层，通过 stdio 传输与 Claude Code 集成。

## 启动方式

```bash
uv pip install -e ".[mcp]"   # 安装可选依赖（mcp、httpx）

onnxtools-mcp                 # 入口脚本（= server.main）
python -m mcp_tools.server    # 或用模块方式
```

入口为 `server.py` 的 `main()`，使用 `mcp.run(transport="stdio")`。

### Claude Code 集成

在 `~/.claude/settings.json` 或项目 `.claude/settings.json` 添加：

```json
{
  "mcpServers": {
    "onnxtools": {
      "command": "python",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "/path/to/onnxtools"
    }
  }
}
```

## 暴露的 MCP 工具

`server.py` 通过 `register_selected_tools(mcp)` + 内联定义，默认仅暴露 3 个工具：

| 工具名称 | 功能 | 返回类型 |
|---------|------|---------|
| `onnxtools_detect_objects` | 车辆/车牌目标检测 | JSON / Markdown |
| `onnxtools_zoom_to_plate` | 放大最高置信度目标并返回图像，LLM 可直接查看 | MCPImage（失败为 str） |
| `onnxtools_server_status` | 服务器状态与模型缓存信息 | JSON |

完整工具集（未暴露，定义在 `tools/`，需改用 `register_all_tools`）：
`onnxtools_recognize_plate`（OCR）、`onnxtools_classify_plate_color_layer`（颜色/层级）、`onnxtools_crop_detections`（裁剪保存）、`onnxtools_annotate_image`（可视化预设标注）、`onnxtools_full_pipeline`（检测+分类+OCR 全流程）。

启用全部工具：在 `server.py` 中将 `register_selected_tools(mcp)` 替换为 `register_all_tools(mcp)`。

## 目录结构

```
mcp_tools/
├── server.py             # FastMCP 服务器 + 工具注册 + main() 入口
├── config.py             # SERVER_NAME 等配置常量
├── models.py             # Pydantic 输入/输出模型
├── tools/                # 工具实现
│   ├── detection.py      # 检测、zoom、full_pipeline
│   ├── ocr.py            # OCR 识别
│   ├── classification.py # 颜色/层级分类
│   └── visualization.py  # 可视化、裁剪
└── utils/
    ├── image_loader.py   # 图像加载（文件 / URL / base64）
    ├── model_manager.py  # 模型懒加载 + 缓存
    ├── response_formatter.py
    └── error_handler.py
```

## 关键约定

- **图像输入**：所有工具的 `image_path` 配合 `image_source` 枚举（`file` / `url` / `base64`），支持 JPG/PNG/BMP/WebP。
- **模型懒加载**：模型首次调用才加载（~2-5s），之后命中缓存（<100ms）。缓存上限 10 个，满则淘汰最旧；服务器关闭时 `clear_cache()` 自动清理（见 `server.py` 的 `app_lifespan`）。
- **检测默认值**：`model_type` 默认 `rtdetr`（可选 `yolo`/`rfdetr`），`model_path` 默认 `models/rtdetr.onnx`，`conf_threshold` 默认 `0.5`。
- **裁剪/zoom 参数**：`gain`（边框扩展系数，默认 1.02）、`pad`（额外填充像素，默认 10）。
- **错误处理**：所有工具经 `error_handler` 返回用户友好的错误字符串，不抛裸异常。
- **新增工具**：在 `tools/` 下新建文件定义函数，并在 `tools/__init__.py` 注册。

## 与 onnxtools 的关系

```
Claude/LLM → MCP (stdio) → mcp_tools (FastMCP) → onnxtools (Python API) → ONNX Runtime → 模型
```

## 调试

```bash
python -c "import mcp_tools"        # 导入/语法检查
python -m mcp_tools.server --help   # 启动测试
npx @modelcontextprotocol/inspector # 交互式 MCP Inspector
```
