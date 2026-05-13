# MCP 集成

`mcp_tools/` 是基于 [FastMCP](https://github.com/jlowin/fastmcp) 的 Model Context Protocol 服务器,把 onnxtools 推理能力暴露给 Claude / 任何兼容 MCP 的 LLM。

## 安装与启动

```bash
uv pip install -e ".[mcp]"

# 启动(stdio 传输,适合本地集成)
onnxtools-mcp                  # 入口脚本
python -m mcp_tools.server     # 等价
```

Claude Code 配置 (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "onnxtools": {
      "command": "python",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "/absolute/path/to/onnxtools"
    }
  }
}
```

## 默认暴露的工具(精简版,3 个)

| 工具 | 功能 | 返回 |
|---|---|---|
| `onnxtools_detect_objects` | 车辆/车牌检测 | JSON / Markdown |
| `onnxtools_zoom_to_plate` | 裁剪并直接返回车牌图给 LLM 查看 | MCPImage |
| `onnxtools_server_status` | 服务器状态、模型缓存 | JSON |

如需启用全部 8 个工具(`recognize_plate` / `classify_plate_color_layer` /
`crop_detections` / `annotate_image` / `full_pipeline`),将 `server.py`
中的 ``register_selected_tools`` 替换为 ``register_all_tools``。

## 模型缓存

所有模型懒加载,首次调用 ~2-5s,后续 <100ms。缓存上限 10 个,LRU 淘汰。

## 输入图像来源

支持三种 `image_source`:`file` / `url` / `base64`,在所有工具中均可用。

## 调试

```bash
# 用 MCP Inspector 交互测试
npx @modelcontextprotocol/inspector python -m mcp_tools.server
```

详见 `mcp_tools/CLAUDE.md` 中的工具完整契约说明。
