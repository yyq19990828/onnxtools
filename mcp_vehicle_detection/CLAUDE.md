[根目录](../CLAUDE.md) > **mcp_vehicle_detection**

# MCP车辆检测服务模块

## 模块职责

基于模型上下文协议(MCP)的车辆检测服务实现，提供标准化的车辆和车牌检测API接口，支持与外部系统的无缝集成。

## 入口和启动

- **MCP服务器**: `server.py` - MCP协议服务器实现
- **检测服务**: `main.py` - 车辆检测核心服务逻辑
- **快速测试**: `quick_test.py` - 服务功能验证和测试

## 外部接口

### MCP服务启动
```bash
# 启动MCP服务器
python mcp_vehicle_detection/server.py

# 指定配置文件启动
python mcp_vehicle_detection/server.py --config config.yaml
```

### 车辆检测API
```python
# MCP客户端调用示例
import mcp_client

client = mcp_client.connect("stdio://mcp_vehicle_detection/server.py")

# 检测单张图像
result = client.call_tool("detect_vehicle", {
    "image_path": "/path/to/image.jpg",
    "conf_threshold": 0.5,
    "return_crops": True
})

# 批量检测
batch_result = client.call_tool("detect_batch", {
    "image_paths": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
    "conf_threshold": 0.5
})
```

### 配置管理
```yaml
# config.yaml
model:
  detection_model: "../models/rtdetr-2024080100.onnx"
  ocr_model: "../models/ocr.onnx"
  color_model: "../models/color_layer.onnx"

server:
  host: "localhost"
  port: 8080
  max_workers: 4

detection:
  conf_threshold: 0.5
  iou_threshold: 0.5
  max_detections: 100
```

## 关键依赖和配置

### MCP协议依赖
- **mcp**: 模型上下文协议核心库
- **anyio**: 异步I/O支持
- **pydantic**: 数据验证和序列化

### 检测功能依赖
- **父级模块**: 依赖根目录的 `infer_onnx` 和 `utils` 模块
- **配置文件**: `config.yaml` 服务配置
- **第三方集成**: DINO-X-MCP集成模块

### 运行环境
- **Python >= 3.10**
- **异步运行时**: 支持asyncio事件循环
- **GPU支持**: 可选CUDA加速

## 数据模型

### MCP检测请求
```python
class DetectionRequest(BaseModel):
    image_path: str                    # 图像文件路径
    conf_threshold: float = 0.5        # 置信度阈值
    iou_threshold: float = 0.5         # NMS IoU阈值
    return_crops: bool = False         # 是否返回裁剪区域
    include_ocr: bool = True           # 是否进行OCR识别
```

### MCP检测响应
```python
class DetectionResponse(BaseModel):
    status: str                        # 检测状态
    detections: List[Detection]        # 检测结果列表
    processing_time: float             # 处理时间(ms)
    image_info: ImageInfo              # 图像信息

class Detection(BaseModel):
    bbox: List[float]                  # 边界框 [x1, y1, x2, y2]
    confidence: float                  # 置信度
    class_name: str                    # 类别名称
    class_id: int                      # 类别ID
    plate_info: Optional[PlateInfo]    # 车牌信息（如果是车牌）

class PlateInfo(BaseModel):
    text: str                          # 识别文本
    color: str                         # 车牌颜色
    layer: str                         # 车牌层数
    ocr_confidence: float              # OCR置信度
```

### 服务配置模型
```python
class ServerConfig(BaseModel):
    model_config: ModelConfig          # 模型配置
    detection_config: DetectionConfig  # 检测参数
    server_config: NetworkConfig       # 网络配置
```

## 测试和质量

### 功能测试
- [ ] MCP协议通信测试
- [ ] 检测API接口测试
- [ ] 批量处理功能测试
- [ ] 错误处理和异常测试

### 性能测试
- [ ] 并发请求处理能力 (>50 req/s)
- [ ] 内存使用稳定性 (< 4GB for 4 workers)
- [ ] 检测延迟测试 (< 100ms per image)

### 集成测试
- [ ] 与DINO-X-MCP集成测试
- [ ] 第三方MCP客户端兼容性
- [ ] 长时间运行稳定性测试

## 常见问题 (FAQ)

### Q: 什么是MCP协议？
A: 模型上下文协议(Model Context Protocol)是一个标准化的AI模型服务接口协议，支持工具调用和资源管理

### Q: 如何自定义检测参数？
A: 1) 修改 `config.yaml` 配置文件; 2) 通过MCP请求参数动态调整; 3) 使用环境变量覆盖默认配置

### Q: 服务支持多模型切换吗？
A: 是的，可以通过配置文件指定不同的模型路径，或通过MCP接口动态切换模型

### Q: 如何监控服务性能？
A: 1) 查看MCP服务器日志; 2) 使用 `quick_test.py` 进行性能测试; 3) 监控GPU和内存使用率

## 相关文件列表

### 核心服务文件
- `server.py` - MCP协议服务器主程序
- `main.py` - 车辆检测服务核心逻辑
- `quick_test.py` - 服务功能快速测试工具

### 配置和集成
- `config.yaml` - 服务配置文件
- `third-party/DINO-X-MCP/` - DINO-X-MCP集成模块

### 项目管理文件
- `pyproject.toml` - 子项目依赖和配置

## 变更日志 (Changelog)

**2025-09-15 20:01:23 CST** - 初始化MCP车辆检测服务模块文档，建立MCP协议集成规范

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/mcp_vehicle_detection/`*