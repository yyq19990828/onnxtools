# Polygraphy Reshape Destroyer

一个示例 `polygraphy run` 扩展模块，可以将 ONNX 模型中的无操作 `Reshape` 节点
替换为 `Identity` 节点并运行推理。
