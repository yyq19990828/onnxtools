# Rider + Vehicle ReID

这个目录预留给“骑行者整体框”的 ReID 模型。

这里的“骑行者”指检测框同时包含人和车，例如：

- 人 + 自行车
- 人 + 电动车
- 人 + 摩托车

这类目标不等同于纯 person ReID。当前 `models/reid/person/` 下的 OSNet / MobileNetV2 / ResNet50 模型主要面向竖直行人 crop，输入多为 `[N,3,256,128]`，直接用于“人+车”宽框时可能会有明显形变，效果需要实测。

推荐后续路线：

1. 先用 person OSNet 作为基线跑一版，看 ID switch 是否下降。
2. 收集本项目骑行者 crop，按轨迹 ID 或人工 ID 做微调数据。
3. 用 OSNet / FastReID / 轻量 CNN 重新训练或微调，导出 ONNX 后放到本目录。
4. 如果检测器能分别输出人和车，可以考虑双分支特征：人上半身/整人特征 + 车体特征，再拼接或加权融合。
