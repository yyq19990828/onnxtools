#!/bin/bash

# 使用debug预设以显示完整的OCR车牌信息
# debug预设包含: 圆角框 + 置信度条 + 详细标签(包含OCR文字、颜色、层数)
python examples/demo_pipeline.py \
    --model-path models/rtdetr-2024080100.onnx \
    --model-type rtdetr \
    --input data/苏州图片 \
    --output-mode save \
    --conf-thres 0.7 \
    --annotator-preset debug
    # --save-json \
    # --save-frame

# 其他可用的预设:
# --annotator-preset standard      # 标准模式: 边框角点 + 简单标签
# --annotator-preset debug          # 调试模式: 圆角框 + 置信度条 + 详细OCR信息 (推荐)
# --annotator-preset lightweight    # 轻量模式: 点标记 + 小字标签
# --annotator-preset privacy        # 隐私模式: 边框 + 车牌模糊
# --annotator-preset high_contrast  # 高对比: 区域填充 + 背景变暗
