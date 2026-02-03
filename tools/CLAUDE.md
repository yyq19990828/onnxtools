[根目录](../CLAUDE.md) > **tools**

# 调试工具模块 (tools)

## 模块职责

提供模型评估、TensorRT引擎构建、性能分析和调试工具，支持模型优化、精度验证和性能基准测试。

## 入口和启动

- **模型评估**: `eval.py` - COCO数据集评估主入口
- **引擎构建**: `build_engine.py` - TensorRT引擎构建工具
- **性能比较**: `compare_onnx_engine.py` - ONNX vs TensorRT性能对比

## 外部接口

### 模型评估
```bash
# COCO数据集评估
python tools/eval.py \
    --model-path models/rtdetr-2024080100.onnx \
    --model-type rtdetr \
    --dataset-path /path/to/coco \
    --annotations-path /path/to/annotations.json
```

### TensorRT引擎构建
```bash
# 构建TensorRT引擎
python tools/build_engine.py \
    --onnx-path models/rtdetr-2024080100.onnx \
    --engine-path models/rtdetr-2024080100.engine \
    --precision fp16 \
    --max-batch-size 8
```

### 性能对比分析
```bash
# ONNX vs TensorRT性能对比
python tools/compare_onnx_engine.py \
    --onnx-path models/rtdetr-2024080100.onnx \
    --engine-path models/rtdetr-2024080100.engine \
    --input-shape 1,3,640,640 \
    --iterations 100
```

### OCR数据集评估
```bash
# OCR模型评估（表格输出）
python tools/eval_ocr.py \
    --label-file data/ocr_rec_dataset_examples/val.txt \
    --dataset-base data/ocr_rec_dataset_examples \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5

# 深度错误分析
python tools/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --error-analysis error_report.json

# JSON格式导出用于模型比较
python tools/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr_v2.onnx \
    --config configs/plate.yaml \
    --output-format json > results_v2.json
```

## 关键依赖和配置

### 核心依赖
- **tensorrt**: TensorRT引擎构建和推理
- **polygraphy**: NVIDIA模型调试和优化工具
- **onnx**: ONNX模型操作和验证
- **matplotlib**: 性能可视化和图表绘制

### 调试脚本依赖
- **bash**: Shell脚本执行环境
- **nvidia-ml-py**: GPU监控和资源管理

### 配置文件
- `debug/` 目录下的调试配置脚本
- Polygraphy配置模板文件

## 数据模型

### 评估结果格式
```python
evaluation_metrics = {
    'mAP': float,                    # 平均精度
    'mAP_50': float,                 # IoU@0.5的mAP
    'mAP_75': float,                 # IoU@0.75的mAP
    'mAP_small': float,              # 小目标mAP
    'mAP_medium': float,             # 中等目标mAP
    'mAP_large': float,              # 大目标mAP
    'per_class_ap': dict,            # 每类别AP
    'inference_time': float,         # 平均推理时间(ms)
    'total_images': int              # 总测试图像数
}
```

### 性能基准结果
```python
benchmark_result = {
    'onnx_runtime': {
        'mean_latency': float,       # 平均延迟(ms)
        'std_latency': float,        # 延迟标准差
        'throughput': float,         # 吞吐量(FPS)
        'memory_usage': float        # 内存使用(MB)
    },
    'tensorrt_runtime': {
        'mean_latency': float,
        'std_latency': float,
        'throughput': float,
        'memory_usage': float
    },
    'speedup_ratio': float           # 加速比
}
```

### 引擎构建配置
```python
engine_config = {
    'precision': str,                # 'fp32', 'fp16', 'int8'
    'max_batch_size': int,           # 最大批次大小
    'max_workspace_size': int,       # 最大工作空间(bytes)
    'input_shapes': dict,            # 输入形状范围
    'optimization_level': int        # 优化级别 0-5
}
```

## 测试和质量

### 测试覆盖范围
- [ ] 多模型架构评估兼容性
- [ ] TensorRT引擎构建成功率
- [ ] 性能基准测试稳定性
- [ ] 调试脚本执行正确性

### 性能基准
- [ ] COCO评估完成时间 (< 30min for val2017)
- [ ] TensorRT引擎构建时间 (< 10min for typical model)
- [ ] 性能对比测试精度 (误差 < 5%)

### 质量指标
- [ ] 评估结果准确性验证
- [ ] 引擎精度损失监控 (< 1% mAP drop)
- [ ] 调试工具可靠性

## 常见问题 (FAQ)

### Q: TensorRT引擎构建失败怎么解决？
A: 1) 检查ONNX模型兼容性; 2) 验证TensorRT版本匹配; 3) 调整工作空间大小; 4) 使用Polygraphy调试

### Q: 模型评估精度异常怎么排查？
A: 1) 验证数据集标注格式; 2) 检查预处理一致性; 3) 确认类别映射正确; 4) 对比少量样本的推理结果

### Q: 如何使用Polygraphy进行深度调试？
A: 参考 `../docs/polygraphy使用指南/` 目录下的详细文档和示例

### Q: 性能对比结果不稳定怎么办？
A: 1) 增加测试迭代次数; 2) 确保GPU处于稳定状态; 3) 关闭其他GPU进程; 4) 使用固定的输入数据

## 相关文件列表

### 核心工具脚本
- `eval.py` - COCO数据集模型评估主程序
- **`eval_ocr.py`** - OCR数据集评估命令行工具 (支持深度错误分析)
- `build_engine.py` - TensorRT引擎构建工具
- `compare_onnx_engine.py` - ONNX vs TensorRT性能对比
- `network_postprocess.py` - 网络后处理分析工具

### 可视化和分析
- `draw_engine.py` - TensorRT引擎结构可视化
- `layer_statistics.py` - 模型层统计分析
- `tensor_selector.py` - 张量选择和分析工具

### 调试脚本
- `debug/01_debug_subonnx_fp16.sh` - FP16子图调试
- `debug/01_debug_subonnx_fp32.sh` - FP32子图调试
- `debug/02_debug_subonnx_fp32.sh` - 高级FP32调试
- `debug/debug_fp16.sh` - FP16精度调试
- `build.sh` - 批量构建脚本
- `eval.sh` - 批量评估脚本

### 模板和配置
- `debug/data_loader.py.template` - 数据加载器模板

## 变更日志 (Changelog)

**2025-09-15 20:01:23 CST** - 初始化调试工具模块文档，建立模型评估和优化工具规范

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tools/`*
