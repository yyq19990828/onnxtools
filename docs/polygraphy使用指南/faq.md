# Polygraphy 常见问题解答 (FAQ)

本文档收集了使用 Polygraphy 过程中最常遇到的问题和解决方案。

## 🚀 安装和环境问题

### Q1: 如何安装 Polygraphy？
**A:** 推荐通过 pip 安装：
```bash
pip install polygraphy
```

对于开发版本：
```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/Polygraphy
make install  # Linux
# 或 .\install.ps1  # Windows
```

### Q2: 提示缺少依赖怎么办？
**A:** 设置自动安装依赖：
```bash
export POLYGRAPHY_AUTOINSTALL_DEPS=1
polygraphy run model.onnx --onnxrt --trt
```

或手动安装常用依赖：
```bash
pip install onnx onnxruntime tensorrt
```

### Q3: 在 Docker 环境中使用注意事项？
**A:** 确保 Docker 容器有 GPU 访问权限：
```bash
docker run --gpus all -it nvcr.io/nvidia/tensorrt:22.12-py3
pip install polygraphy
```

## 🔧 模型转换问题

### Q4: ONNX 转 TensorRT 失败怎么办？
**A:** 按以下步骤排查：

1. **检查模型有效性**：
```bash
polygraphy inspect model model.onnx
polygraphy check lint model.onnx
```

2. **清理模型**：
```bash
polygraphy surgeon sanitize model.onnx --fold-constants --output clean.onnx
```

3. **检查 TensorRT 兼容性**：
```bash
polygraphy inspect model clean.onnx --convert-to trt --verbose
```

4. **使用详细日志**：
```bash
polygraphy convert clean.onnx --convert-to trt --verbose --extra-verbose --output debug.engine
```

### Q5: 动态形状模型转换问题？
**A:** 明确指定所有形状参数：
```bash
polygraphy convert model.onnx --convert-to trt \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[4,3,224,224] \
  --trt-max-shapes input:[8,3,224,224] \
  --output dynamic.engine
```

### Q6: INT8 量化失败怎么解决？
**A:** 常见原因和解决方案：

1. **校准数据不足或质量差**：
```python
# 使用更多代表性数据
def load_data():
    for i in range(500):  # 增加到500张
        # 使用真实数据而非随机数据
        yield {"input": preprocess_real_image(f"image_{i}.jpg")}
```

2. **某些层不支持 INT8**：
```bash
polygraphy convert model.onnx --convert-to trt --int8 \
  --precision-constraints sensitive_layer:fp16 \
  --calibration-cache calib.cache
```

### Q7: 内存不足错误？
**A:** 减少内存占用：
```bash
# 减少工作空间
polygraphy convert model.onnx --convert-to trt --workspace 512M

# 使用更小的最大形状
polygraphy convert model.onnx --convert-to trt \
  --trt-max-shapes input:[4,3,224,224]  # 降低批次大小

# 分批处理大模型
polygraphy surgeon extract model.onnx --inputs input --outputs intermediate
```

## 🎯 精度和推理问题

### Q8: 跨框架精度不匹配？
**A:** 分层调试步骤：

1. **调整容差**：
```bash
polygraphy run model.onnx --onnxrt --trt --rtol 1e-3 --atol 1e-3
```

2. **逐层比较**：
```bash
polygraphy run model.onnx --onnxrt --trt --mark-all --save-outputs layer_outputs.json
```

3. **使用相同精度**：
```bash
polygraphy run model.onnx --onnxrt --trt --tf32  # 禁用混合精度
```

4. **减少问题模型**：
```bash
polygraphy debug reduce model.onnx --output minimal_problem.onnx
```

### Q9: NaN 或 Inf 输出问题？
**A:** 检查和修复步骤：

1. **检查输入数据**：
```bash
polygraphy inspect data inputs.json --show-values --statistics
```

2. **检查权重**：
```bash
polygraphy inspect model model.onnx --show-weights --mode=full
```

3. **使用数值稳定的配置**：
```bash
polygraphy convert model.onnx --convert-to trt --fp16 --tf32
```

### Q10: 推理速度慢？
**A:** 性能优化建议：

1. **使用 FP16**：
```bash
polygraphy convert model.onnx --convert-to trt --fp16
```

2. **增加工作空间**：
```bash
polygraphy convert model.onnx --convert-to trt --workspace 4G
```

3. **优化动态形状**：
```bash
# 设置合适的 opt-shapes
polygraphy convert model.onnx --convert-to trt \
  --trt-opt-shapes input:[typical_batch,3,224,224]
```

4. **使用策略缓存**：
```bash
polygraphy convert model.onnx --convert-to trt \
  --save-tactics tactics.cache  # 第一次构建
polygraphy convert model.onnx --convert-to trt \
  --load-tactics tactics.cache  # 后续构建
```

## 🛠️ 调试和开发问题

### Q11: 如何调试复杂的精度问题？
**A:** 系统性调试流程：

1. **确认问题范围**：
```bash
polygraphy run model.onnx --onnxrt --trt --save-outputs results.json
```

2. **减少模型**：
```bash
polygraphy debug reduce model.onnx --mode=bisect --output reduced.onnx
```

3. **详细分析**：
```bash
polygraphy debug precision reduced.onnx \
  --golden-outputs onnxrt_outputs.json \
  --mark-all --save-layer-outputs analysis.json
```

4. **尝试修复**：
```bash
# 为问题层添加精度约束
polygraphy convert reduced.onnx --convert-to trt \
  --precision-constraints problematic_layer:fp16
```

### Q12: 间歇性推理失败？
**A:** 可能是 TensorRT 策略问题：

1. **多次构建测试**：
```bash
polygraphy debug build model.onnx --num-iterations 10 --save-tactics multiple.json
```

2. **分析策略差异**：
```bash
polygraphy debug diff-tactics build1.json build2.json
```

3. **排除问题策略**：
```bash
polygraphy convert model.onnx --convert-to trt \
  --exclude-tactics bad_tactics.json
```

### Q13: 如何处理自定义算子？
**A:** 自定义算子处理方法：

1. **检查算子支持**：
```bash
polygraphy inspect model model.onnx --convert-to trt --verbose
```

2. **使用插件**：
```python
# 注册自定义插件
import tensorrt as trt
trt.init_libnvinfer_plugins(None, "")
```

3. **替换不支持的算子**：
```bash
# 使用 surgeon 替换或移除算子
polygraphy surgeon prune model.onnx --remove-node-types CustomOp
```

## 📊 数据和格式问题

### Q14: 输入数据格式不匹配？
**A:** 数据预处理检查：

1. **检查模型输入要求**：
```bash
polygraphy inspect model model.onnx --mode=basic
```

2. **验证输入数据**：
```python
import numpy as np
# 确保数据类型匹配
input_data = input_data.astype(np.float32)
# 确保形状匹配
assert input_data.shape == expected_shape
```

3. **使用数据加载器脚本**：
```python
def load_data():
    # 确保预处理步骤正确
    image = cv2.imread("image.jpg")
    image = cv2.resize(image, (224, 224))
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = image / 255.0  # 归一化
    yield {"input": image[np.newaxis, :].astype(np.float32)}
```

### Q15: 模型输出结果异常？
**A:** 输出验证步骤：

1. **检查输出范围**：
```bash
polygraphy inspect data outputs.json --statistics --show-values
```

2. **与参考结果比较**：
```bash
polygraphy inspect diff expected.json actual.json --rtol=1e-3
```

3. **检查后处理步骤**：
```python
# 确保后处理逻辑正确
outputs = model_inference(inputs)
results = postprocess(outputs)  # 检查这一步
```

## 🔍 高级用法问题

### Q16: 如何批量处理多个模型？
**A:** 批量处理脚本示例：

```bash
#!/bin/bash
for model in models/*.onnx; do
    echo "处理: $model"
    model_name=$(basename "$model" .onnx)
    
    # 验证模型
    polygraphy check lint "$model" || continue
    
    # 转换模型
    polygraphy convert "$model" --convert-to trt \
      --fp16 --workspace 2G \
      --output "engines/${model_name}.engine"
    
    # 验证精度
    polygraphy run "$model" --onnxrt \
      --trt-engine "engines/${model_name}.engine" \
      --save-outputs "results/${model_name}.json"
done
```

### Q17: 如何集成到 CI/CD 流水线？
**A:** CI/CD 集成示例：

```yaml
# GitHub Actions 示例
- name: Model Validation
  run: |
    for model in changed_models/*.onnx; do
      # 基础验证
      polygraphy check lint "$model" || exit 1
      
      # 兼容性检查
      polygraphy check compatibility "$model" --onnxrt || exit 1
      
      # 性能基准（可选）
      timeout 300 polygraphy convert "$model" --convert-to trt --workspace 1G || echo "TRT conversion timeout"
    done
```

### Q18: 如何优化大批量推理性能？
**A:** 大批量推理优化：

1. **使用合适的批次大小**：
```bash
# 找到最优批次大小
for bs in 1 2 4 8 16; do
    polygraphy run model.onnx --trt \
      --input-shapes input:[$bs,3,224,224] \
      --warm-up-runs 10
done
```

2. **启用多流处理**：
```python
# TensorRT 多流推理
context.set_optimization_profile(0)
stream1 = cuda.Stream()
stream2 = cuda.Stream()
```

3. **使用内存池**：
```bash
polygraphy convert model.onnx --convert-to trt --pooled-outputs
```

## ⚠️ 故障排除指南

### 常见错误信息及解决方案

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `ONNX model is invalid` | ONNX 模型损坏 | `polygraphy check lint model.onnx` |
| `Unsupported operator` | TensorRT 不支持的算子 | 检查算子兼容性或使用插件 |
| `Out of memory` | GPU 内存不足 | 减少批次大小或工作空间 |
| `Calibration failed` | INT8 校准数据问题 | 检查校准数据质量和数量 |
| `Shape mismatch` | 输入形状不匹配 | 检查模型输入要求和实际数据形状 |

### 获取更多帮助

1. **详细日志**：
```bash
polygraphy <command> --verbose --extra-verbose --log-file debug.log
```

2. **官方资源**：
   - [GitHub Issues](https://github.com/NVIDIA/TensorRT/issues)
   - [官方文档](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/)
   - [TensorRT 开发者论坛](https://forums.developer.nvidia.com/c/accelerated-computing/deep-learning/tensorrt/)

3. **社区支持**：
   - Stack Overflow (标签: `tensorrt`, `polygraphy`)
   - NVIDIA 开发者社区

---

*这个 FAQ 会持续更新，如果遇到新问题，欢迎通过 GitHub Issues 反馈。*