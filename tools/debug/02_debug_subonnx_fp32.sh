#!/bin/bash
# step 2 线性法定位 step 1的ONNX模型中精度问题的节点

# 参数解析
CONTINUE_MODE=false
model_basename=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            model_basename="$1"
            shift
            ;;
    esac
done

if [ -z "$model_basename" ]; then
    echo "Usage: $0 [--continue] <model_basename>"
    echo "Example: $0 yolov8s_640"
    echo "Example: $0 --continue yolov8s_640  # Continue from previous debug session"
    echo "Note: This script uses the reduced ONNX model from step 1"
    exit 1
fi

RUN="runs"
model_dir="models"
debug_dir="DEBUG/FP32/${model_basename}"
reduced_onnx_model="${debug_dir}/${model_basename}_reduced.onnx"

# 检查step 1的输出文件是否存在
if [ ! -f "${reduced_onnx_model}" ]; then
    echo "Error: Reduced ONNX model not found: ${reduced_onnx_model}"
    echo "Please run 01_debug_subonnx_fp32.sh first to generate the reduced model"
    exit 1
fi

# 确保DEBUG目录存在
mkdir -p "${debug_dir}"

# 检查中间文件是否已存在，智能跳过已完成的步骤
step2_inputs_file="${debug_dir}/step2_inputs.json"
step2_golden_file="${debug_dir}/step2_layerwise_golden.json"
step2_combined_file="${debug_dir}/step2_layerwise_inputs.json"
step2_replay_file="${debug_dir}/step2_polygraphy_debug_replay.json"

# 步骤1: 生成ONNX Runtime的中间结果作为FP32的参考值（使用reduced模型）
if [ -f "$step2_inputs_file" ] && [ -f "$step2_golden_file" ]; then
    echo "✓ Step 1: Skipping ONNX Runtime inference on reduced model (files already exist)"
else
    echo "→ Step 1: Generating ONNX Runtime intermediate results for reduced model..."
    polygraphy run ${reduced_onnx_model} --onnxrt \
        --save-inputs "$step2_inputs_file" \
        --onnx-outputs mark all --save-outputs "$step2_golden_file"
fi

# 步骤2: 合并输入和参考输出
if [ -f "$step2_combined_file" ]; then
    echo "✓ Step 2: Skipping input combination (file already exists)"
else
    echo "→ Step 2: Combining inputs and outputs for step 2..."
    polygraphy data to-input "$step2_inputs_file" "$step2_golden_file" -o "$step2_combined_file"
fi

# 步骤3: 逐层调试（线性模式）
echo "→ Step 3: Running linear debug reduce..."

debug_reduce_cmd="polygraphy debug reduce ${reduced_onnx_model} \
            -o \"${debug_dir}/${model_basename}_linear_reduced.onnx\" \
            --mode=linear \
            --load-inputs \"$step2_combined_file\" \
            --save-debug-replay \"$step2_replay_file\""

# 根据--continue参数决定是否加载现有的replay文件
if [ "$CONTINUE_MODE" = true ] && [ -f "$step2_replay_file" ]; then
    echo "  Continue mode: Loading existing debug replay..."
    debug_reduce_cmd="$debug_reduce_cmd --load-debug-replay \"$step2_replay_file\""
fi

debug_reduce_cmd="$debug_reduce_cmd --check polygraphy run polygraphy_debug.onnx \
                    --trt \
                    --load-inputs \"$step2_combined_file\" \
                    --load-outputs \"$step2_golden_file\""

eval $debug_reduce_cmd

echo "✅ Linear debug process completed. All intermediate files saved in: ${debug_dir}"
echo "✅ Final linear reduced ONNX model saved as: ${debug_dir}/${model_basename}_linear_reduced.onnx"