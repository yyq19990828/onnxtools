#!/bin/bash
# step 1 二分法定位ONNX模型中导致精度问题的节点

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
    exit 1
fi

RUN="runs"
model_dir="models"
onnx_model="${model_dir}/${model_basename}.onnx"
debug_dir="DEBUG/${model_basename}"

# 创建DEBUG目录
mkdir -p "${debug_dir}"

# 检查中间文件是否已存在，智能跳过已完成的步骤
inputs_file="${debug_dir}/inputs.json"
golden_file="${debug_dir}/layerwise_golden.json"
combined_file="${debug_dir}/layerwise_inputs.json"
replay_file="${debug_dir}/polygraphy_debug_replay.json"

# 步骤1: 生成ONNX Runtime的中间结果作为FP32的参考值
if [ -f "$inputs_file" ] && [ -f "$golden_file" ]; then
    echo "✓ Step 1: Skipping ONNX Runtime inference (files already exist)"
else
    echo "→ Step 1: Generating ONNX Runtime intermediate results..."
    polygraphy run ${onnx_model} --onnxrt \
        --save-inputs "$inputs_file" \
        --onnx-outputs mark all --save-outputs "$golden_file"
fi

# 步骤2: 合并输入和参考输出
if [ -f "$combined_file" ]; then
    echo "✓ Step 2: Skipping input combination (file already exists)"
else
    echo "→ Step 2: Combining inputs and outputs..."
    polygraphy data to-input "$inputs_file" "$golden_file" -o "$combined_file"
fi

# 步骤3: 逐层调试
echo "→ Step 3: Running bisect debug reduce..."

debug_reduce_cmd="polygraphy debug reduce ${onnx_model} \
            -o \"${debug_dir}/${model_basename}_reduced.onnx\" \
            --mode=bisect \
            --load-inputs \"$combined_file\" \
            --save-debug-replay \"$replay_file\""

# 根据--continue参数决定是否加载现有的replay文件
if [ "$CONTINUE_MODE" = true ] && [ -f "$replay_file" ]; then
    echo "  Continue mode: Loading existing debug replay..."
    debug_reduce_cmd="$debug_reduce_cmd --load-debug-replay \"$replay_file\""
fi

debug_reduce_cmd="$debug_reduce_cmd --check polygraphy run polygraphy_debug.onnx \
                    --trt \
                    --load-inputs \"$combined_file\" \
                    --load-outputs \"$golden_file\""

eval $debug_reduce_cmd

echo "✅ Debug process completed. All intermediate files saved in: ${debug_dir}"
echo "✅ Reduced ONNX model saved as: ${debug_dir}/${model_basename}_reduced.onnx"