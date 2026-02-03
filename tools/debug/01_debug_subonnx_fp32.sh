#!/bin/bash
# step 1 二分法定位ONNX模型中导致精度问题的节点

# 参数解析
CONTINUE_MODE=false
onnx_model=""
input_size=640  # 默认尺寸

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        --size|-s)
            input_size="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            onnx_model="$1"
            shift
            ;;
    esac
done

if [ -z "$onnx_model" ]; then
    echo "Usage: $0 [OPTIONS] <onnx_model_path>"
    echo ""
    echo "Options:"
    echo "  --continue        Continue from previous debug session"
    echo "  --size, -s SIZE   Input size for model (default: 640)"
    echo ""
    echo "Examples:"
    echo "  $0 models/yolov8s_640.onnx"
    echo "  $0 --size 1280 models/yolov8s_1280.onnx"
    echo "  $0 --continue models/yolov8s_640.onnx"
    exit 1
fi

# 自动解析模型路径
model_dir=$(dirname "$onnx_model")
model_filename=$(basename "$onnx_model")
model_basename="${model_filename%.onnx}"

# 验证模型文件是否存在
if [ ! -f "$onnx_model" ]; then
    echo "Error: ONNX model not found: $onnx_model"
    exit 1
fi

echo "Model info:"
echo "  - Path: $onnx_model"
echo "  - Directory: $model_dir"
echo "  - Basename: $model_basename"
echo "  - Input size: ${input_size}x${input_size}"

RUN="runs"
debug_dir="DEBUG/FP32/${model_dir}/${model_basename}"

# 创建DEBUG目录
mkdir -p "${debug_dir}"

# 生成数据加载器函数
generate_data_loader() {
    local size=$1
    local template_file="tools/debug/data_loader.py.template"
    local output_file="${debug_dir}/data_loader.py"

    # 确保模板文件存在
    if [ ! -f "$template_file" ]; then
        echo "Error: Template file not found: $template_file"
        exit 1
    fi

    # 使用 sed 替换占位符
    sed "s/{{INPUT_SIZE}}/$size/g" "$template_file" > "$output_file"

    echo "✓ Generated data_loader.py with input size: ${size}x${size}"
}

# 生成自定义数据加载器
echo "→ Generating custom data loader..."
generate_data_loader $input_size

# 检查中间文件是否已存在，智能跳过已完成的步骤
inputs_file="${debug_dir}/inputs.json"
golden_file="${debug_dir}/layerwise_golden.json"
combined_file="${debug_dir}/layerwise_inputs.json"
replay_file="${debug_dir}/polygraphy_debug_replay.json"
folded_onnx="${debug_dir}/folded.onnx"

# 步骤0: 优化ONNX模型
polygraphy surgeon sanitize ${onnx_model} -o ${folded_onnx} --fold-constants

# 步骤1: 生成ONNX Runtime的中间结果作为FP32的参考值
if [ -f "$inputs_file" ] && [ -f "$golden_file" ]; then
    echo "✓ Step 1: Skipping ONNX Runtime inference (files already exist)"
else
    echo "→ Step 1: Generating ONNX Runtime intermediate results..."
    polygraphy run ${folded_onnx} --onnxrt \
        --data-loader-script "${debug_dir}/data_loader.py" --save-inputs "$inputs_file" \
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

debug_reduce_cmd="polygraphy debug reduce ${folded_onnx} \
            -o \"${debug_dir}/${model_basename}_reduced.onnx\" \
            --mode=bisect \
            --show-output \
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
                    --load-outputs \"$golden_file\" \
                    --check-error-stat quantile \
                    --atol 1e-2 --rtol 1e-2 \
                    --error-quantile 0.95"

eval $debug_reduce_cmd

echo "✅ Debug process completed. All intermediate files saved in: ${debug_dir}"
echo "✅ Reduced ONNX model saved as: ${debug_dir}/${model_basename}_reduced.onnx"
