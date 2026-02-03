#!/bin/bash

# ONNX模型优化和TensorRT引擎构建脚本
# 功能：支持命令行输入模型路径，在同位置生成同名的优化onnx文件和engine文件

set -e  # 遇到错误立即退出

# 显示帮助信息
show_help() {
    echo "用法: $0 <onnx_model_path> [options]"
    echo ""
    echo "参数:"
    echo "  onnx_model_path    输入的ONNX模型文件路径（必需）"
    echo ""
    echo "选项:"
    echo "  --fp32            使用FP32精度（默认FP16）"
    echo "  --no-optimize     跳过ONNX模型优化步骤"
    echo "  --keep-temp       保留临时的优化onnx文件"
    echo "  -h, --help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 models/yolov8s_640.onnx"
    echo "  $0 models/detection.onnx --fp32"
    echo "  $0 /path/to/model.onnx --no-optimize --keep-temp"
}

# 解析命令行参数
ONNX_MODEL=""
USE_FP16=true
OPTIMIZE=true
KEEP_TEMP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --fp32)
            USE_FP16=false
            shift
            ;;
        --no-optimize)
            OPTIMIZE=false
            shift
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        -*)
            echo "错误: 未知选项 $1" >&2
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$ONNX_MODEL" ]]; then
                ONNX_MODEL="$1"
            else
                echo "错误: 只能指定一个ONNX模型文件" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [[ -z "$ONNX_MODEL" ]]; then
    echo "错误: 必须指定ONNX模型文件路径" >&2
    show_help
    exit 1
fi

# 检查输入文件是否存在
if [[ ! -f "$ONNX_MODEL" ]]; then
    echo "错误: ONNX模型文件不存在: $ONNX_MODEL" >&2
    exit 1
fi

# 获取文件路径信息
ONNX_DIR=$(dirname "$ONNX_MODEL")
ONNX_BASENAME=$(basename "$ONNX_MODEL" .onnx)
ONNX_FULLPATH=$(realpath "$ONNX_MODEL")

# 生成输出文件路径（在同目录下）
FOLDED_ONNX="$ONNX_DIR/${ONNX_BASENAME}.onnx"
ENGINE_PATH="$ONNX_DIR/${ONNX_BASENAME}.engine"

echo "========================================="
echo "ONNX模型优化和TensorRT引擎构建"
echo "========================================="
echo "输入模型: $ONNX_FULLPATH"
echo "输出ONNX: $FOLDED_ONNX"
echo "输出引擎: $ENGINE_PATH"
echo "精度模式: $([ "$USE_FP16" = true ] && echo "FP16" || echo "FP32")"
echo "优化ONNX: $([ "$OPTIMIZE" = true ] && echo "是" || echo "否")"
echo "========================================="

# 步骤1: 优化ONNX模型（可选）
if [[ "$OPTIMIZE" = true ]]; then
    echo "步骤1/2: 优化ONNX模型..."
    TEMP_FOLDED="${ONNX_DIR}/.${ONNX_BASENAME}_temp.onnx"

    if ! polygraphy surgeon sanitize "$ONNX_MODEL" -o "$TEMP_FOLDED" --fold-constants; then
        echo "错误: ONNX模型优化失败" >&2
        rm -f "$TEMP_FOLDED"
        exit 1
    fi

    # 将优化后的模型重命名为最终名称
    mv "$TEMP_FOLDED" "$FOLDED_ONNX"
    echo "✓ ONNX模型优化完成: $FOLDED_ONNX"

    INPUT_FOR_ENGINE="$FOLDED_ONNX"
else
    echo "步骤1/2: 跳过ONNX优化，直接使用原始模型"
    INPUT_FOR_ENGINE="$ONNX_MODEL"
fi

# 步骤2: 构建TensorRT引擎
echo "步骤2/2: 构建TensorRT引擎..."

# 构建命令参数
BUILD_ARGS="--onnx-path $INPUT_FOR_ENGINE --engine-path $ENGINE_PATH"
if [[ "$USE_FP16" = true ]]; then
    BUILD_ARGS="$BUILD_ARGS --fp16"
else
    BUILD_ARGS="$BUILD_ARGS --no-fp16"
fi

# 执行引擎构建
if ! python tools/build_engine.py $BUILD_ARGS; then
    echo "错误: TensorRT引擎构建失败" >&2
    exit 1
fi

echo "✓ TensorRT引擎构建完成: $ENGINE_PATH"

# 清理临时文件（如果需要）
if [[ "$OPTIMIZE" = true && "$KEEP_TEMP" = false && "$FOLDED_ONNX" != "$ONNX_MODEL" ]]; then
    echo "清理临时优化文件..."
    # 只有当优化文件不是原始文件时才删除
    if [[ "$FOLDED_ONNX" != "$ONNX_FULLPATH" ]]; then
        rm -f "$FOLDED_ONNX"
        echo "✓ 已删除临时文件: $FOLDED_ONNX"
    fi
fi

echo "========================================="
echo "构建完成！"
echo "输出文件:"
if [[ "$OPTIMIZE" = true && "$KEEP_TEMP" = true ]]; then
    echo "  优化ONNX: $FOLDED_ONNX"
fi
echo "  TensorRT引擎: $ENGINE_PATH"
echo "========================================="
