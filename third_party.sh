#!/bin/bash

# 第三方库初始化脚本
# 使用方法:
#   ./third_party.sh          # 默认模式，只在库不存在时拉取
#   ./third_party.sh --force  # 强制模式，删除已有库并重新拉取

set -e  # 遇到错误时退出

# 解析命令行参数
FORCE_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_MODE=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --force, -f    强制重新拉取所有第三方库"
            echo "  --help,  -h    显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建 third_party 目录
mkdir -p third_party
cd third_party

echo "开始初始化第三方库..."
if [ "$FORCE_MODE" = true ]; then
    echo "⚠️  强制模式：将删除已有库并重新拉取"
fi

# 拉取 TensorRT 的 Polygraphy 和 trt-engine-explorer
echo "正在处理 TensorRT 工具..."
if [ "$FORCE_MODE" = true ]; then
    if [ -d "Polygraphy" ]; then
        echo "删除已有的 Polygraphy 目录..."
        rm -rf Polygraphy
    fi
    if [ -d "trt-engine-explorer" ]; then
        echo "删除已有的 trt-engine-explorer 目录..."
        rm -rf trt-engine-explorer
    fi
fi

if [ ! -d "Polygraphy" ] || [ ! -d "trt-engine-explorer" ]; then
    echo "拉取 TensorRT Polygraphy 和 trt-engine-explorer..."
    # 临时克隆TensorRT仓库
    git clone --no-checkout https://github.com/NVIDIA/TensorRT.git temp_tensorrt
    cd temp_tensorrt
    git sparse-checkout init --cone
    git sparse-checkout set tools/Polygraphy tools/experimental/trt-engine-explorer
    git checkout main
    
    # 移动到third_party目录
    if [ -d "tools/Polygraphy" ]; then
        mv tools/Polygraphy ../Polygraphy
    fi
    if [ -d "tools/experimental/trt-engine-explorer" ]; then
        mv tools/experimental/trt-engine-explorer ../trt-engine-explorer
    fi
    
    cd ..
    rm -rf temp_tensorrt
    echo "✅ TensorRT 工具拉取完成"
else
    echo "📁 TensorRT 工具已存在，跳过"
fi

# 拉取 Ultralytics
echo "正在处理 Ultralytics..."
if [ "$FORCE_MODE" = true ] && [ -d "ultralytics" ]; then
    echo "删除已有的 ultralytics 目录..."
    rm -rf ultralytics
fi

if [ ! -d "ultralytics" ]; then
    echo "拉取 Ultralytics..."
    git clone --depth 1 https://github.com/ultralytics/ultralytics.git temp_ultralytics
    mv temp_ultralytics/ultralytics ultralytics
    rm -rf temp_ultralytics
    echo "✅ Ultralytics 拉取完成"
else
    echo "📁 Ultralytics 已存在，跳过"
fi

# 拉取 RF-DETR
echo "正在处理 RF-DETR..."
if [ "$FORCE_MODE" = true ] && [ -d "rfdetr" ]; then
    echo "删除已有的 rfdetr 目录..."
    rm -rf rfdetr
fi

if [ ! -d "rfdetr" ]; then
    echo "拉取 RF-DETR..."
    git clone --depth 1 https://github.com/roboflow/rf-detr.git temp_rfdetr
    mv temp_rfdetr/rfdetr rfdetr
    rm -rf temp_rfdetr
    echo "✅ RF-DETR 拉取完成"
else
    echo "📁 RF-DETR 已存在，跳过"
fi

# 拉取 MCP-Vision
echo "正在处理 MCP-Vision..."
if [ "$FORCE_MODE" = true ] && [ -d "mcp-vision" ]; then
    echo "删除已有的 mcp-vision 目录..."
    rm -rf mcp-vision
fi

if [ ! -d "mcp-vision" ]; then
    echo "拉取 MCP-Vision..."
    git clone --depth 1 https://github.com/groundlight/mcp-vision.git mcp-vision
    echo "✅ MCP-Vision 拉取完成"
else
    echo "📁 MCP-Vision 已存在，跳过"
fi

# 返回原目录
cd ..

echo ""
echo "🎉 第三方库初始化完成！"
echo "📂 库存储位置: third_party/"
echo "   ├── Polygraphy/          # TensorRT 工具包"
echo "   ├── trt-engine-explorer/ # TensorRT 引擎分析工具"
echo "   ├── ultralytics/         # YOLO 框架"
echo "   ├── rfdetr/              # RF-DETR 检测框架"
echo "   └── mcp-vision/          # MCP 视觉工具服务器"