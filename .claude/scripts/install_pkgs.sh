#!/bin/bash
# SessionStart hook: 自动安装项目依赖
# 用于 Claude Code 远程会话 (Claude Code on the web) 环境初始化

set -e

# 仅在远程环境中运行完整安装
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

echo "[install_pkgs] 远程环境检测到，开始安装依赖..."

# 安装 uv (如果不存在)
if ! command -v uv &>/dev/null; then
  echo "[install_pkgs] 安装 uv 包管理器..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 安装项目依赖
echo "[install_pkgs] 安装项目依赖 (uv sync)..."
cd "$CLAUDE_PROJECT_DIR"

uv sync --extra mcp --no-extra trt

# 持久化虚拟环境和 PATH 到后续 bash 命令
if [ -n "$CLAUDE_ENV_FILE" ]; then
  VENV_DIR="$CLAUDE_PROJECT_DIR/.venv"
  echo "VIRTUAL_ENV=$VENV_DIR" >> "$CLAUDE_ENV_FILE"
  echo "PATH=$VENV_DIR/bin:$HOME/.local/bin:$PATH" >> "$CLAUDE_ENV_FILE"
fi

echo "[install_pkgs] 依赖安装完成"
exit 0
