#!/bin/bash
# Python 文件自动格式化 hook
# 从 stdin 读取 JSON 输入，提取文件路径，对 .py 文件运行 ruff

# 读取 stdin JSON
INPUT=$(cat)

# 提取文件路径 (兼容 Edit 和 Write 工具)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
file_path = data.get('tool_input', {}).get('file_path', '')
print(file_path)
" 2>/dev/null)

# 检查是否是 Python 文件
if [[ "$FILE_PATH" == *.py ]] && [[ -f "$FILE_PATH" ]]; then
    # 运行 ruff format 和 ruff check --fix
    ruff format "$FILE_PATH" 2>/dev/null
    ruff check --fix "$FILE_PATH" 2>/dev/null
fi

# 始终返回成功，避免阻塞编辑操作
exit 0
