#!/usr/bin/env bash

# Interactive clean export script based on rsync.
# Copies a source tree while excluding git metadata, caches, virtualenvs,
# build outputs, and large runtime artifacts by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_SOURCE="${REPO_ROOT}"
DEFAULT_DEST="${HOME}/onnxtools-export"
DEFAULT_REMOTE_USER="${USER}"
DEFAULT_REMOTE_PATH="~/onnxtools-export"

EXCLUDE_PATTERNS=(
  ".git/"
  ".svn/"
  ".hg/"
  ".DS_Store"
  "Thumbs.db"
  "__pycache__/"
  "*.py[cod]"
  "*.pyo"
  "*.pyd"
  ".pytest_cache/"
  ".ruff_cache/"
  ".mypy_cache/"
  ".pyre/"
  ".cache/"
  ".tox/"
  ".nox/"
  ".coverage"
  "htmlcov/"
  ".ipynb_checkpoints/"
  ".venv/"
  "venv/"
  "env/"
  "ENV/"
  ".env"
  "*.egg-info/"
  ".eggs/"
  "build/"
  "dist/"
  "site/"
  "runs/"
  "DEBUG/"
  "*.log"
  "*.tmp"
  "*.temp"
  "*.bak"
  "*.swp"
  "*.swo"
  "*.onnx"
  "*.engine"
  "*.plan"
  "*.trt"
  "*.pt"
  "*.pth"
  "*.weights"
  "*.ckpt"
  "*.h5"
  "*.pb"
  "*.tflite"
  "models/"
  "data/"
  "third_party/"
)

print_header() {
  echo "========================================="
  echo "onnxtools clean rsync export"
  echo "========================================="
}

ask_text() {
  local prompt="$1"
  local default_value="$2"
  local value

  read -r -p "${prompt} [${default_value}]: " value
  if [[ -z "${value}" ]]; then
    printf '%s\n' "${default_value}"
  else
    printf '%s\n' "${value}"
  fi
}

ask_yes_no() {
  local prompt="$1"
  local default_value="$2"
  local value
  local suffix

  if [[ "${default_value}" == "y" ]]; then
    suffix="Y/n"
  else
    suffix="y/N"
  fi

  while true; do
    read -r -p "${prompt} [${suffix}]: " value
    value="${value:-${default_value}}"
    case "${value}" in
      y|Y|yes|YES) return 0 ;;
      n|N|no|NO) return 1 ;;
      *) echo "请输入 y 或 n" ;;
    esac
  done
}

is_remote_path() {
  [[ "$1" == *:* ]]
}

quote_remote_path() {
  local remote_path="$1"
  local rest

  if [[ "${remote_path}" == "~"* ]]; then
    rest="${remote_path#~}"
    printf "~"
    printf "%q" "${rest}"
  else
    printf "%q" "${remote_path}"
  fi
}

build_ssh_cmd() {
  local port="$1"
  local identity_file="$2"
  local ssh_cmd=("ssh")

  if [[ -n "${port}" ]]; then
    ssh_cmd+=("-p" "${port}")
  fi

  if [[ -n "${identity_file}" ]]; then
    ssh_cmd+=("-i" "${identity_file}")
  fi

  printf '%q ' "${ssh_cmd[@]}"
}

build_exclude_args() {
  local pattern
  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    printf '%s\0%s\0' "--exclude" "${pattern}"
  done
}

run_rsync() {
  local source_dir="$1"
  local dest_dir="$2"
  local delete_extra="$3"
  local ssh_cmd="${4:-}"
  local -a args
  local -a exclude_args

  args=(-aH --human-readable --info=stats2,progress2)

  if [[ -n "${ssh_cmd}" ]]; then
    args+=(-e "${ssh_cmd}")
  fi

  if [[ "${delete_extra}" == "true" ]]; then
    args+=(--delete)
  fi

  while IFS= read -r -d '' item; do
    exclude_args+=("${item}")
  done < <(build_exclude_args)

  rsync "${args[@]}" "${exclude_args[@]}" "${source_dir%/}/" "${dest_dir%/}/"
}

ensure_remote_dest() {
  local remote_host="$1"
  local remote_path="$2"
  local ssh_port="$3"
  local identity_file="$4"
  local ssh_args=()

  if [[ -n "${ssh_port}" ]]; then
    ssh_args+=("-p" "${ssh_port}")
  fi

  if [[ -n "${identity_file}" ]]; then
    ssh_args+=("-i" "${identity_file}")
  fi

  ssh "${ssh_args[@]}" "${remote_host}" "mkdir -p $(quote_remote_path "${remote_path}")"
}

main() {
  print_header

  if ! command -v rsync >/dev/null 2>&1; then
    echo "错误: 未找到 rsync，请先安装 rsync。" >&2
    exit 1
  fi

  local source_dir
  local dest_dir
  local sync_remote=false
  local remote_host
  local remote_user
  local remote_path
  local ssh_port
  local identity_file
  local ssh_cmd=""
  local delete_extra=false

  source_dir="$(ask_text "源目录" "${DEFAULT_SOURCE}")"
  if ask_yes_no "是否同步到远程服务器" "y"; then
    sync_remote=true
    remote_host="$(ask_text "远程服务器 Host/IP" "192.168.1.10")"
    remote_user="$(ask_text "远程用户名" "${DEFAULT_REMOTE_USER}")"
    remote_path="$(ask_text "远程目标目录" "${DEFAULT_REMOTE_PATH}")"
    ssh_port="$(ask_text "SSH 端口，留空表示默认 22" "")"
    identity_file="$(ask_text "SSH 私钥路径，留空使用默认配置" "")"
    dest_dir="${remote_user}@${remote_host}:${remote_path}"
    ssh_cmd="$(build_ssh_cmd "${ssh_port}" "${identity_file}")"
  else
    dest_dir="$(ask_text "本地目标目录" "${DEFAULT_DEST}")"
  fi

  if [[ ! -d "${source_dir}" ]]; then
    echo "错误: 源目录不存在: ${source_dir}" >&2
    exit 1
  fi

  if ask_yes_no "是否删除目标中源目录没有的多余文件" "n"; then
    delete_extra=true
  fi

  echo ""
  echo "源目录: ${source_dir}"
  echo "目标:   ${dest_dir}"
  if [[ "${sync_remote}" == "true" ]]; then
    echo "SSH:    ${ssh_cmd:-ssh}"
  fi
  echo "delete: ${delete_extra}"
  echo ""
  echo "将排除 git 元数据、缓存、虚拟环境、构建产物、运行输出、模型和数据目录。"
  echo ""

  if [[ "${sync_remote}" == "true" ]]; then
    echo "检查远程目标目录..."
    ensure_remote_dest "${remote_user}@${remote_host}" "${remote_path}" "${ssh_port}" "${identity_file}"
  elif ! is_remote_path "${dest_dir}"; then
    mkdir -p "${dest_dir}"
  fi

  echo "开始同步..."
  run_rsync "${source_dir}" "${dest_dir}" "${delete_extra}" "${ssh_cmd}"
  echo "完成: ${dest_dir}"
}

main "$@"
