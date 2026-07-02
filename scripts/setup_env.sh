#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "[1/4] 创建 Python 虚拟环境: $VENV_DIR"
if [[ -d "$VENV_DIR" && ( ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/bin/activate" ) ]]; then
  echo "发现不完整的虚拟环境，先删除后重建: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if ! python3 -m venv "$VENV_DIR"; then
    echo "标准 venv 创建失败，尝试使用 --without-pip 创建基础虚拟环境"
    python3 -m venv --without-pip "$VENV_DIR"
  fi
fi

echo "[2/4] 激活虚拟环境"
source "$VENV_DIR/bin/activate"

echo "[3/4] 升级 pip/setuptools/wheel"
if ! python -m pip --version >/dev/null 2>&1; then
  echo "虚拟环境里没有 pip，正在下载 get-pip.py 引导安装 pip"
  GET_PIP="$PROJECT_DIR/.get-pip.py"
  python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

urlretrieve("https://bootstrap.pypa.io/get-pip.py", Path(".get-pip.py"))
PY
  python "$GET_PIP"
  rm -f "$GET_PIP"
fi
python -m pip install --upgrade pip setuptools wheel

echo "[4/4] 安装项目依赖"
python -m pip install -r "$PROJECT_DIR/requirements.txt"
echo "[额外步骤] 安装 mamba-ssm"
# mamba-ssm 的构建脚本会 import torch，因此必须先安装 torch，再关闭 build isolation。
if ! python -m pip install --no-build-isolation mamba-ssm; then
  echo ""
  echo "警告：mamba-ssm 安装失败。常见原因是当前环境没有 nvcc，且 PyPI 没有匹配的预编译 wheel。"
  echo "benchmark 仍可运行 torch_baseline；官方 Mamba/Mamba2 会在报告中标为 skipped。"
fi

echo ""
echo "环境安装完成。下一步运行："
echo "source .venv/bin/activate"
echo "python benchmarks/benchmark_mamba_blocks.py --quick"
