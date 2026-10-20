#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p .cache/pip .cache/tmp .cache/huggingface .cache/fermion .cache/phonon \
  .cache/xdg .cache/cuda .cache/torch .cache/torch_extensions .cache/torchinductor .cache/triton
export HF_HOME="$ROOT/.cache/huggingface"
export FERMION_CACHE_DIR="$ROOT/.cache/fermion"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TMPDIR="$ROOT/.cache/tmp"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export TORCH_HOME="$ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"

python3 -m venv .venv
"$ROOT/.venv/bin/python" -m pip install --no-input \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  -r experiments/phonon2-gpu/requirements.lock
