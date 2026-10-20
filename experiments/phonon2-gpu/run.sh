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
export PHONON_HOME="$ROOT/.cache/phonon"
export PHONON2_CUDA_DTYPE=float16
export PHONON2_CUDA_WARMUP=first
export PHONON2_CUDA_PAD_S=5

exec "$ROOT/.venv/bin/python" "$ROOT/experiments/phonon2-gpu/benchmark.py"
