#!/usr/bin/env bash
# AMD ROCm GPU build targeting RDNA 3 (gfx1100).
# For CPU-only or NVIDIA setups, skip this script and use: uv sync
set -euo pipefail

if ! command -v hipcc &>/dev/null && [ ! -d /opt/rocm ]; then
    echo "ERROR: ROCm not found. Install ROCm first or use 'uv sync' for CPU-only." >&2
    exit 1
fi

# pyproject sets no-binary-package for llama-cpp-python, so uv builds it from
# source. Pass the HIP build flags through `uv sync` itself: the resulting wheel
# is cached and reused by later `uv sync` / `uv run` calls, so plain `uv run`
# stays consistent instead of reverting to a CPU build (which is what happens if
# you `uv pip install` over the top of an already-synced environment).
export CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx1100}"
export FORCE_CMAKE=1
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

uv sync

echo "Verifying llama.cpp GPU offload..."
uv run --no-sync python -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'GPU offload NOT available'; print('llama-cpp-python', llama_cpp.__version__, '- GPU offload OK')"
