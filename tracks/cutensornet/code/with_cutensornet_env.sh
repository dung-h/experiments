#!/usr/bin/env bash
# Run an arbitrary Python entry point with the cuQuantum CUDA 13 libraries.
# Set CUTENSORNET_PYTHON to reuse an existing environment; otherwise this uses
# the reproducible local `.venv-cutensornet` convention.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_python="${CUTENSORNET_PYTHON:-$task_root/.venv-cutensornet/bin/python}"
task_venv="$(cd "$(dirname "$task_python")/.." 2>/dev/null && pwd || true)"
task_cuda_root="$(find "$task_venv/lib" -type d -path '*/site-packages/nvidia/cu13' -print -quit 2>/dev/null || true)"

if [[ ! -x "$task_python" || ! -d "$task_cuda_root" ]]; then
  echo "Missing cuTensorNet environment. Install cuquantum-python-cu13==26.6.0 and cupy-cuda13x[ctk] in .venv-cutensornet, or set CUTENSORNET_PYTHON." >&2
  exit 1
fi

export CUDA_PATH="$task_cuda_root"
export LD_LIBRARY_PATH="$task_cuda_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$task_python" "$@"
