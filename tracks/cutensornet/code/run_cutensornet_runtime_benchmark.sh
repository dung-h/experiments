#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_venv="$task_root/.venv-cutensornet"
task_cuda_root="$(find "$task_venv/lib" -type d -path '*/site-packages/nvidia/cu13' -print -quit 2>/dev/null || true)"

if [[ ! -x "$task_venv/bin/python" || ! -d "$task_cuda_root" ]]; then
  echo "Missing cuTensorNet environment. Run: python3 -m venv .venv-cutensornet && .venv-cutensornet/bin/pip install 'cuquantum-python-cu13==26.6.0' 'cupy-cuda13x[ctk]'" >&2
  exit 1
fi

export CUDA_PATH="$task_cuda_root"
export LD_LIBRARY_PATH="$task_cuda_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$task_venv/bin/python" "$task_root/tracks/cutensornet/code/run_cutensornet_runtime_benchmark.py" "$@"
