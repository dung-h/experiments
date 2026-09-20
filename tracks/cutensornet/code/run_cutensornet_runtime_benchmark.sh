#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "$task_root/tracks/cutensornet/code/with_cutensornet_env.sh" \
  "$task_root/tracks/cutensornet/code/run_cutensornet_runtime_benchmark.py" "$@"
