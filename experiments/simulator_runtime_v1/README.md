# Matched dense-statevector runtime matrix v1

**Status:** completed local baseline on 20 September 2026. The 96-row raw
matrix, evaluation and report are in
[`artifacts/simulator_runtime_v1/torch_dense_statevector_v1/`](../../artifacts/simulator_runtime_v1/torch_dense_statevector_v1/).

This track creates actual runtime labels for one deliberately narrow target:
`prepared_execute_reset_state`. A measurement allocates a fresh zero state,
applies a materialized dense-statevector gate schedule, then evaluates
`<Z_0>`. Circuit construction, gate materialization, compiler/transpiler work,
sampling and cloud/QPU timing are retained outside that target.

The runner uses one PyTorch dense-statevector reference implementation for all
supported contexts. Thus `CPU/GPU × complex64/complex128` is a matched matrix
when CUDA is available. It is not a Qiskit Aer, CUDA-Q or cuTensorNet timing
table, and its labels must not be pooled with those tracks.

## Matrix

- Families: GHZ, HEA (2/4 layers), cycle-QAOA (p=2/4), random brickwork
  (4/8 layers), QFT.
- Widths: 16, 20, 24 by default.
- Contexts: `cpu:complex64`, `cpu:complex128`, `cuda:complex64`,
  `cuda:complex128`.
- Repeats: first post-materialization execution, one warm-up and three warm
  executions by default. CUDA is synchronized around every measurement.

Every raw record carries a circuit hash, logical gate/depth features, device,
precision, cold/warm/end-to-end timings, statevector size, CUDA peak allocation
when relevant, terminal status and error text. Failed or unsupported contexts
remain rows; they are not converted into runtime labels.

## Run

Use a CUDA-enabled PyTorch environment. The checked-in run used Python 3.13,
PyTorch `2.11.0+cu128` and CUDA 12.8. A fresh CUDA environment can be made as
follows; select the PyTorch wheel index appropriate for the review machine if
its CUDA stack differs:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
python3 -m venv "$ROOT/.venv-torch-matrix"
"$ROOT/.venv-torch-matrix/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-torch-matrix/bin/python" -m pip install 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu128
PY="$ROOT/.venv-torch-matrix/bin/python"

"$PY" "$ROOT/experiments/simulator_runtime_v1/run_torch_statevector_matrix.py" \
  --output-dir "$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1"
```

The runner is checkpointed by row. Re-running the same command resumes from
`records.jsonl`; use `--no-resume` only with a new output directory. Start with
a four-context smoke test if the machine state is uncertain:

```bash
"$PY" "$ROOT/experiments/simulator_runtime_v1/run_torch_statevector_matrix.py" \
  --output-dir /tmp/dense_statevector_smoke \
  --families ghz --widths 16 --max-jobs 4
```

Evaluate point-estimation baselines in a separate locked environment:

```bash
python3 -m venv "$ROOT/.venv-runtime-eval"
"$ROOT/.venv-runtime-eval/bin/python" -m pip install \
  -r "$ROOT/experiments/simulator_runtime_v1/requirements-evaluation.txt"
EVAL_PY="$ROOT/.venv-runtime-eval/bin/python"
"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v1/evaluate_torch_statevector_matrix.py" \
  --input "$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1/records.csv" \
  --output-dir "$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation"
```

The evaluator reports random-row, circuit-group, family-held-out,
width-held-out and context-held-out splits. Random-row accuracy is not a
release criterion.

Regenerate the narrative report from raw rows and evaluation outputs:

```bash
"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v1/summarize_torch_statevector_matrix.py" \
  --run-dir "$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1"
```
