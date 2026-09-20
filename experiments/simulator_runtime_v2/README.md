# Structural dense-statevector runtime estimator v2

**Completed on 20 September 2026.** This is the second, deliberately narrow
local-runtime study. It extends the v1 matched matrix with two seeded random
topology families and a GPU feasibility frontier. The committed raw data and
final report are in
[`artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/`](../../artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/).

Read [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) before interpreting
the metrics. In particular, the target is
`prepared_execute_reset_state`: allocate a fresh PyTorch dense statevector,
apply a pre-materialized direct gate schedule, and calculate `<Z_0>`. It does
not include circuit construction, compilation/transpilation, sampling,
framework setup, cloud, or QPU time. It is therefore not pooled with Aer,
CUDA-Q, cuTensorNet, Qonductor, or Ma–Li labels.

## What is committed

- `records.jsonl` and `records.csv`: 112 v2 checkpointed observations.
  Four CUDA OOM observations remain explicit `resource_limit` labels.
- `canonical_corpus.csv`: the compatible 96-row v1 reference plus v2, with a
  `source_run` field. It has 208 rows, 204 successful duration labels and 68
  logical circuits.
- `evaluation_final/`: out-of-fold predictions, per-fold and aggregate
  metrics, memory-envelope tables and report.
- `REPORT.md`: the short report of claims, negative results and limits.

The directories named `evaluation_phase*` and `evaluation_seed_extension` are
local intermediate diagnostics, intentionally not versioned. The committed
`evaluation_final` is regenerated from the complete canonical corpus.

## Environment

The measurement was collected with Python 3.13 and PyTorch `2.11.0+cu128` on
an RTX 5070 Ti. A reviewer needs a CUDA-enabled PyTorch build only to rerun
GPU cells; CPU cells run without CUDA. Use a distinct output directory for a
fresh rerun so the committed fixture remains unchanged.

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
python3 -m venv "$ROOT/.venv-torch-runtime-v2"
"$ROOT/.venv-torch-runtime-v2/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-torch-runtime-v2/bin/python" -m pip install 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu128
PY="$ROOT/.venv-torch-runtime-v2/bin/python"

python3 -m venv "$ROOT/.venv-runtime-v2-eval"
"$ROOT/.venv-runtime-v2-eval/bin/python" -m pip install \
  -r "$ROOT/experiments/simulator_runtime_v2/requirements-evaluation.txt"
EVAL_PY="$ROOT/.venv-runtime-v2-eval/bin/python"
```

## Reproduce the measurement phases

The first command creates the matched structural extension (48 rows): two
families × three widths × two depth settings × four contexts. Rows are
checkpointed in `records.jsonl`; a repeated command resumes safely.

```bash
OUT="$ROOT/work/dense_statevector_v2_rerun"
"$PY" "$ROOT/experiments/simulator_runtime_v2/run_dense_statevector_v2.py" \
  --output-dir "$OUT" \
  --families random_matching,random_star --widths 16,20,24 --seeds 17 \
  --contexts cpu:complex64,cpu:complex128,cuda:complex64,cuda:complex128
```

The seed extension is deliberately GPU-only, because it probes topology
variation under matched GPU contexts rather than pretending it has a four-way
matrix.

```bash
"$PY" "$ROOT/experiments/simulator_runtime_v2/run_dense_statevector_v2.py" \
  --output-dir "$OUT" \
  --families random_matching,random_star --widths 16,20,24 --seeds 43,101 \
  --contexts cuda:complex64,cuda:complex128
```

The frontier command records both successful q26/q28 executions and CUDA OOM
as `resource_limit`. Do not delete those failed rows or replace them with a
censored duration.

```bash
"$PY" "$ROOT/experiments/simulator_runtime_v2/run_dense_statevector_v2.py" \
  --output-dir "$OUT" \
  --families random_matching,random_star --widths 26,28 --seeds 17 \
  --contexts cuda:complex64,cuda:complex128
```

## Build and evaluate the canonical corpus

The v1 file is a compatible reference run under the same target contract.
`build_corpus.py` refuses mismatched semantics and retains a source label.

```bash
"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v2/build_corpus.py" \
  --v1-csv "$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1/records.csv" \
  --v2-csv "$OUT/records.csv" --output "$OUT/canonical_corpus.csv"

"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v2/evaluate_runtime_v2.py" \
  --input "$OUT/canonical_corpus.csv" --output-dir "$OUT/evaluation_final"

"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v2/summarize_runtime_v2.py" \
  --run-dir "$OUT"
```

The primary score is `circuit_group`: every CPU/GPU/precision measurement of a
test logical circuit is held out together. `width_held_out` is a harder
extrapolation test. The seed test is restricted to GPU/q16–q24 by design.
Random-row results are descriptive only.

## Expected final checks

For the committed RTX 5070 Ti fixture, the canonical corpus contains 208 rows
and 204 duration rows. The circuit-held-out `logical_hgb` has log-R² 0.9794;
the more extrapolative width-held-out test favors the analytical Ridge
(0.8754 versus 0.7295). The resource envelope calibrated only through q24
rejects q28 CUDA complex128, where all four observed cells are
`resource_limit`, while accepting q28 CUDA complex64, where all four finish.
Timing values themselves are machine-dependent, so a rerun should compare
protocol, row coverage, statuses and qualitative boundaries rather than demand
bit-identical seconds.
