# Simulator-estimator validation pass

**Date:** 20 September 2026
**Machine:** one NVIDIA GeForce RTX 5070 Ti; all CPU/GPU configuration fields
are retained in the corresponding raw artifacts.

## Scope and target boundary

This validation pass expands six simulator-related tracks.  It does **not**
create a pooled `runtime_seconds` target.  Each target below has different
semantics, hardware and inclusion rules.

| Track | Rows / units | Target | Main split or decision check |
|---|---:|---|---|
| VQCSim | 90 GPU rows; base plus matched mirror circuit | prepared-circuit float32 inference CUDA-event wall time | circuit-group, family- and width-held-out |
| Noisy Aer | 1,192 existing local rows | Aer execution after fake-backend transpilation; transpilation excluded | grouped-circuit, family and backend held out; train-fold conformal interval |
| Zero-Setup PPS | 36 new CPU rows | `propagate_through_rotation_gates` only | configuration-group, delta- and Trotter-held-out |
| cuTensorNet plan choice | 98 feasible candidate contractions | warm scalar contraction CUDA-event time | minimum-FLOP decision against observed per-network oracle |
| EMU-MPS | 24 threshold labels | smallest `max_bond_dim` rung reaching final-state fidelity 0.99 | leave-one-size-out with known families |
| cuTensorNet precision | 52 matched pairs | warm scalar contraction time plus a complex64/complex128 scalar-agreement label | out-of-fold precision policy |

Path search, first call, preparation and execution are kept as separate
columns whenever a runner reports them.  The full target contracts are in the
track reports linked below.

The companion [`DEEP_ANALYSIS_2026-09-20.md`](DEEP_ANALYSIS_2026-09-20.md)
derives residual, calibration-shift, feasibility and numerical-regime
diagnostics from the raw and OOF artifacts.  It distinguishes observed facts
from evidence-consistent mechanisms.

## Results

### VQCSim: within-contract runtime estimator

The fixed environment has five variational families, widths 16--24, batch
size one and float32.  Every base and mirror row completed and passed the
source runner's Qiskit parity check (90/90).  The mirror transformation is a
stability check: its median timing ratio to the base circuit is 1.014 and its
mean absolute timing change is 1.72%.

| Split | Best model | Log MAE | Log R2 | MAE ms | R2 ms |
|---|---|---:|---:|---:|---:|
| Circuit group (primary) | HistGradientBoosting | 0.131 | 0.971 | 56.413 | 0.875 |
| Family held out | HistGradientBoosting | 0.315 | 0.895 | 60.536 | 0.895 |
| Width held out | HistGradientBoosting | 0.289 | 0.876 | 100.512 | 0.644 |

The result supports a small **VQCSim-only** estimator under this exact GPU,
precision, batch-size and prepared-inference contract.  It does not test CPU,
sampling, preparation time, a second GPU or a different simulator.

Raw data and evaluation:

- `vqcsim_rq2_gpu_fp32_16_24_20260920/records.csv` (base)
- `vqcsim_rq2_gpu_fp32_mirror_16_24_20260920/records.csv` (mirror)
- `vqcsim_estimator_base_mirror_20260920/evaluation.md`

### Noisy Aer: point estimates do not transfer their intervals to a backend shift

The target is a local noisy-Aer execution after fake-backend transpilation,
not QPU execution or cloud turnaround.  A HistGradientBoosting point model and
90% train-fold conformal residual band were fitted inside every outer fold.

| Split | Log MAE | Log R2 | MAE s | R2 s | 90% interval coverage | Mean interval width s |
|---|---:|---:|---:|---:|---:|---:|
| Grouped logical circuit | 0.068 | 0.559 | 0.167 | 0.239 | 90.5% | 0.524 |
| Family held out | 0.067 | 0.553 | 0.154 | 0.348 | 87.5% | 0.360 |
| Fake backend held out | 0.133 | 0.356 | 0.257 | 0.405 | 21.7% | 0.247 |

The critical result is the last row.  A nominally calibrated within-domain
band is severely under-covering after backend shift.  Therefore neither the
point estimator nor its current interval must be marketed as backend-agnostic.
Resource-censored rows remain a feasibility boundary, never runtime labels.

Raw data and evaluation:

- `../azizov_independent/feasible_q9_runtime.csv`
- `aer_interval_validation_20260920/evaluation.md`

### Zero-Setup PPS: approximation control is a required pre-run feature

The new controlled grid has 4 Trotter step counts, 3 declared approximation
thresholds (`delta`) and 3 trials per configuration.  Post-run Pauli counts,
truncated norm and expectation are excluded from estimator inputs.  The target
is the local CPU propagation kernel only.

| Split | Best model | Log MAE | Log R2 | MAE s | R2 s |
|---|---|---:|---:|---:|---:|
| Configuration group | Ridge | 0.935 | 0.521 | 0.114 | 0.606 |
| Delta held out | Ridge | 1.033 | 0.457 | 0.160 | -0.041 |
| Trotter held out | HistGradientBoosting | 0.812 | 0.442 | 0.102 | 0.674 |

The tree model's delta-held-out log-R2 is only 0.075 (and seconds R2 is
-0.232), despite its stronger configuration-group result.  This is evidence
against extrapolating a learned approximation-control relationship from this
small grid.

Raw data and evaluation:

- `zero_setup_pps_grid_20260920.jsonl`
- `zero_setup_pps_estimator_20260920/evaluation.md`

### cuTensorNet plan choice: feasibility is part of the candidate definition

Each of 15 small circuit-derived tensor networks was contracted using a native
`TIME_TUNED` plan, two deterministic paths and four connectivity-preserving
random paths.  There are 98 feasible candidate rows.  Seven further explicit
paths were rejected before timing: six require more than the declared 50% GPU
memory limit and one is unsupported by cuTensorNet.  They are preserved in
`candidates_connected.failures.json`; they are not silently converted into
runtime observations.

| Quantity | Result |
|---|---:|
| Fixed-network candidate sets | 15 |
| Feasible timed candidate rows | 98 |
| Minimum-FLOP exact-fastest selection | 80.0% |
| Minimum-FLOP median / worst regret | 1.000x / 1.164x |
| Native `TIME_TUNED` median regret | 1.000x |

This is a useful analytical baseline over an explicit feasible candidate set.
It is not a replication of a learned-ranker dataset, candidate generator,
model, or cross-GPU study.

Raw data and evaluation:

- `explicit_plan_ranking_expanded_20260920/candidates_connected.csv`
- `explicit_plan_ranking_expanded_20260920/candidates_connected.failures.json`
- `explicit_plan_ranking_expanded_20260920/evaluation_connected.md`

### EMU-MPS: threshold prediction remains distinct from runtime prediction

Twelve predeclared pulse cases were added to the original 12-row local pilot:
four pulse families at 8, 12 and 16 qubits.  A high-chi local reference was
constructed per case, then the smallest rung in `{1, 2, 4, ..., 256}` reaching
final-state fidelity 0.99 was labelled.  All 24 references had observed
maximum chi below 256, so no label is censored.

| Method | Exact rung | Within one rung | Mean absolute rung error |
|---|---:|---:|---:|
| Shared numeric model | 37.5% | 100.0% | 0.625 |
| Inferred-family residual model | 37.5% | 100.0% | 0.625 |

Family inference is 100% in these folds because the constructed families have
distinct pulse-metadata signatures.  The residual correction does not improve
the shared numeric baseline.  This is a negative result for the tested
family-aware mechanism, not evidence that family-aware modelling is useless
on an independent QASM corpus.

Raw data and evaluation:

- `emu_mps_threshold_extension_20260920/threshold_labels.csv`
- `emu_mps_threshold_extension_20260920/threshold_sweep.jsonl`
- `family_aware_emu_mps_expanded_20260920/evaluation.md`

### cuTensorNet precision: speed trade-off exists; safe automatic choice is not established

The 52 pairs contract the same scalar network once in `complex64` and once in
`complex128`.  The latter scalar is an in-run numerical comparison point, not
an exact statevector-fidelity oracle.

| Quantity | Result |
|---|---:|
| Matched pairs | 52 |
| Median complex128 / complex64 time | 3.405x |
| Ratio range | 2.465--4.685x |
| Maximum relative scalar difference | 5.471e-06 |
| `complex64` unsafe at relative tolerance `5e-7` | 53.8% |

An out-of-fold random-split policy has AUC 0.869, but family-held-out AUC falls
to 0.539 and 66.7% of its few complex64 selections are unsafe.  The policy is
therefore not deployable as an automatic precision selector.

Raw data and evaluation:

- `precision_selection_expanded_20260920/records.csv`
- `precision_selection_expanded_20260920/evaluation.md`
- `precision_selection_expanded_20260920/policy.md`

## Reproduction sequence

The pinned public-source information and isolated environment instructions are
in `experiments/simulator_papers/README.md`.  From this repository root, the
evaluation-only commands do not contact a cloud service:

```bash
PY=/path/to/python-with-pandas-and-scikit-learn

$PY experiments/simulator_papers/evaluate_vqcsim_runtime_estimator.py \
  --input base=artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/records.csv \
  --input mirror=artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_mirror_16_24_20260920/records.csv \
  --output-prefix /tmp/vqcsim/evaluation

$PY experiments/simulator_papers/evaluate_aer_runtime_intervals.py \
  artifacts/azizov_independent/feasible_q9_runtime.csv --output-prefix /tmp/aer/evaluation

$PY experiments/simulator_papers/evaluate_zero_setup_pps_estimator.py \
  artifacts/simulator_papers/zero_setup_pps_grid_20260920.jsonl --output-prefix /tmp/pps/evaluation

$PY experiments/simulator_papers/evaluate_family_aware_threshold_proxy.py \
  artifacts/pasqal_emu_mps/threshold_sweep_labels.csv \
  artifacts/simulator_papers/emu_mps_threshold_extension_20260920/threshold_labels.csv \
  --output-prefix /tmp/emu/evaluation

python3 experiments/simulator_papers/evaluate_explicit_plan_ranking_proxy.py \
  artifacts/simulator_papers/explicit_plan_ranking_expanded_20260920/candidates_connected.csv \
  --output-prefix /tmp/plan/evaluation
```

For a new dated GPU run, set `CUTENSORNET_PYTHON` to a CUDA-enabled Python
environment containing compatible `cuquantum` and `cupy`, then run the
following. The explicit paths use a 50% VRAM limit by design; candidate
failures are part of the output rather than a reason to retry at a higher
limit.

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
export CUTENSORNET_PYTHON=/path/to/cuda-python

bash "$ROOT/tracks/cutensornet/code/with_cutensornet_env.sh" \
  "$ROOT/experiments/simulator_papers/run_explicit_plan_ranking_proxy.py" \
  --output "$ROOT/artifacts/simulator_papers/new_plan/candidates_connected.csv" \
  --qubits 8,10,12 --random-path-count 4 --memory-limit 50%

bash "$ROOT/tracks/cutensornet/code/with_cutensornet_env.sh" \
  "$ROOT/experiments/simulator_papers/run_precision_selection_proxy.py" \
  --output "$ROOT/artifacts/simulator_papers/new_precision/records.csv" \
  --qubits 8,12,16,20 --depth-multipliers 1,2 --seed 137
```

For the PPS extension, first run `scripts/bootstrap_upstreams.sh`, then use its
pinned `zero_setup` checkout:

```bash
"$ROOT/work/venvs/vqcsim/bin/python" \
  "$ROOT/experiments/simulator_papers/run_zero_setup_pps_grid.py" \
  --upstream-root "$ROOT/work/zero_setup" \
  --output "$ROOT/artifacts/simulator_papers/new_pps.jsonl" \
  --trotter-steps 4,8,12,20 --deltas 1e-2,5e-3,1e-3 --trials 3
```

For EMU-MPS, create a CUDA-enabled environment containing `torch`,
`emu-mps==2.9.1`, `pulser==1.9.1` and its dependencies.  The runner itself
contains the predeclared schedules, layouts, rungs and resume protocol:

```bash
EMU_PY=/path/to/emu-mps-cuda-python
"$EMU_PY" "$ROOT/tracks/pasqal_emu_mps/scripts/run_threshold_extension.py" \
  --output-prefix "$ROOT/artifacts/simulator_papers/new_emu/threshold" \
  --target-fidelity 0.99 --reference-max-bond-dim 256
```

Regeneration can produce different wall-clock values. It must be recorded as a
new dated artifact rather than overwriting this validation pass.

## Conclusion

The evidence supports source- and contract-specific estimators.  It rejects
three tempting but invalid shortcuts: pooling simulator targets, relying on
random splits alone, and treating pre-run feasibility/precision/threshold
proxies as measured execution time.  The strongest next engineering task is a
versioned simulator benchmark harness with more circuit seeds and independent
hardware configurations within each contract, followed by transfer evaluation
only after the source-specific baselines are stable.
