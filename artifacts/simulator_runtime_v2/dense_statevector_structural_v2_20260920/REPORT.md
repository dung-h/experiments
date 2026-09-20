# Dense-statevector v2: estimator and feasibility findings

**Finding recorded:** 20 September 2026.

## Claim boundary

This report concerns one local PyTorch dense-statevector reference kernel on the recorded RTX 5070 Ti and CPU. The target is `prepared_execute_reset_state`: allocate a new state, apply a pre-materialized direct gate schedule, then calculate `<Z_0>`. It excludes construction, materialization, transpilation, sampling, cloud service and QPU time. It must not be pooled with Aer, CUDA-Q, cuTensorNet or observed-QPU targets.

The final corpus has **208 raw rows**, **204 successful duration rows**, 68 logical circuits and five widths (16, 20, 24, 26, 28). It retains **4 resource-limit rows** instead of replacing them with duration labels.

## RQ1 — Runtime prediction for an unseen logical circuit

The primary `circuit_group` split holds every context of the logical test circuit outside training. Thus CPU/GPU/precision measurements of that circuit cannot leak into training. Models predict log warm execution time from pre-execution features and known context.

| model                       | n_test | mae_log | rmse_log | r2_log | mae_seconds | r2_seconds |
| --------------------------- | ------ | ------- | -------- | ------ | ----------- | ---------- |
| analytical_calibrated_ridge | 204    | 0.4252  | 0.5411   | 0.9633 | 1.0208      | 0.8591     |
| analytical_hgb              | 204    | 0.3043  | 0.5240   | 0.9656 | 0.7495      | 0.8590     |
| graph_hgb                   | 204    | 0.2280  | 0.4117   | 0.9788 | 0.7740      | 0.8320     |
| graph_hgb_with_family       | 204    | 0.2272  | 0.4140   | 0.9785 | 0.7603      | 0.8355     |
| logical_hgb                 | 204    | 0.2218  | 0.4058   | 0.9794 | 0.6649      | 0.8792     |

The calibrated analytical Ridge uses only final-state bytes, `gate_count × 2^width`, and context. `logical_hgb` adds logical depth and one/two-qubit gate counts. It improves circuit-held-out log-R² from 0.9633 to 0.9794 and lowers log-MAE from 0.4252 to 0.2218. This is evidence that logical structure adds value within this fixed simulator contract.

The graph ablation is a negative result: `graph_hgb` reaches 0.9788 log-R², below `logical_hgb` 0.9794. Adding the coarse family label also does not improve it. The present data therefore do **not** support a claim that interaction-graph features add incremental value beyond logical depth/gate composition.

## RQ2 — Interpolation is not width extrapolation

| model                       | n_test | mae_log | r2_log | mae_seconds | r2_seconds |
| --------------------------- | ------ | ------- | ------ | ----------- | ---------- |
| analytical_calibrated_ridge | 204    | 0.7858  | 0.8754 | 1.5136      | 0.7302     |
| graph_hgb                   | 204    | 1.3712  | 0.7162 | 2.5049      | 0.2914     |
| logical_hgb                 | 204    | 1.3303  | 0.7295 | 2.4731      | 0.2995     |
| structural_ridge            | 204    | 0.8021  | 0.8692 | 1.4940      | 0.7551     |

When an entire width is absent from training, analytical Ridge is more robust: log-R² 0.8754 versus 0.7295 for logical HGB. A nonlinear model that performs best inside the observed width range is not automatically the safe model for a wider circuit. This is why random-row metrics are not used as the main conclusion.

## RQ3 — Independent random-topology seeds

The GPU-only seed holdout leaves out topology seeds 17, 43 or 101 in turn at q16–q24. The two random families retain the same width/depth profile across seeds; this is a controlled test of topology variation, not a broad circuit-distribution benchmark.

| model                       | n_test | mae_log | r2_log | mae_seconds | r2_seconds |
| --------------------------- | ------ | ------- | ------ | ----------- | ---------- |
| analytical_calibrated_ridge | 72     | 0.3146  | 0.9789 | 0.5300      | 0.8391     |
| graph_hgb                   | 72     | 0.0333  | 0.9997 | 0.1086      | 0.9934     |
| logical_hgb                 | 72     | 0.0369  | 0.9997 | 0.0884      | 0.9954     |

Logical HGB reaches 0.9997 log-R² on this matched seed split. The result confirms stability for these generated topologies, but it does not overturn the graph-ablation result: its relevant message is that the pre-execution logical workload/context contract is stable across these seeds.

## RQ4 — GPU feasibility requires a peak-memory envelope

A naive feasibility check uses only `statevector_bytes <= VRAM`. The direct gate kernel allocates temporary tensors, so that is a lower bound rather than a sufficient condition. The frontier envelope calibrates the maximum peak-reserved/statevector ratio through q24, then predicts q26/q28 without using their outcomes.

| context_id      | num_qubits | statevector_bytes | device_total_memory_bytes | observed_rows | observed_ok | observed_resource_limit | observed_error | calibration_largest_width | calibration_peak_ratio | raw_statevector_fits | envelope_peak_bytes | envelope_fits |
| --------------- | ---------- | ----------------- | ------------------------- | ------------- | ----------- | ----------------------- | -------------- | ------------------------- | ---------------------- | -------------------- | ------------------- | ------------- |
| cuda:complex128 | 26         | 1073741824        | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.0859                 | True                 | 4387241984.0000     | True          |
| cuda:complex128 | 28         | 4294967296        | 16608788480.0000          | 4             | 0           | 4                       | 0              | 24                        | 4.0859                 | True                 | 17548967936.0000    | False         |
| cuda:complex64  | 26         | 536870912         | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.1719                 | True                 | 2239758336.0000     | True          |
| cuda:complex64  | 28         | 2147483648        | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.1719                 | True                 | 8959033344.0000     | True          |

At q28, all complex128 rows are `resource_limit` while all complex64 rows complete: {('cuda:complex128', 'resource_limit'): 4, ('cuda:complex64', 'ok'): 4}. The raw complex128 statevector is 4 GiB and therefore passes the naive 15.47 GiB VRAM check, but the q≤24 envelope predicts 17.55 GB peak and correctly rejects it. q28 complex64 is predicted feasible and all four rows complete.

### q28 observed rows

| family          | depth_parameter | context_id      | status         | warm_execution_median_seconds | statevector_bytes | torch_cuda_peak_reserved_bytes |
| --------------- | --------------- | --------------- | -------------- | ----------------------------- | ----------------- | ------------------------------ |
| random_matching | 4               | cuda:complex128 | resource_limit | nan                           | 4294967296        | nan                            |
| random_matching | 4               | cuda:complex64  | ok             | 6.9289                        | 2147483648        | 8613003264.0000                |
| random_matching | 8               | cuda:complex128 | resource_limit | nan                           | 4294967296        | nan                            |
| random_matching | 8               | cuda:complex64  | ok             | 13.8400                       | 2147483648        | 8613003264.0000                |
| random_star     | 4               | cuda:complex128 | resource_limit | nan                           | 4294967296        | nan                            |
| random_star     | 4               | cuda:complex64  | ok             | 6.2748                        | 2147483648        | 8613003264.0000                |
| random_star     | 8               | cuda:complex128 | resource_limit | nan                           | 4294967296        | nan                            |
| random_star     | 8               | cuda:complex64  | ok             | 12.5417                       | 2147483648        | 8613003264.0000                |

This is the strongest engineering result in the study: before attempting execution, a context-calibrated peak envelope separates the feasible q28 complex64 context from the infeasible q28 complex128 context on this kernel/GPU.

## Context-selection result

| oracle_context | circuit_groups |
| -------------- | -------------- |
| cuda:complex64 | 36             |

The full four-context groups all have `cuda:complex64` as observed fastest. Consequently the 100% context-selection score in the evaluator is trivial and is **not** presented as an estimator contribution. It is a machine/kernel property for this benchmark range.

## What can and cannot be claimed

- Supported: a source-aware local estimator can predict warm runtime accurately inside this fixed dense-statevector contract; logical gate/depth features improve the analytical workload baseline on unseen circuits in-range; an empirically calibrated peak-memory envelope predicts the q28 precision feasibility boundary.
- Not supported: universal simulator runtime prediction, cross-GPU transfer, Aer/CUDA-Q/cuTensorNet equivalence, QPU-runtime prediction, sampling/noise/transpilation timing, or incremental graph-feature benefit in this corpus.
- Next test: add a second physical host/GPU and an independently implemented simulator as separately labelled domains. A domain-transfer result must be evaluated without pooling their runtimes.

## Reproduce

Run `experiments/simulator_runtime_v2/run_dense_statevector_v2.py` according to `experiments/simulator_runtime_v2/README.md`, then `build_corpus.py`, `evaluate_runtime_v2.py`, and this script. Raw measurements are `records.jsonl`/`records.csv`; all OOF predictions and metrics are under `evaluation_final/`.
