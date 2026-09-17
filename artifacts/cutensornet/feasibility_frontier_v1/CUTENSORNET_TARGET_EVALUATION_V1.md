# cuTensorNet target-semantics evaluation v1

This is a single-environment evaluation on the RTX 5070 Ti. cuTensorNet
returns one native `RUNTIME_EST` per optimized network; it does not
return separate estimates for warm, first-contraction, and end-to-end
latency. The same estimate is evaluated against each measured target
below. This is not a GPU-held-out or universal-runtime claim.

Input: `experiments/simulator_benchmark/outputs/feasibility_frontier_v1/cutensornet_feasibility_frontier.csv` (55 rows).

## What is compared

- `cutensornet_runtime_est_s`: the single estimate returned after path optimization, before contraction.
- `contract_gpu_median_s`: warm CUDA-event contraction; build/path/allocation are excluded.
- `first_contract_gpu_s`: first contraction after path setup.
- `end_to_end_first_s`: tensor-network build + path optimization + first contraction.

## Global metrics: one estimate, three measured targets

Each row reuses the same `cutensornet_runtime_est_s` prediction and changes only the measured target used for evaluation.

| Measured target (same estimate reused) | n | MAE | RMSE | R2 | Bias (estimate−actual) | Median actual/estimate |
|---|---:|---:|---:|---:|---:|---:|
| contract_gpu_median_s | 55 | 2.645 ms | 3.377 ms | -8.469 | 2.645 ms | 0.215 |
| first_contract_gpu_s | 55 | 5.054 ms | 8.595 ms | 0.051 | -4.239 ms | 1.242 |
| end_to_end_first_s | 55 | 395.913 ms | 422.114 ms | -7.101 | -395.913 ms | 137.972 |

## Warm-contract family ratios

| Family | Rows | Median actual/estimate | Median absolute log error |
|---|---:|---:|---:|
| ghz | 5 | 0.236 | 1.445 |
| hea | 15 | 0.208 | 1.568 |
| qaoa_cycle | 15 | 0.220 | 1.513 |
| qft | 5 | 0.320 | 1.140 |
| random_brickwork | 15 | 0.206 | 1.581 |

## Interpretation

The warm row asks whether the single native estimate is useful for the GPU
contraction segment. The first-contraction and end-to-end rows answer different
questions about what a user experiences on the first call. It is therefore
incorrect to interpret these rows as three cuTensorNet predictions or
to use the warm-target error as the error of the complete simulator invocation.

The native estimator is a useful B0 path/feasibility signal, but its
heuristic throughput model is not a live benchmark. A production
user-facing estimator should either (a) report warm contraction and
planning latency separately, or (b) fit a local calibration for a
declared end-to-end target. The latter must be retrained when the GPU,
CUDA/cuTensorNet version, precision or memory policy changes.
