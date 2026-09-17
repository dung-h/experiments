# cuTensorNet target-semantics evaluation v1

This is a single-environment evaluation on the RTX 5070 Ti. cuTensorNet
returns one native `RUNTIME_EST` per optimized network; it does not
return separate estimates for warm, first-contraction, and end-to-end
latency. The same estimate is evaluated against each measured target
below. This is not a GPU-held-out or universal-runtime claim.

Input: `/home/server/Documents/Capstone-Project/experiments/simulator_benchmark/outputs/current_grid/cutensornet_runtime_benchmark.csv` (25 rows).

## What is compared

- `cutensornet_runtime_est_s`: the single estimate returned after path optimization, before contraction.
- `contract_gpu_median_s`: warm CUDA-event contraction; build/path/allocation are excluded.
- `first_contract_gpu_s`: first contraction after path setup.
- `end_to_end_first_s`: tensor-network build + path optimization + first contraction.

## Global metrics: one estimate, three measured targets

Each row reuses the same `cutensornet_runtime_est_s` prediction and changes only the measured target used for evaluation.

| Measured target (same estimate reused) | n | MAE | RMSE | R2 | Bias (estimate−actual) | Median actual/estimate |
|---|---:|---:|---:|---:|---:|---:|
| contract_gpu_median_s | 25 | 1.973 ms | 2.401 ms | -1.774 | 1.973 ms | 0.196 |
| first_contract_gpu_s | 25 | 2.322 ms | 3.456 ms | -0.215 | 0.843 ms | 0.314 |
| end_to_end_first_s | 25 | 123.110 ms | 182.831 ms | -0.805 | -123.110 ms | 22.635 |

## Warm-contract family ratios

| Family | Rows | Median actual/estimate | Median absolute log error |
|---|---:|---:|---:|
| ghz | 5 | 0.209 | 1.567 |
| hea | 5 | 0.189 | 1.666 |
| qaoa_cycle | 5 | 0.192 | 1.651 |
| qft | 5 | 0.320 | 1.139 |
| random_brickwork | 5 | 0.193 | 1.646 |

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
