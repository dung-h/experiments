# CUDA-Q matched runtime pilot

**Run date:** 21 September 2026  
**CUDA-Q:** 0.15.1, revision `aca5853a76d499ecc3d5f97c2e06163ae99d9c75`  
**GPU:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB  
**Status:** local simulator pilot; not QPU or cloud-runtime data

## Question

Can a pre-run feature table support a runtime estimator when the simulator
target and timing contract are explicitly fixed? The first call and the warm
call are intentionally separate because CUDA-Q may compile/lazily initialise a
backend on first use.

## Protocol

The generator is
[`run_cudaq_matrix.py`](../../../experiments/cudaq_runtime/run_cudaq_matrix.py).
It creates deterministic kernels and records width, logical depth, total
gates, two-qubit gates, analytical statevector bytes, simulator target and
precision before calling `cudaq.sample`.

| Target | Widths | Precision | Rows |
| --- | --- | --- | ---: |
| `qpp-cpu` | 16, 20, 22 | FP64 | 12 |
| `nvidia` | 16, 20, 24, 28 | FP32 | 16 |
| `nvidia` | 16, 20, 24, 28 | FP64 | 16 |

Each row uses 32 shots, one warm-up call and three measured warm calls.
`first_sample_s` is the wall-clock around the first `cudaq.sample` call.
`warm_sample_median_s` is the median of later calls, including simulation and
result extraction but excluding queue and QPU time.

Raw data and environment manifest:

- [`cudaq_matrix.csv`](cudaq_matrix.csv)
- [`cudaq_matrix.environment.json`](cudaq_matrix.environment.json)
- [`cpu_matrix.csv`](cpu_matrix.csv) and [`gpu_matrix.csv`](gpu_matrix.csv)

## Observed timings

| Target | Rows | Median first call (s) | Median warm call (s) | Maximum warm call (s) |
| --- | ---: | ---: | ---: | ---: |
| CPU QPP FP64 | 12 | 2.2551 | 2.3448 | 28.8395 (HEA-22) |
| GPU FP32 | 16 | 0.0327 | 0.0026 | 0.1346 (HEA-28) |
| GPU FP64 | 16 | 0.0186 | 0.0182 | 1.3828 (HEA-28) |

The full table, including per-family values and first/warm repeats, is the
CSV rather than a single pooled runtime column. The largest analytical
statevector sizes were 64 MiB for CPU FP64, 2 GiB for GPU FP32 and 4 GiB for
GPU FP64. These are feasibility descriptors, not measured peak allocations.

The warm-family medians illustrate why width alone is insufficient:

| Family | CPU FP64 (s) | GPU FP32 (s) | GPU FP64 (s) |
| --- | ---: | ---: | ---: |
| GHZ | 0.5467 | 0.0022 | 0.0168 |
| HEA | 6.3727 | 0.0046 | 0.0405 |
| QAOA cycle | 2.2761 | 0.0022 | 0.0195 |
| Random brickwork | 3.2481 | 0.0034 | 0.0316 |

The GPU FP32 first-call ratio has a long tail because backend/kernel
initialisation appears in a few cells; the median first/warm ratio is 1.09 but
the maximum is 138×. This is evidence for keeping first-call latency as a
separate target, not evidence that the ratio is a universal constant.

## Baseline estimator evaluation

[`evaluate_cudaq_matrix.py`](../../../experiments/cudaq_runtime/evaluate_cudaq_matrix.py)
fits log-runtime median, Ridge and HistGradientBoosting separately for each
target. It evaluates random 5-fold, family-held-out and width-held-out splits.
The result files are:

- [`summary_warm_sample_median_s.csv`](evaluation/summary_warm_sample_median_s.csv)
- [`summary_first_sample_s.csv`](evaluation/summary_first_sample_s.csv)
- the corresponding `metrics_*.csv` files and JSON manifests

Representative warm-call Ridge results (log-R²) are:

| Target | Random 5-fold | Family held-out | Width held-out |
| --- | ---: | ---: | ---: |
| CPU FP64 | −1.966 | 0.370 | −0.385 |
| GPU FP32 | −997.538 | 0.835 | −798.815 |
| GPU FP64 | −2,995.557 | 0.866 | −34,523.478 |

These extremely negative random/width R² values are not typos. The pilot has
only one deterministic circuit instance per family/width and very small
absolute GPU timings; a few outliers dominate R². The comparatively better
family-held-out scores should also not be overinterpreted: the feature set and
corpus are still too small for a deployment claim. The useful finding is
diagnostic: a model can interpolate this controlled grid, yet extrapolation to
an unseen width or timing regime is not established.

## Interpretation and limits

1. The experiment is a genuine pre-run local-simulator target, but the target
   is explicitly `CUDA-Q sample` wall-clock, not a QPU/cloud runtime.
2. First-call and warm-call behaviour differ enough that they cannot be pooled
   as one label.
3. Precision and target materially change scale; CPU QPP FP64, GPU FP32 and
   GPU FP64 are separate domains.
4. Circuit family changes the timing at the same width, especially on CPU and
   at the larger GPU widths. Width, gate count and statevector bytes alone are
   not yet a sufficient universal estimator.
5. The pilot does not include MPS, tensor-network contraction, sampling
   scaling, independent random seeds or a second GPU. Those are follow-up
   domains, not missing rows to silently append here.

