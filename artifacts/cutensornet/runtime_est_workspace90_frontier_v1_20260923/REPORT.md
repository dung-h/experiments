# QFT width frontier at a 90% cuTensorNet workspace limit

Finding recorded: 2026-09-23.

This is a local RTX 5070 Ti scalar-contraction experiment. It is not a QPU
runtime result, not a CUDA-Q `sample` wall-clock, and not an extension of the
packaged 25-row / 55-row grids that used a 70% workspace policy.

## Purpose

The question is not whether cuTensorNet can build a contraction plan. The
library already sees the tensor-network graph, the workspace cap, and the GPU
architecture. The question is whether `RUNTIME_EST` after `TIME_TUNED` path
optimization is a wall-time measurement of that plan, or only a cost proxy
that still needs regime-local calibration.

```text
graph + VRAM cap + architecture
≠ exact kernel, layout and memory-traffic time of the selected plan
```

Every timed row uses QFT, complex64, `TIME_TUNED`, 32 optimizer samples,
optimizer seed 137, architecture 12, and `memory_limit="90%"`. Widths 28--42
use two warm-ups and five timed repeats. Width 44 uses one warm-up and three
timed repeats. The GPU is an RTX 5070 Ti with 16.61 GB visible memory and
compute capability 12.0.

NVIDIA documents `TIME_TUNED` as an architecture-specific heuristic over
pairwise contraction times, and `RUNTIME_EST` as an experimental estimate for
one pass over the selected slices. It is not a rehearsal of the exact kernel
sequence that later runs. See the cuTensorNet type documentation for
cuQuantum 24.08.1:
https://docs.nvidia.com/cuda/cuquantum/24.08.1/cutensornet/api/types.html

`EFFECTIVE_FLOPS_EST` is not measured throughput. Under a time objective
NVIDIA defines it as `RUNTIME_EST × ops_peak`. On this run the ratio is
constant at `2.00e13` FLOP/s for every timed width. That is not
`FLOP_COUNT / RUNTIME_EST`, and it is not `FLOP_COUNT / warm actual`.

## What a 90% workspace limit does

`90%` is a feasibility cap of about 14.95 GB. It tells the optimizer not to
select a path whose reported workspace demand exceeds that quota. It does not
mean the plan uses 90% of VRAM, that the largest intermediate is the fastest
choice, or that runtime is determined by VRAM occupancy.

The selected plans themselves show the distinction:

```text
q36: 1 slice, largest intermediate 1.074e9 complex64 (~8 GiB)
q40: 2 slices, largest intermediate 5.369e8 (~4 GiB)
q42: 4 slices, ~4 GiB
q44: 8 slices, ~4 GiB
```

VRAM therefore changes the plan. It does not convert `RUNTIME_EST` into a
benchmark of that plan.

## Timed results

| QFT width | slices | `RUNTIME_EST` | warm actual | actual / estimate | first contraction | FLOPs | largest intermediate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | 1 | 5.617 ms | 5.114 ms | 0.910 | 24.685 ms | 15.27 B | 4.19 M |
| 32 | 1 | 10.612 ms | 20.418 ms | 1.924 | 20.556 ms | 47.49 B | 16.78 M |
| 36 | 1 | 185.939 ms | 778.812 ms | 4.189 | 778.009 ms | 2.73 T | 1.074 B |
| 40 | 2 | 0.726 s | 3.641 s | 5.015 | 3.643 s | 10.82 T | 536.87 M |
| 42 | 4 | 2.229 s | 10.402 s | 4.668 | 10.410 s | 34.37 T | 536.87 M |
| 44 | 8 | 5.796 s | 29.634 s | 5.113 | 29.614 s | 94.35 T | 536.87 M |

Warm actual is `contract_gpu_median_s`. First contraction is
`first_contract_gpu_s`. End-to-end first-call latency remains a separate
column and is not an NVIDIA estimate.

## Small-width context, 70% 25-row grid, not this CSV

The packaged 25-row grid used the same GPU and `TIME_TUNED`, but a 70%
workspace policy and a mixed family set. Its QFT rows over-predict the cheap
widths and land near unity at q28/q30:

| QFT width | `RUNTIME_EST` | warm actual | actual / estimate |
|---:|---:|---:|---:|
| 16 | 1.671 ms | 0.425 ms | 0.254 |
| 20 | 2.495 ms | 0.693 ms | 0.278 |
| 24 | 3.528 ms | 1.129 ms | 0.320 |
| 28 | 5.617 ms | 5.394 ms | 0.960 |
| 30 | 6.385 ms | 6.039 ms | 0.946 |

Those small kernels can finish faster than the architecture table assumes.
q28 is a sweet spot, not proof that the cost model became exact. The 90%
q28 row in this artifact is a second run of a similar unsliced plan: estimate
5.617 ms, warm 5.114 ms (0.910).

The 25-row grid therefore made `RUNTIME_EST` look conservative. The 90%
large-QFT rows reverse the sign of the residual. The estimator is not
uniformly conservative.

## Large-width residual is not slicing alone

q36 is still one slice and already 4.19× slow relative to `RUNTIME_EST`.
Slicing cannot be the only error source.

From q36 through q44 the raw estimate is 4.19--5.11× below warm actual. The
cost model has to approximate many pairwise contractions whose kernel
shapes, intermediate sizes, layouts, slice reductions and synchronization
are not a single table lookup. Supplying architecture 12 tells the
optimizer which heuristic table to use; it does not measure the exact
driver, library, workspace state and plan that then run.

q44 first contraction is 29.614 s and warm contraction is 29.634 s. The
5.11× residual is therefore not Python launch, cold start or first-call
allocation. It is leftover error of the large contraction plan.

## Path-selection variance

The timed table contains one selected plan per width, not a unique QFT plan.
A contraction-free q40 probe, repeated three times with the same nominal
options (90%, `TIME_TUNED`, 32 samples, seed 137), returned:

| probe | `RUNTIME_EST` | FLOPs after slicing | slices | path-search wall time |
|---:|---:|---:|---:|---:|
| 1 | 0.417 s | 5.986 T | 2 | 8.999 s |
| 2 | 0.415 s | 6.277 T | 2 | 6.547 s |
| 3 | 0.687 s | 7.742 T | 4 | 6.149 s |

These rows do not execute a contraction. They establish candidate-plan
variance, not a variance of warm time. The library documents a seed, but
the present high-level multi-threaded hyper-optimizer configuration is not
empirically deterministic. A later run should pin the optimizer to one
thread and control smart/pathfinder settings before attributing every
residual to `RUNTIME_EST`.

The separate `plan_probe/` CSV, one pass over q28--q44 without contraction,
already selected different q40/q42/q44 summaries from the timed plans. Path
search and timed contraction are therefore not interchangeable fixtures.

## Scaling

| Width transition | actual-time factor | factor per added qubit |
|---|---:|---:|
| q28 -> q32 | 3.99× | 1.41× |
| q32 -> q36 | 38.14× | 2.49× |
| q36 -> q40 | 4.68× | 1.47× |
| q40 -> q42 | 2.86× | 1.69× |
| q42 -> q44 | 2.85× | 1.69× |

q32→q36 is a path/intermediate regime change. A single exponential across
q28--q44 is not a valid extrapolator. Inside the sliced regime, q40→q42 at
1.69× per extra qubit predicts q44 as 29.72 s; the measured warm time is
29.63 s (0.3% lower). That is a two-width description, not a deployment
validator.

A train-only median actual/estimate factor from q36, q40 and q42 (4.668)
predicts q44 as 27.06 s versus 29.63 s observed (8.7% low). It is a
regime-local illustration, not a universal correction.

## Interpretation

cuTensorNet has enough information to choose a feasible plan and to emit a
useful cost proxy. It does not promise that `RUNTIME_EST` is wall-time.
For a runtime estimator the usable form is:

```text
RUNTIME_EST
+ workspace policy
+ plan statistics (slices, FLOPs, largest intermediate)
+ fixed optimizer configuration
→ calibrate separately on small / unsliced-large / sliced-large
→ predict warm contraction time
```

Do not treat the 25-row median actual/estimate of 0.196 as a correction for
this 90% QFT frontier. The two workspace policies and the two width regimes
are different measurement tables.

## Reproduction

Timed contraction, same stack as the packaged runner:

```bash
tracks/cutensornet/code/run_cutensornet_runtime_benchmark.sh \
  --families qft --qubits 28,32,36,40 \
  --memory-limit 90% --optimizer-samples 32 --optimizer-seeds 137 \
  --optimizer-cost-functions TIME_TUNED \
  --warmups 2 --repeats 5 \
  --output-dir run-output/cutensornet/qft90_stage_a
```

q42 and q44 were run as separate stages with the same options except q44
`--warmups 1 --repeats 3`. A different GPU, CUDA, cuTensorNet version,
thread count or workspace policy is a new measurement, not a failed copy of
these seconds.

Plan-only q40 repeats, no contraction:

```bash
tracks/cutensornet/code/with_cutensornet_env.sh \
  tracks/cutensornet/code/probe_cutensornet_plan_metrics.py \
  --family qft --qubits 40,40,40 \
  --memory-limit 90% --optimizer-samples 32 --optimizer-seed 137 \
  --output-dir run-output/cutensornet/q40_plan_repeat
```

## Raw files

- `stage_a/cutensornet_runtime_benchmark.csv` — timed q28, q32, q36, q40.
- `stage_b_q42/cutensornet_runtime_benchmark.csv` — timed q42.
- `stage_c_q44/cutensornet_runtime_benchmark.csv` — timed q44.
- `plan_probe/cutensornet_plan_metrics.csv` — one plan-only pass over q28--q44.
- `plan_repeat_q40/cutensornet_plan_metrics.csv` — three q40 plan-only repeats.

The contraction runner is
`tracks/cutensornet/code/run_cutensornet_runtime_benchmark.py`.
The plan probe is `tracks/cutensornet/code/probe_cutensornet_plan_metrics.py`.

## What this does not claim

- It does not replace the 25-row or 55-row 70% fixtures.
- It does not measure full-statevector, sampling, or expectation-value time.
- It does not transfer these seconds onto another GPU.
- It does not make `RUNTIME_EST` a universal simulator runtime estimator.
