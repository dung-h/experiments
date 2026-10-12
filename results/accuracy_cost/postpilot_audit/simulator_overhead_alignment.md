> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Dense simulator alignment

| context | circuit_hashes | graph_slower | mlp_slower | median_graph_ratio | median_mlp_ratio |
|---|---|---|---|---|---|
| dense_fp32 | 30 | 27 | 20 | 13.498039752917748 | 2.0307128552730322 |
| dense_fp64 | 30 | 28 | 22 | 16.0967698307204 | 2.446228184639695 |

Each row in simulator_comparisons.csv joins an exact QASM SHA256, original member alias, precision, fold, normalized and legacy IR digest. Execution is CUDA-Q nvidia sample(32) warm host-return time, excluding build and explicit result iteration. Three session medians of five successful repetitions supply each saved target. B1 is the median of 30 repeated three-seed host-return estimator calls for that hash, with inputs resident and CUDA sync. Repetitions are not independent circuits. The targets are not first-call, end-to-end, state retrieval or pure device kernel times.

| item | historical | estimator |
|---|---|---|
| GPU model / driver | NVIDIA GeForce RTX 5070 Ti, 595.91.07, 16303 MiB | NVIDIA GeForce RTX 5070 Ti, GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e, 595.91.07, 1174 MiB, 16303 MiB, 4 %, 45, 2512 MHz, 43.81 W |
| Python / framework | 3.13.15 (main, Aug 14 2026, 15:34:59) [Clang 22.1.3 ] / CUDA-Q Version 0.15.1 (https://github.com/NVIDIA/cuda-quantum aca5853a76d499ecc3d5f97c2e06163ae99d9c75) | 3.10.21 / Torch 2.7.1+cu128 |
| CUDA runtime | 13.0 recorded CuPy runtime API13000; CUDA-Q cu13 | 12.8 |
| Kernel | Linux-7.0.0-34-generic-x86_64-with-glibc2.43 | Linux-7.0.0-38-generic-x86_64-with-glibc2.43 |
| CPU/affinity/load/clocks | not fully recorded for historical sessions | i7-14700F, SMT8/9 one P-core; residual desktop load; not exclusive whole GPU |
| Historical GPU UUID | not recorded | GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e |

Classification B: explicit historical cross-software/time comparison on matched GPU model and driver, rather than A fully matched contemporaneous environment. Historical GPU UUID, CPU affinity, clock/load and complete hardware reservation are unavailable. Sources point to the same project workstation but do not prove all operating conditions match. Estimator uses FP32 arithmetic for both simulation precision targets.

Correction to historical accuracy_cost_analysis.md and the completion message: retain the four numeric counts as descriptive ratios against saved CUDA-Q execution labels. Do not present them as a measured operational speedup/break-even, universal reason to skip prediction, or contemporaneous same-environment comparison. Size support is q2–9, no large-width law. Prediction-only excludes raw-input preparation and setup. These omissions generally add estimator cost but do not quantify total deployment latency.
