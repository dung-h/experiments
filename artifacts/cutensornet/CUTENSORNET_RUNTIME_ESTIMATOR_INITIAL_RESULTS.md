# cuTensorNet pre-run runtime estimate: initial GPU benchmark

## Scope

This simulator-only result compares cuTensorNet's experimental pre-run
`RUNTIME_EST` against CUDA-event timing of the **same scalar tensor-network
contraction** on the local RTX 5070 Ti. It is not QPU execution time, cloud
turnaround, full-statevector materialisation, or Python end-to-end latency.

- Rows: **25** circuit-derived tensor networks.
- Families: GHZ, hardware-efficient ansatz (HEA), QAOA MaxCut cycle p=3,
  random brickwork, and QFT.
- Widths: 16, 20, 24, 28, and 30 qubits.
- Precision: complex64; warm-up: 2 contractions; measured repeats: 5.
- Optimizer: `CUTENSORNET_OPTIMIZER_COST_TIME_TUNED`; no slices were required
  in this initial grid.

## Results

| family | qubits | depth | 2Q_gates | RUNTIME_EST_ms | actual_contract_ms | actual_over_estimate | FLOPs | slices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ghz | 16 | 16 | 15 | 0.470 | 0.114 | 0.243 | 852 | 1 |
| hea | 16 | 12 | 30 | 1.890 | 0.357 | 0.189 | 6.71e+03 | 1 |
| qaoa_cycle | 16 | 52 | 48 | 1.430 | 0.278 | 0.194 | 2.1e+04 | 1 |
| qft | 16 | 31 | 120 | 1.671 | 0.425 | 0.254 | 1.77e+06 | 1 |
| random_brickwork | 16 | 24 | 60 | 3.470 | 0.666 | 0.192 | 4.78e+04 | 1 |
| ghz | 20 | 20 | 19 | 0.590 | 0.138 | 0.235 | 1.14e+03 | 1 |
| hea | 20 | 12 | 38 | 2.370 | 0.444 | 0.187 | 8.66e+03 | 1 |
| qaoa_cycle | 20 | 64 | 60 | 1.790 | 0.341 | 0.190 | 2.72e+04 | 1 |
| qft | 20 | 39 | 190 | 2.495 | 0.693 | 0.278 | 5.68e+07 | 1 |
| random_brickwork | 20 | 24 | 76 | 4.350 | 0.833 | 0.192 | 6.35e+04 | 1 |
| ghz | 24 | 24 | 23 | 0.710 | 0.148 | 0.209 | 1.42e+03 | 1 |
| hea | 24 | 12 | 46 | 2.850 | 0.563 | 0.198 | 1.05e+04 | 1 |
| qaoa_cycle | 24 | 76 | 72 | 2.150 | 0.412 | 0.192 | 3.34e+04 | 1 |
| qft | 24 | 47 | 276 | 3.528 | 1.129 | 0.320 | 7.65e+08 | 1 |
| random_brickwork | 24 | 24 | 92 | 5.230 | 1.008 | 0.193 | 7.6e+04 | 1 |
| ghz | 28 | 28 | 27 | 0.830 | 0.170 | 0.204 | 1.66e+03 | 1 |
| hea | 28 | 12 | 54 | 3.330 | 0.623 | 0.187 | 1.23e+04 | 1 |
| qaoa_cycle | 28 | 88 | 84 | 2.510 | 0.485 | 0.193 | 3.88e+04 | 1 |
| qft | 28 | 55 | 378 | 5.617 | 5.394 | 0.960 | 1.53e+10 | 1 |
| random_brickwork | 28 | 24 | 108 | 6.110 | 1.197 | 0.196 | 9.3e+04 | 1 |
| ghz | 30 | 30 | 29 | 0.890 | 0.175 | 0.196 | 1.77e+03 | 1 |
| hea | 30 | 12 | 58 | 3.570 | 0.695 | 0.195 | 1.32e+04 | 1 |
| qaoa_cycle | 30 | 94 | 90 | 2.690 | 0.516 | 0.192 | 4.12e+04 | 1 |
| qft | 30 | 59 | 435 | 6.385 | 6.039 | 0.946 | 1.13e+10 | 1 |
| random_brickwork | 30 | 24 | 116 | 6.550 | 1.309 | 0.200 | 9.92e+04 | 1 |

`actual_over_estimate` is `actual_contract_gpu_s / RUNTIME_EST_s`. A value of
1 is perfect; below 1 means the native estimate over-predicted this warm
contraction time.

- Median actual/estimate ratio: **0.196**.
- Geometric-mean actual/estimate ratio: **0.235**.
- Mean absolute log error of the native estimate: **1.450**.
- No row had actual contraction time above the native estimate in this grid.

## Family calibration pattern

| family | median_actual_over_estimate | geometric_mean_actual_over_estimate |
| --- | --- | --- |
| ghz | 0.209 | 0.217 |
| hea | 0.189 | 0.191 |
| qaoa_cycle | 0.192 | 0.192 |
| qft | 0.320 | 0.460 |
| random_brickwork | 0.193 | 0.195 |

## Interpretation

The estimator is genuinely pre-run and tracks the difficult QFT 28/30-qubit
cases much more closely than the low-cost structured circuits. However, it
systematically over-predicts the warm scalar-contraction timings for GHZ, HEA,
QAOA-cycle, and random-brickwork in this small benchmark. It is therefore a
strong **B0** baseline, not ground truth.

The next B1 experiment should fit a calibration model using `RUNTIME_EST`,
FLOP count, largest intermediate tensor, slices, circuit family, width, and
precision; evaluate it with family-held-out and qubit-range-held-out splits.

## Limits retained deliberately

- The scalar contraction target is the correct direct comparison for the
  cuTensorNet path estimator, but is not full simulator end-to-end latency.
- This is one fixed circuit instance per family/width and uses only complex64.
- Path-search time, tensor construction, allocation, compilation, sampling,
  and host/result-transfer are separate timing targets and must not be folded
  into `contract_gpu_median_s`.
- `RUNTIME_EST` is marked experimental by NVIDIA and must be rechecked whenever
  the cuTensorNet version, CUDA runtime, GPU, precision, memory limit, or
  optimizer configuration changes.
