# Azizov independent reproduction — P0 smoke report

This is a bounded smoke run, not the paper's full 1,402-circuit result.
The target is one timed `AerSimulator.run(...).result()` after one
untimed backend warm-up; transpilation is measured separately.

## Coverage

- Rows: **168** (168 successful, 0 errors).
- Backends: FakeSherbrooke, FakeWashingtonV2.
- Families represented: **21**.
- Optimization levels: 0, 1, 2, 3.
- Shots: 1024.
- The 22nd source family (`twolocalrandom`) has no unique circuit with eight or fewer logical qubits after source-hash deduplication, so it is not present in this P0 selection.

## Runtime summary (seconds)

| Backend | Opt | n | median T_transpile | median T_exec | p95 T_exec |
|---|---:|---:|---:|---:|---:|
| FakeSherbrooke | 0 | 21 | 0.0078 | 0.5545 | 3.4747 |
| FakeSherbrooke | 1 | 21 | 0.0121 | 0.4910 | 1.6667 |
| FakeSherbrooke | 2 | 21 | 0.0150 | 0.5200 | 1.5643 |
| FakeSherbrooke | 3 | 21 | 0.0175 | 0.4409 | 3.3320 |
| FakeWashingtonV2 | 0 | 21 | 0.0057 | 0.6521 | 1.8012 |
| FakeWashingtonV2 | 1 | 21 | 0.0089 | 0.6452 | 1.2304 |
| FakeWashingtonV2 | 2 | 21 | 0.0129 | 0.6431 | 2.9512 |
| FakeWashingtonV2 | 3 | 21 | 0.0135 | 0.6327 | 2.9237 |

## Interpretation

The P0 run verifies that the public Ma–Li QASM pool can be loaded, transpiled against both current fake-provider snapshots, and executed with Aer noise at 1,024 shots. It also exposes a non-trivial family and backend spread in `T_exec`. These numbers are local measurements under Qiskit 2.5.2/Aer 0.17.2 and must not be compared as exact values to the paper's HPC measurements.

Next: run the full P1 matrix with explicit timeout/censoring records, then fit grouped baselines and compare source versus transpiled feature blocks.

## Source versus compiled feature smoke ablation

The same circuit-ID grouped split is reused for every block.
These results are P0 diagnostics, not paper-scale metrics.

| Block | Model | R² log | R² seconds | RMSE seconds | MAE seconds |
|---|---|---:|---:|---:|---:|
| source | ridge | 0.7764 | 0.6863 | 0.5792 | 0.2776 |
| source | random_forest | 0.7488 | 0.6130 | 0.6433 | 0.3129 |
| source | hist_gradient_boosting | 0.6642 | 0.4787 | 0.7467 | 0.3347 |
| compiled | ridge | 0.8249 | 0.7889 | 0.4751 | 0.2768 |
| compiled | random_forest | 0.7376 | 0.5964 | 0.6570 | 0.3356 |
| compiled | hist_gradient_boosting | 0.7605 | 0.6461 | 0.6152 | 0.2970 |
| hybrid | ridge | 0.7578 | 0.6434 | 0.6176 | 0.3087 |
| hybrid | random_forest | 0.7381 | 0.5979 | 0.6558 | 0.3326 |
| hybrid | hist_gradient_boosting | 0.7788 | 0.7007 | 0.5658 | 0.3108 |

On this small subset, compiled global features improve Ridge over source-only features, while the hybrid block does not improve every model. This is the intended hypothesis check; the full P1/P2 runs must determine whether the pattern is stable.
