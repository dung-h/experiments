# Ma–Li / QCRE transpiler-seed sensitivity

The same 340 QASM rows were transpiled with a fixed Qiskit/FakeBackend
configuration and three `seed_transpiler` values. All scores use the
same logical-QASM-hash grouped five-fold split (split seed 1234).

Seeds: **1234, 2025, 31415**; optimization level: **1**.

## Grouped out-of-sample R² by transpiler seed

| Seed | Physical depth | Physical 2Q depth | Weighted path |
|---:|---:|---:|---:|
| 1234 | 0.7879 | 0.7731 | 0.6692 |
| 2025 | 0.7897 | 0.7769 | 0.6756 |
| 31415 | 0.7764 | 0.7661 | 0.6437 |

## Stability summary

| Feature | Mean R² | Std R² | Min | Max |
|---|---:|---:|---:|---:|
| `physical_depth` | 0.7847 | 0.0059 | 0.7764 | 0.7897 |
| `physical_two_qubit_depth` | 0.7720 | 0.0045 | 0.7661 | 0.7769 |
| `qcre_weighted_critical_path_seconds` | 0.6628 | 0.0138 | 0.6437 | 0.6756 |

A narrow range supports reporting a central estimate; a wide range
would require reporting the seed range and treating one seed as
exploratory. This check changes only transpiler seed, not labels,
target semantics, grouped split or calibration protocol.
