# Azizov independent reproduction — local feasible subset (q<=9, ops/depth<=4000)

The target is local Qiskit Aer noisy-simulator execution after backend-aware transpilation.
This report describes the local matrix only; it is not an exact reconstruction of the unavailable author artifact.

## Coverage

- Total rows: **1192**.
- Successful rows: **1192**.
- Status counts: `{'ok': 1192}`.
- Unique circuits: **149**.
- Families: **21**.
- Backends: FakeSherbrooke, FakeWashingtonV2.
- Optimization levels: 0, 1, 2, 3.

## Runtime summary

| Backend | Opt | n successful | median T_transpile | median T_exec | p95 T_exec |
|---|---:|---:|---:|---:|---:|
| FakeSherbrooke | 0 | 149 | 0.0063 | 0.4769 | 1.8078 |
| FakeSherbrooke | 1 | 149 | 0.0101 | 0.4722 | 1.2296 |
| FakeSherbrooke | 2 | 149 | 0.0126 | 0.4658 | 1.1167 |
| FakeSherbrooke | 3 | 149 | 0.0143 | 0.4628 | 1.1294 |
| FakeWashingtonV2 | 0 | 149 | 0.0052 | 0.6355 | 1.0797 |
| FakeWashingtonV2 | 1 | 149 | 0.0079 | 0.6365 | 1.1740 |
| FakeWashingtonV2 | 2 | 149 | 0.0105 | 0.6322 | 0.9607 |
| FakeWashingtonV2 | 3 | 149 | 0.0120 | 0.6324 | 1.2917 |

## Observed boundaries

- Successful `T_exec` range: 0.3675–10.7751 s.
- Maximum logical width in successful rows: 9.
- Maximum transpiled DAG row size: 153548 nodes.

The matrix is checkpointed row-by-row. Timeout rows are retained as censored observations and must not be treated as ordinary zero or median targets in model training.
