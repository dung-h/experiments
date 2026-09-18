# Azizov independent reproduction — P1 q<=16 stratified (3 circuits/family)

The target is local Qiskit Aer noisy-simulator execution after backend-aware transpilation.
This report describes the local matrix only; it is not an exact reconstruction of the unavailable author artifact.

## Coverage

- Total rows: **504**.
- Successful rows: **504**.
- Status counts: `{'ok': 504}`.
- Unique circuits: **63**.
- Families: **21**.
- Backends: FakeSherbrooke, FakeWashingtonV2.
- Optimization levels: 0, 1, 2, 3.
- Selection: three smallest eligible circuits per family under the configured
  16-qubit bound; observed maximum width is 9 qubits.

## Runtime summary

| Backend | Opt | n successful | median T_transpile | median T_exec | p95 T_exec |
|---|---:|---:|---:|---:|---:|
| FakeSherbrooke | 0 | 63 | 0.0058 | 0.4031 | 1.6044 |
| FakeSherbrooke | 1 | 63 | 0.0087 | 0.3980 | 1.0209 |
| FakeSherbrooke | 2 | 63 | 0.0111 | 0.4031 | 0.8650 |
| FakeSherbrooke | 3 | 63 | 0.0120 | 0.4249 | 0.6477 |
| FakeWashingtonV2 | 0 | 63 | 0.0051 | 0.6074 | 0.8579 |
| FakeWashingtonV2 | 1 | 63 | 0.0076 | 0.6220 | 0.8154 |
| FakeWashingtonV2 | 2 | 63 | 0.0096 | 0.6202 | 0.7076 |
| FakeWashingtonV2 | 3 | 63 | 0.0107 | 0.6245 | 0.7910 |

## Observed boundaries

- Successful `T_exec` range: 0.3660–8.5006 s.
- Maximum logical width in successful rows: 9.
- Maximum transpiled DAG row size: 47342 nodes.

The matrix is checkpointed row-by-row. Timeout rows are retained as censored observations and must not be treated as ordinary zero or median targets in model training.
