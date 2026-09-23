# Ma--Li family feature-space audit

**Finding recorded:** 2026-09-22.

This is Experiment A: a structural comparison of the 10 MQT families
that appear in the Ma--Li Osaka/Kyoto observed-runtime table against
the 12 families present in the local MQT Bench extract but absent from
that hardware table. It does **not** score a runtime estimator and it
does not create labels for the missing families.

Target semantics remain unchanged. The 340 hardware rows are observed
`result.time_taken` on retired `ibm_osaka` / `ibm_kyoto` jobs. The 12
missing families have no such labels here. Physical features, when
present, are reconstructed from the current FakeOsaka/FakeKyoto
snapshot (`CURRENT_FAKE_SNAPSHOT_PROXY`).

## Coverage

- MQT QASM files indexed: 1510
- Selected families: qft, qftentangled, qpeexact, qpeinexact, qnn, random, su2random, realamprandom, twolocalrandom, qwalk-noancilla
- Missing families: ghz, wstate, graphstate, dj, ae, vqe, qaoa, portfoliovqe, portfolioqaoa, grover-noancilla, grover-v-chain, qwalk-v-chain
- Hardware-sampled QASM hashes: 170
- Logical rows ok: 1510
- Physical rows ok: 3020
- Physical errors/timeouts: 0

Experiment B remains blocked: `ibm_osaka` retired 2024-08-13 and
`ibm_kyoto` retired 2024-09-05, and this workspace has no
`result.time_taken` for the 12 missing families.

## Logical structure, selected vs missing

Standardized log1p centroid distance: **4.462**.
Leave-one-family nearest-group accuracy: **0.909**
(961 selected circuits, 549 missing).

| Feature | Median selected | Median missing | Ratio missing/selected | KS | p |
|---|---:|---:|---:|---:|---:|
| `logical_width` | 74.000 | 61.000 | 0.824 | 0.131 | 0.0000 |
| `logical_depth` | 253.000 | 64.000 | 0.253 | 0.616 | 0.0000 |
| `logical_two_qubit_depth` | 246.000 | 51.000 | 0.207 | 0.730 | 0.0000 |
| `logical_two_qubit_count` | 4606.000 | 81.000 | 0.018 | 0.837 | 0.0000 |
| `logical_gate_count` | 5301.000 | 217.000 | 0.041 | 0.826 | 0.0000 |
| `logical_parallelism` | 18.686 | 3.091 | 0.165 | 0.705 | 0.0000 |
| `interaction_n_edges` | 2628.000 | 68.000 | 0.026 | 0.838 | 0.0000 |
| `interaction_density` | 1.000 | 0.034 | 0.034 | 0.849 | 0.0000 |
| `interaction_mean_degree` | 71.000 | 1.981 | 0.028 | 0.850 | 0.0000 |
| `interaction_max_degree` | 72.000 | 2.000 | 0.028 | 0.661 | 0.0000 |

### Width buckets

The selected/missing gap is not only a different width mix. At 91--127
qubits the two-qubit-count and interaction-density distributions do
not overlap. At 2--16 qubits the same comparison is milder.

| Bucket | Feature | Median selected | Median missing | KS |
|---|---|---:|---:|---:|
| 2-16 | `logical_depth` | 22.000 | 13.000 | 0.268 |
| 2-16 | `logical_two_qubit_count` | 30.000 | 13.000 | 0.331 |
| 2-16 | `interaction_density` | 1.000 | 0.500 | 0.595 |
| 17-50 | `logical_depth` | 142.000 | 46.000 | 0.618 |
| 17-50 | `logical_two_qubit_count` | 1395.000 | 46.000 | 0.807 |
| 17-50 | `interaction_density` | 1.000 | 0.054 | 0.807 |
| 51-90 | `logical_depth` | 250.000 | 76.000 | 0.683 |
| 51-90 | `logical_two_qubit_count` | 4134.000 | 79.000 | 0.904 |
| 51-90 | `interaction_density` | 1.000 | 0.029 | 0.904 |
| 91-127 | `logical_depth` | 388.000 | 110.500 | 0.782 |
| 91-127 | `logical_two_qubit_count` | 11389.000 | 114.500 | 1.000 |
| 91-127 | `interaction_density` | 1.000 | 0.018 | 1.000 |

### Hardware sampling inside the 10 selected families

The hardware table is not a uniform sample of those families. The
comparison below is selected-and-sampled versus selected-but-unsampled.

| Feature | Median HW-sampled | Median unsampled | Ratio unsampled/sampled | KS | p |
|---|---:|---:|---:|---:|---:|
| `logical_width` | 91.000 | 70.000 | 0.769 | 0.323 | 0.0000 |
| `logical_depth` | 294.000 | 218.000 | 0.741 | 0.475 | 0.0000 |
| `logical_two_qubit_depth` | 289.000 | 213.000 | 0.737 | 0.476 | 0.0000 |
| `logical_two_qubit_count` | 6497.000 | 3444.000 | 0.530 | 0.524 | 0.0000 |
| `logical_gate_count` | 7194.000 | 3871.000 | 0.538 | 0.632 | 0.0000 |
| `logical_parallelism` | 25.249 | 16.810 | 0.666 | 0.426 | 0.0000 |
| `interaction_n_edges` | 4155.500 | 2415.000 | 0.581 | 0.334 | 0.0000 |
| `interaction_density` | 1.000 | 1.000 | 1.000 | 0.040 | 0.9540 |
| `interaction_mean_degree` | 90.000 | 67.900 | 0.754 | 0.345 | 0.0000 |
| `interaction_max_degree` | 90.000 | 69.000 | 0.767 | 0.332 | 0.0000 |

### Leave-one-family group assignment

| Family | True group | Predicted | d(selected) | d(missing) |
|---|---|---|---:|---:|
| `ae` | missing | selected | 1.781 | 3.418 |
| `dj` | missing | missing | 4.132 | 2.775 |
| `ghz` | missing | missing | 4.824 | 2.162 |
| `graphstate` | missing | missing | 5.391 | 3.263 |
| `grover-noancilla` | missing | missing | 4.939 | 2.108 |
| `grover-v-chain` | missing | missing | 5.392 | 1.762 |
| `portfolioqaoa` | missing | missing | 4.960 | 1.631 |
| `portfoliovqe` | missing | missing | 4.995 | 1.644 |
| `qaoa` | missing | missing | 6.120 | 2.077 |
| `qft` | selected | selected | 1.033 | 4.460 |
| `qftentangled` | selected | selected | 1.007 | 4.438 |
| `qnn` | selected | selected | 1.255 | 5.557 |
| `qpeexact` | selected | selected | 0.715 | 4.466 |
| `qpeinexact` | selected | selected | 0.710 | 4.489 |
| `qwalk-noancilla` | selected | missing | 4.771 | 3.436 |
| `qwalk-v-chain` | missing | missing | 5.465 | 1.793 |
| `random` | selected | selected | 1.253 | 5.305 |
| `realamprandom` | selected | selected | 0.871 | 5.125 |
| `su2random` | selected | selected | 0.897 | 5.153 |
| `twolocalrandom` | selected | selected | 0.904 | 5.160 |
| `vqe` | missing | missing | 6.025 | 1.788 |
| `wstate` | missing | missing | 4.453 | 2.340 |

## Physical proxy structure, selected vs missing

These columns use current FakeOsaka/FakeKyoto transpilation. They
answer whether compiled depth, routing and duration-weighted path
also separate the two family groups. They are not observed QPU
runtimes.

Physical centroid distance: **5.875**.
Leave-one-family nearest-group accuracy: **0.909**.

| Feature | Median selected | Median missing | Ratio missing/selected | KS | p |
|---|---:|---:|---:|---:|---:|
| `logical_width` | 74.000 | 61.000 | 0.824 | 0.131 | 0.0000 |
| `logical_depth` | 253.000 | 64.000 | 0.253 | 0.616 | 0.0000 |
| `logical_two_qubit_depth` | 246.000 | 51.000 | 0.207 | 0.730 | 0.0000 |
| `logical_two_qubit_count` | 4606.000 | 81.000 | 0.018 | 0.837 | 0.0000 |
| `logical_gate_count` | 5301.000 | 217.000 | 0.041 | 0.826 | 0.0000 |
| `logical_parallelism` | 18.686 | 3.091 | 0.165 | 0.705 | 0.0000 |
| `interaction_n_edges` | 2628.000 | 68.000 | 0.026 | 0.838 | 0.0000 |
| `interaction_density` | 1.000 | 0.034 | 0.034 | 0.849 | 0.0000 |
| `interaction_mean_degree` | 71.000 | 1.981 | 0.028 | 0.850 | 0.0000 |
| `interaction_max_degree` | 72.000 | 2.000 | 0.028 | 0.661 | 0.0000 |
| `physical_depth` | 17041.500 | 275.000 | 0.016 | 0.832 | 0.0000 |
| `physical_two_qubit_depth` | 4829.000 | 81.000 | 0.017 | 0.831 | 0.0000 |
| `physical_two_qubit_gate_count` | 27129.000 | 182.000 | 0.007 | 0.843 | 0.0000 |
| `physical_swap_count` | 0.000 | 0.000 | NA | 0.000 | 1.0000 |
| `physical_gate_count` | 149748.500 | 1249.500 | 0.008 | 0.841 | 0.0000 |
| `qcre_weighted_critical_path_seconds` | 0.003 | 0.000 | 0.017 | 0.831 | 0.0000 |
| `routing_depth_ratio` | 62.214 | 7.592 | 0.122 | 0.808 | 0.0000 |
| `routing_two_qubit_ratio` | 5.699 | 3.000 | 0.526 | 0.454 | 0.0000 |

## What this can and cannot claim

- It can show whether the 12 missing families occupy a different
  region of logical, and if computed, compiled feature space.
- It can show that Ma--Li's 170 hardware QASMs are a biased subset
  even of the 10 selected families.
- It cannot convert that structural gap into a runtime score for
  the 12 missing families.
- Family-held-out metrics on the 340 rows remain a statement about
  the 10 selected families only.
