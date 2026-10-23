# P0 coverage and provenance report

## Facts

- Frozen qualified panel: **4515 observations**, **165 groups**; source counts are {'mali_real_qpu': 340, 'qonductor_single_circuit_ibm': 230, 'qpack_mcp': 3945}.
- The pinned compiled-unified ledger joined all 4515 panel IDs.  This audit read 346 distinct saved graph files once each; no circuit was transpiled or regenerated.
- Available saved physical-DAG inputs: **4515/4515** observations; unique graph payloads within the qualified panel: **346**.  The graph-key count is **374**, because the same graph payload can occur with a distinct pinned snapshot identity.
- A saved deterministic nominal single-shot duration proxy is available for all 4515 IDs.  It is recorded as a timing proxy, not historical execution-day gate timing or a target label.
- The graph inventory is stored in `unique_graphs.csv`; exact node/edge quantiles and large-graph counts are in `dag_size_distribution.csv`.

### Coverage by source

| source_id | observation_rows | available_rows | unique_groups | unique_graph_files | unique_physical_graph_keys |
| --- | --- | --- | --- | --- | --- |
| mali_real_qpu | 340 | 340 | 148 | 300 | 300 |
| qonductor_single_circuit_ibm | 230 | 230 | 11 | 32 | 48 |
| qpack_mcp | 3945 | 3945 | 6 | 14 | 26 |

### Coverage by provenance

| structure_provenance | timing_provenance | observation_rows | available_rows | unique_groups |
| --- | --- | --- | --- | --- |
| EXACT_SUBMITTED_PHYSICAL | FROZEN_SNAPSHOT_PROXY | 230 | 230 | 11 |
| FROZEN_RECONSTRUCTION | FROZEN_SNAPSHOT_PROXY | 4285 | 4285 | 154 |

### Saved physical-DAG size distribution

| measure | n | median | p90 | p95 | p99 | max | over_50000 | over_100000 | over_300000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dag_node_count | 346 | 240464.5 | 322596.5 | 337292.25 | 358109.9 | 372846 | 300 | 284 | 76 |
| dag_edge_count | 346 | 284132.0 | 380968.5 | 398588.75 | 423417.05 | 440324 | 300 | 291 | 158 |

## Provenance interpretation

`EXACT_SUBMITTED_PHYSICAL` applies only to the Qonductor route which preserves archived submitted physical QASM.  `FROZEN_RECONSTRUCTION` means the saved physical view was deterministically materialized from the declared Ma–Li logical or QPack structural route; it does **not** assert recovery of the historical executed compilation.  Snapshot tiers are carried separately and are not promoted to historical calibration timing.

## Decision

P0 is an availability/provenance audit only.  It has not evaluated H1–H4, fitted a learner, extracted new timing features, or authorized P1/P2.  Any P1 comparison must use the saved feature/graph identity and an exact common-success population rather than substituting newly transpiled inputs.

## Inputs

- `artifacts/real_qpu/qualified_logical_panel/panel.csv` SHA256 `b4ade914955d7807e466cf5cf0c5265d21b39a81fdcee8e91dc2fb6300eb978a`
- `artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv` SHA256 `cad9f42f414c5fc551aef1a52d715dcb13aee6636cda9972d8bf970e2d84e802`
- `artifacts/real_qpu/qualified_logical_panel/analytical/attempts.csv` SHA256 `df87419ffe4d10b5a201108f40f4a8817713112e14a306e0e55ccf55ef7a1ed8`
- `artifacts/postfreeze_track_a_transpiled_representation_v1/PROTOCOL.md` SHA256 `09519f8281f9a2e7785ca539c897dae6afb0f59f0e3d8e364c7ce0b699537aa8`
