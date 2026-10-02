# Wave 2B — Simulator reader pack v1

**Decision date:** 2026-09-30
**Status:** signed reader-facing simulator-domain pack; inventory only
**Artifact:** `artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/`
**Consumes:** Wave 1C `artifacts/benchmark_v3/simulator/reader_tables_v1/`
**Does not consume or mutate:** QPU reader packs v1/v2/v3, `CURRENT.json`, fidelity v1, analytical-wave files, V3-large predictions, raw simulator observations

## Decision

Wave 2B publishes a standalone simulator-domain reader pack as a sibling of the Wave 1C inventory tables. It is not a QPU appendix and must not be placed under `reader_table_pack_v1`, `v2`, or `v3`. The pack copies frozen Wave 1C numbers, source paths, and hashes. It may add `ranking_eligibility`, `equivalence_group`, and `claim_boundary` overlays. It does not invent runtimes, recompute unique-cell counts, time a simulator, train a model, start CUDA, or mutate git.

`CURRENT.json` remains historical until the Wave 3 authority-switch gate.

## Domain boundary

Archived observed QPU service time is a different clock from local simulator first/warm execution, quality-gated MPS execution, and selected-plan contraction estimates. This pack contains zero QPU rows and must not emit `archived_observed_service_execution_time` or QPU observation IDs.

There is no flat cross-engine or cross-clock leaderboard. Local measurements are not `primary_same_clock` QPU ranks.

## Identity

The common exact-QASM panel is `sim_common_q16_v1`: 204 unique `panel_member_id` values (162 `core_q2_q9`, 42 `frontier_q10_q16`). Identity is the panel-member row plus that row's exact QASM SHA-256. Declared hash aliases are allowed (191 unique `qasm_sha256`). Uniqueness of QASM hashes is not required.

## Method split and ranking

- Aer 204-row reduced-warm capsule: overlay method_id `qiskit_aer_noisy_local`, `fidelity_class=local_measurement`, `ranking_eligibility=not_rankable`. Registry V1 has no distinct Aer measurement ID. This capsule is not Azizov predictor coverage.
- Azizov-style predictor `azizov_aer_independent`: coverage-qualified on 162 core members only. Frontier is blocked, not imputed. `ranking_eligibility=not_rankable`.
- CUDA-Q dense `cudaq_dense_local`: first vs warm and FP32 vs FP64 never pooled. Unique cells 1224 per clock (1188 ok, 36 adapter_error). `not_rankable`.
- CUDA-Q MPS `cudaq_mps_quality_constrained`: FP64, bond-16, cutoff-1e-10, `gesvdj`. Unique cells 612 per clock (573 ok, 21 quality_failed, 18 adapter_error). Failures remain in denominators. `not_rankable`.
- cuTensorNet `cutensornet_runtime_est`: native pair only against matching selected-plan warm scalar contraction. `ranking_eligibility=paired_native_diagnostic` on that pair; still not a cross-engine rank. Path-search and network-build are `not_an_estimator_target` telemetry (`not_rankable`). Unique cells 612 per clock (603 ok, 9 timeout).
- Maestro both IDs, Ma--Li simulator source-native, structural DAG adaptation, family-aware residual, Azizov frontier, and Pasqal analog companion: blocked/unavailable. Pasqal is a separate analog pulse/layout panel, not digital OpenQASM.

Equivalence groups on this pack are `none_declared`. Qiskit/QCRE schedule equivalence is a QPU-analytical fact and is not restated here as a simulator winner.

## Immutability

Wave 1C tables, QPU packs v1/v2/v3, fidelity v1, `CURRENT.json`, and raw evidence are not overwritten. The public pack directory is refuse-overwrite except for an explicit temporary `--output-dir` used by tests.
