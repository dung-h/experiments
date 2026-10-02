# Simulator reader-table aggregation contract v1

Date: 2026-09-30. This is a reader-layer, evidence-only aggregation contract for the frozen local simulator artifacts. It neither schedules a timing run nor trains or evaluates a learned model.

## Scope and identity

The common exact-QASM panel is `sim_common_q16_v1`: 204 unique `panel_member_id` values, comprising the 162-member `core_q2_q9` stratum and the 42-member `frontier_q10_q16` stratum. Identity authority is the panel-manifest row plus that row's exact QASM SHA-256. Declared hash aliases are allowed: 191 unique `qasm_sha256` values is not a defect, and uniqueness of QASM hashes is not required. Uniqueness of `panel_member_id` is required. The reader table must retain all terminal statuses in denominators; it must not impute a time for an `adapter_error`, `quality_failed`, timeout, or other unavailable state. This is an inventory-only aggregation.

Every numeric reader row carries its direct evidence `source_path` and `source_sha256`, plus both `evaluation_target_clock` and `method_output_clock`. A common input panel does not imply that those clocks are comparable.

## Permitted table classes

* Aer local structural evidence is a local noisy-Aer clock published under reader overlay method_id `qiskit_aer_noisy_local` (`fidelity_class=local_measurement`). Registry V1 has no distinct Aer measurement ID; `azizov_aer_independent` is reserved for the Azizov-style predictor and must not label the 204-row measurement capsule. The Azizov-style local reimplementation is coverage-qualified on the 162 core members only; frontier predictor rows are blocked, not imputed. The Aer measurement capsule remains a local engine/configuration record, not Azizov coverage and not a QPU comparison.
* CUDA-Q dense FP32 and FP64 first and warm sample clocks are separate local descriptive configurations. Precision and clock are never pooled. All 36 adapter-error cells remain in each relevant denominator.
* CUDA-Q MPS is local descriptive evidence only under the fixed FP64, bond-16, cutoff-1e-10, `gesvdj` quality contract. Its 18 adapter-error and 21 quality-failed cells remain visible. A quality-gated MPS row is not a dense-equivalence assertion.
* cuTensorNet `RUNTIME_EST` is a native paired diagnostic only against the warm scalar-contraction clock from the same selected plan, workspace, precision, and session. Path-search and network-build values are descriptive telemetry, including the S40--S42 interference qualification, and receive neither a predictor score nor a rank.
* Maestro candidate evidence is not included as common-panel runtime evidence because the available matrix is an archived-QPU-context candidate grid, not a complete held-out common-panel candidate/quality matrix. The paper-exact selector is unavailable.
* Pasqal EMU-MPS remains a separately labeled analog pulse/layout panel. It is not mapped into the digital OpenQASM panel and its pilot formula is not promoted.
* Ma--Li simulator source-native and structural simulator-adaptation claims, the family-aware ladder, and any other method without row-aligned frozen local evidence are blocked/unavailable rather than substituted.

## Reader restrictions

There is no flat cross-engine or cross-clock rank. The required tables are descriptive availability, exact-panel identity, configuration-specific metrics, native-method appendix, and blocked/unavailable evidence. No row is a QPU observation. The builder may only hash and parse existing evidence; it has no timing, training, GPU, or network actions. The validator rejects a missing source hash, duplicate metric composite key, incompatible panel/clock combination, arithmetic mismatch, or an output that attempts to expose a QPU row.
