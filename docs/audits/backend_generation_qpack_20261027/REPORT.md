# Backend generation and QPack analytical-duration audit

Read-only saved-evidence analysis, 2026-10-27. No fitting, simulator/QPU
execution, transpilation, new splits or edits to frozen results.

## Verified backend identity

`backend_inventory.csv` reads `processor_type` and `n_qubits` directly from
the hash-pinned IBM FakeBackend configuration files named in
`artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv`.
The files describe archived hardware configurations, not synthetic runtime
labels. Current cloud listings must not replace these historical identities.

| Processor revision | Backends | Hardware width | Rows |
|---|---|---:|---:|
| Falcon r4T | Belem, Lima, Quito | 5 | 1116 |
| Falcon r5.11L | Manila | 5 | 367 |
| Falcon r5.11H | Jakarta, Lagos, Nairobi, Perth | 7 | 2692 |
| Eagle r3 | Osaka, Kyoto | 127 | 340 |

Falcon basis gates include CX; Eagle snapshots use ECR. These are processor
families/revisions, not arbitrary ordinal generation numbers. No Heron cohort
is represented. Snapshot properties date from 2021 for Falcon and 2023 for
Eagle; these dates are not proven job dates. All 340 Eagle observations are
Ma–Li; all 4175 Falcon observations are QPack/Qonductor. Generation and source
are confounded. This is not a controlled comparison of processor generations.
See `source_generation_coverage.csv` for exact source/backend counts.

Official context: [IBM retired QPUs](https://quantum.cloud.ibm.com/docs/en/guides/retired-qpus).
Exact revision assignments above come from the local hash-verified IBM
configuration paths in the inventory.

## Method performance by revision

The following primary diagnostic uses group-balanced MAE in seconds within
each stratum: compute each represented structural group's MAE in that stratum,
then average groups equally. These are slices of existing frozen OOF
predictions, not models retrained per generation or generation-held-out tests.
No inferential superiority claims or new CIs are assigned.

| Revision | Groups | Graph+MLP | MLP | Regression selector | Qiskit shots + affine | NoTaDS code shots + affine |
|---|---:|---:|---:|---:|---:|---:|
| Falcon r4T | 8 | 0.877601 | 1.366194 | 1.048653 | 2.203593 | 2.185121 |
| Falcon r5.11L | 8 | 1.113218 | 1.316223 | 0.948771 | 2.129972 | 2.124090 |
| Falcon r5.11H | 14 | 1.068326 | 0.881989 | 1.069952 | 1.456191 | 1.439460 |
| Eagle r3 | 148 | 0.572223 | 0.518615 | 0.615512 | 1.024715 | 1.018986 |

Groups can appear on multiple backends/revisions; group counts are not
additive. NoTaDS here is explicitly the learned affine adaptation, not its
raw dimensionless score. `method_stratum_metrics.csv` includes 13 declared
routes, coverage, pooled MAE, group-balanced MAE, bias, Pearson/Spearman,
ratios and backend/source slices. Hyb has missing predictions: its log-cost
Ridge covers only 192/340 Eagle rows. Do not compare partial populations as
matched leaderboards. Original saved overall MAE values replay exactly within
numerical tolerance.

## Qiskit QPack failure: magnitude, not execution failure

Qiskit provides 3945/3945 finite predictions. All raw and shot-scaled
predictions underpredict. All rows have 4096 shots.

| Quantity | Value |
|---|---:|
| Observed target median | 6.589695 s |
| Nominal circuit duration median, one shot | 18.360889 microseconds |
| Nominal duration × shots median | 0.075206 s |
| Median per-row prediction/target ratio | 0.013837 |
| MAE, unscaled duration | 6.066496 s |
| MAE, duration × shots | 5.957751 s |
| MAE, duration × shots + saved OOF affine | 0.964997 s |
| Within-source Pearson, shot-scaled raw | 0.408607 |

Thus omitting shot multiplication is not the whole explanation: multiplication
helps but leaves estimates far below observations. Including measurement is
also insufficient: this Qiskit route already includes measurement. Exact
measurement/reset/repetition/service boundary components in the archived
labels are unresolved; the residual must not be called measured overhead.

### Backend dependence within QPack

Pooled per-backend MAE below weights observations equally; it is not the
group-balanced table above.

| Backend | Rows | Target median | Qiskit shots median | Raw shots MAE | Affine MAE |
|---|---:|---:|---:|---:|---:|
| Lagos | 363 | 6.018800 | 0.063089 | 5.854802 | 0.192719 |
| Nairobi | 1546 | 6.894659 | 0.075206 | 6.782234 | 1.014294 |
| Perth | 326 | 4.485602 | 0.067182 | 4.433921 | 1.409287 |
| Jakarta | 367 | 6.863693 | 0.074332 | 6.818643 | 0.983571 |
| Manila | 334 | 6.447582 | 0.081337 | 6.472303 | 0.639561 |
| Quito | 1009 | 4.973107 | 0.068551 | 4.740387 | 1.124726 |

These backend means account for 84.1855% of total target squared variation
by the descriptive between/within identity. This is not a causal hardware
effect: workload mix, date and collection configuration are not controlled.

`qpack_group_backend.csv` provides a cleaner same-logical-group diagnostic.
For group `0832382feb8ab61485fbcc15d3826d3e43ba5e7470b3ade42033c9a0067226ee`,
mean observed runtime is 4.532329 s on Perth versus 6.839832 s on Jakarta;
nominal shot-scaled estimates are 0.077522 s versus 0.074332 s. The nominal
ordering is reversed, even though both are Falcon r5.11H and the logical
structural group and shots are shared. Exact angles, job dates and physical
submissions remain unobserved. Do not infer a causal slowdown mechanism.

### Representation limitation and repeated observations

QPack's 3945 observations map to 6 structural groups and 26 distinct saved
physical-graph/backend contexts. All 26 contexts have multiple observations;
the duration input and shot count are constant within each context. Therefore
this estimator cannot explain within-context runtime changes.

However, within-context variation accounts for only 4.6700% of total target
squared variation. It must not be presented as the main explanation for a
roughly 6-second raw error. That error primarily documents a large mismatch
between nominal duration and the observed target scale under these inputs;
the physical decomposition of that mismatch is not identified.

For example, one Manila context has 82 observations with the same nominal
0.202331-second workload estimate but targets from 6.396161 to 9.464657 s.
This is one source of irreducible ambiguity for the retained representation,
not a claim that physical execution is inherently unpredictable.

The reconstruction uses fixed RZ(0.3)/RX(0.2), recorded size and QAOA depth,
then a frozen nominal compile policy. Historical optimization angles, routing
and physical QASM are absent. Code:
`benchmark_v1/scripts/materialize_compiled_unified_qpu_v1.py:322-365`;
`benchmark_v1/scripts/run_recovery_c133_c135_v3.py:181-191`.
Changing angles can affect compilation, but its effect on these archived
runtimes is not measured. Snapshot mismatch and reconstructed routing are
limitations, not proven explanations of the error magnitude.

Target provenance: `docs/methodology.md:142-173` and
`docs/audits/qpu_timing_boundary/SOURCE_TARGET_SEMANTICS.md`. QPack milliseconds
are converted once to seconds. The label is per circuit evaluation, not the
entire optimization workflow. A missing or duplicated shots/unit conversion
is not supported by the saved numerical checks.

## Interpretation and unresolved pieces

Supported: method ranking is context-dependent; nominal circuit duration
contains some predictive association but badly underestimates QPack observed
seconds; coarse processor-family information does not explain backend offsets.
The calibrated scalar mainly aligns scale/offset and does not establish a
physical decomposition or a universally correct runtime boundary.

Unknown: exact historical calibration, compiled QPack instances, per-shot
repetition policy, server/controller/result-handling components and their
separate measured durations. A shared `time_taken` label does not resolve
these components. No new generation-specific mechanism or superiority claim.

## Reproduction and validation

Run from repository root:
`/home/server/Documents/.venv-qonductor/bin/python docs/audits/backend_generation_qpack_20261027/evaluate.py`.
`validation.json` records Git HEAD, original inputs/configuration hashes,
identity and saved-MAE validation. `qpack_row_diagnostics.csv` and
`qpack_constant_input_cells.csv` retain all rows and cell evidence.
No models or calibration coefficients were fitted. Frozen data, splits and
predictions remain unchanged. No commit or push.
