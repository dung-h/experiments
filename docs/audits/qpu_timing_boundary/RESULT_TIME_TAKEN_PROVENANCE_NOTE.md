# Provider `Result.time_taken` provenance note

**Purpose.**  This is a retrieval note for the Track-A QPU sources.  It does
not alter Scientific Freeze v2 labels, splits, predictions, or metrics.

## Verified common primitive

The Ma--Li, Qonductor and selected QPack MCP runtime labels originate from
the IBM/Qiskit provider result field conventionally exposed as
`Result.time_taken`. Different source-column names do not establish different
underlying runtime quantities.

- **Ma--Li:** the canonical label is the mean of three recorded
  `Result.time_taken` observations (`mean_of_three_recorded_time_taken_runs`),
  at fixed 1,024 shots.
- **Qonductor:** the canonical label is an archived one-circuit job's
  `Result.time_taken`, with row-varying shots.
- **QPack MCP:** the source `Circuit execution durations [ms]` comes from
  IBM `result["time_taken"]` for one QAOA circuit per optimizer evaluation;
  the canonical conversion is milliseconds divided by 1,000. See
  [`data_registry.json`](../../../benchmark_v1/registry/data_registry.json#L31),
  [`audit_qpack_mcp_structural_reconstruction.py`](../../../benchmark_v1/scripts/audit_qpack_mcp_structural_reconstruction.py#L51)
  and its documented source semantics at line 149.

The source registry and common-panel contract are the authority for this
statement: `benchmark_v1/registry/data_registry.json` and
`benchmark_v1/protocol/common_evaluation_panels_v2.json`.  The detailed
source audit is `SOURCE_TARGET_SEMANTICS.md` in this directory.

## Boundary that must remain explicit

Shared use of the `time_taken` field does **not** establish an identical
measurement contract:

| Source | Label aggregation | Shots | Circuit provenance |
|---|---|---:|---|
| Ma--Li | Mean of three archived runs | 1,024 fixed | Exact logical/source QASM |
| Qonductor | One archived job per row | Varies by row | Exact submitted physical QASM |
| QPack MCP | One circuit at an optimizer evaluation; ms converted to s | 4,096 fixed | Reconstructed logical structure; exact submitted circuit unavailable |

Historical metadata do not establish identical provider/runtime versions or
every internal timing component. Therefore use the wording **“all three
sources use provider-recorded `Result.time_taken` labels, with documented
aggregation and unit conversions.”** Neither identical internal accounting nor
different internal timing boundaries has been established across the sources.
In particular, source heterogeneity alone does not prove a timing-boundary
mismatch or prevent a matched comparison against the frozen targets.

## Correction recorded 2026-10-25

An earlier conversational answer described the three Track-A sources as
having different timing boundaries without direct evidence. That statement was
too strong. The shared provider-field lineage is verified; repetition
aggregation, units and circuit-input provenance differ. Unresolved provider
internals remain unknown. No target, metric, split or frozen conclusion changes.

## Related non-runtime dataset

The public QuEst (ICCAD 2022) release is a reliability/fidelity dataset.  Its
documented saved record is `[QASM/transpiled circuit, noise dictionary, PST]`;
the public generator calls `backend.run(...).result()` only to obtain counts
for PST and does not persist `result.time_taken`.
