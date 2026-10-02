# S67 — Circuit representation V3

**Decision date:** 2026-09-29  
**Status:** signed; feature-sidecar materialization pending  
**Scope:** the common analytical and learned adaptations over the archived
8,767-row real-QPU ledger.

## One declared representation per source lifecycle

| Source | Rows | V3 common representation | Identity claim |
| --- | ---: | --- | --- |
| Ma--Li | 340 | exact logical QASM, deterministically transpiled to that row backend's pinned Target | exact logical input; resulting compiled circuit is a V3 reconstruction |
| Qonductor | 4,482 | exact submitted physical QASM | exact submitted/compiled input; do not transpile again merely to derive common structural features |
| QPack MCP | 3,945 | pinned angle-insensitive QAOA structure reconstructed from recorded workflow fields, then deterministically transpiled | reconstruction-qualified; neither exact submitted QASM nor optimizer angles |

`representation_authority_v2.json` remains the immutable evidence authority
for Qonductor P0/L1/L2/L4 and the QPack generator pin.  This V3 decision is
the feature-layer overlay: it makes the table above the only input route for
common analytical and learned adaptations.  It does not claim an inverse
transpilation of Qonductor P0 or an exact QPack circuit.

## Sidecar requirement

`C134` creates a new feature sidecar and does not edit canonical raw targets.
Each attempt receives:

```text
canonical_row_id, source_id, representation_id, lifecycle_stage,
representation_digest, compiled_or_submitted_digest, target_snapshot_id,
active_width, structural_depth, one_qubit_count, two_qubit_count,
swap_like_count, measurement_count, shots, circuit_count=1,
availability_status, terminal_reason
```

All feature values originate in the declared circuit representation before
execution.  A row with a missing feature is an explicit failed/unavailable
method attempt; it is not completed with a source-specific median or mask.

## Required Qonductor semantic audit

Before retaining any source-local C122 prediction, audit every submitted QASM
against the archive columns.  The report must specifically determine whether
the historical labels were swapped semantically:

```text
archive sum_depth       <-> QASM two-qubit count
archive sum_qubits      <-> QASM structural depth
archive sum_2q_gates    <-> QASM active width
```

The audit is an evidence table, not a name-based guess.  If all 4,482 mappings
are semantically faithful after a documented column interpretation, C122 stays
as source-local with a correction sidecar.  Any substantive mismatch requires
new source-local features and a grouped rerun; a routine worker cannot choose
which interpretation produces better MAE.

## Representation prohibitions

- Do not treat Qonductor physical QASM as byte-exact original logical QASM.
- Do not attach an MQTBench recipe without the existing L1 archive hash join.
- Do not derive a logical circuit from width, family, backend capacity, target
  duration or observed runtime.
- Do not manufacture QPack angles, routes, per-iteration submitted QASM, or a
  missing structural feature.
- Do not pool representation identity/tier as a learned source proxy.  It is
  retained for coverage and stratified reporting only.

## Deterministic target-transpilation policy

Only Ma--Li and QPack require V3 target transpilation.  Its seed is derived
from `seed_registry.json` stream `20260925-transpile`; the target snapshot,
Qiskit version, transpiler settings, output digest and terminal failure are
recorded.  Qonductor P0 is re-transpiled only in a separately named
target-compatibility/schedule method, never silently for the common structural
feature sidecar.
