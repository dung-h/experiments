# S69 — Qonductor feature authority and C122 adjudication

**Decision date:** 2026-09-29  
**Status:** signed; no new model fit is authorized by this decision alone  
**Scope:** Qonductor's 4,482 canonical IBM one-circuit observations, the
historical C122 outputs, and their V3 replacements.  This is additive: raw
observations and historical artifacts remain immutable.

## Decision in one sentence

The 4,482 canonical Qonductor rows have a verified direct source-DB
job-to-circuit-to-archived-QASM association; the five C134 failures disprove
only the claim that legacy archive `sum_*` columns are parsed-QASM feature
aliases.  They do not invalidate the QASM association or the V3 common
structural sidecar.

## Evidence and findings

The reproducible provenance audit
`artifacts/benchmark_v3/qonductor_db_qasm_provenance_20260929_v3/report.json`
joins the canonical raw record to Qonductor's SQLite `job` and `circuit`
tables, then checks the circuit's archived ZIP member and SHA-256.

| Check | Result |
| --- | ---: |
| canonical Qonductor IBM rows | 4,482 |
| canonical rows with direct DB / ZIP / digest check | 4,482 |
| failures | 0 |
| all-provider one-circuit DB jobs (context only) | 4,613 |
| C134 `sum_*`-to-QASM alias checks passing | 4,477 / 4,482 |
| C134 alias checks failing | 5 / 4,482 |

The additional 131 database jobs are not a discrepancy: the canonical source
is the frozen, enriched IBM eligibility extract, not every one-circuit job in
the source database.

The relevant source checkout is
`/home/server/Documents/Qonductor-SC25` at
`28477f54c1670709093c7cb43c84cc634eb75bc6`.  Its
`src/execution_time/regression_estimator.py` obtains its feature vector from
the circuit object as, in order, `depth`, `num_qubits`, `swap`, `shots`, and
`circuit_count`.  In that implementation, `swap` counts `cx`, `cz`, and
`ecr`; `depth` is its operation-stack depth; and `num_qubits` is the active
qubit set.  It does **not** consume the enriched extract's legacy
`sum_depth`, `sum_qubits`, or `sum_2q_gates` columns.

Consequently, C134's 4,477/5 result remains valuable evidence: the names
`sum_*` cannot be silently reinterpreted as the source estimator's QASM
features.  It is not evidence that the five submitted QASM files belong to
the wrong jobs.  The direct foreign-key and digest audit is the authority for
the latter claim.

## Feature authority by method

| Method / result class | Input authority | V3 treatment |
| --- | --- | --- |
| historical C122 V2 `source_poly_grid` | enriched archive `sum_depth`, `sum_qubits`, `sum_2q_gates` | Preserve immutable, relabel **archive-metadata adaptation**; it is not a source-code Qonductor reproduction and is not a V3 primary score. |
| Qonductor source-local V3 | exact submitted QASM parsed with the pinned upstream extraction semantics, plus archived `shots`, `circuit_count=1` | New grouped retrain required.  It is a leakage-safe **source-code adaptation**, because frozen grouped outer/inner splits replace upstream row-wise KFold. |
| unified Qonductor polynomial V3 | S68 common V3 QASM sidecar: active width, structural depth, two-qubit count, swap-like count, shots, one circuit | Remains eligible to attempt all 4,482 Qonductor rows.  It intentionally does not reuse archive `sum_*` columns. |
| unified Ma--Li graph V3 | S67 exact submitted physical QASM / V3 DAG | The five alias discrepancies do not affect the QASM association.  Dynamic control has its own method-specific rule below. |

No result may call the archive-metadata features `Qonductor source features`,
nor use their apparent 4,477-row correspondence to repair the remaining five
values.  They are distinct feature views and must have distinct method IDs.

## Dynamic-control row 4477

`qonductor_single_circuit_ibm|row4477` is an exact submitted OpenQASM 3
circuit with top-level `if_else`.  The preflight availability decision remains
authoritative for schedule-dependent methods: Qiskit duration, snapshot QCRE
critical path, and the schedule-dependent Hyb-HANAS component emit explicit
`unavailable` on this one row.

For learned structural methods, S69 freezes the following narrow policy:

- The source-local Qonductor V3 extractor may process the row because the
  pinned upstream code accepts the parsed circuit and treats `if_else` as its
  top-level operation.  Its result is therefore source-code-semantics, not a
  branch-duration estimate.
- Unified polynomial V3 may use the direct, top-level structural sidecar for
  the row.  It must record `control_flow_policy=opaque_top_level_operation`;
  it does not claim to count or time the unrolled branches.
- Unified Ma--Li graph V3 and `qcre_gate_aware_depth_rank` emit explicit
  `unavailable` for the row unless a later signed contract specifies a
  branch-aware DAG / control-flow cost representation.  A flat DAG that
  omits branch contents would not be the S68 declared graph input.

This method-specific availability difference retains the canonical observed
label and the full-envelope denominator.  It is not a deletion, a target
substitution, or an invitation to fabricate feedback latency.

## Supersession and execution gates

This decision refines the C122 line of
`WAVE0_SUPERSESSION_RECORD_V3_20260929.md`:

1. Historical C122 is no longer merely “pending S67 adjudication”; it is
   quarantined from source-code-primary comparisons as an
   `archive_metadata_adaptation_v2`.
2. A fresh source-local Qonductor V3 runner must materialize upstream QASM
   features, pin the source code and QASM parser, use the existing frozen
   grouped splits, and produce all five outer folds before it receives a V3
   score.
3. The common structural C134 sidecar is admissible for the S68 unified
   polynomial after the availability overlay is applied.  Its original
   `mapping_status` field is a legacy-alias diagnostic only, not its
   row-to-QASM provenance status.

S69 permits routine work to materialize an eligibility overlay and to build
the two fresh V3 runners.  It does **not** authorize their training, full
analytical runs, or final aggregation until the required runner and first-fold
QA gates are signed.

## Pinned evidence and reproduction

- Canonical ledger SHA-256:
  `920d745e03dd00e8155118d9323baea18ff865e4df2620f3a059bd5503a05ba4`
- Source SQLite SHA-256:
  `5c3ba115956ead6c131fa4bfed32112edc7b818ff0de79e92626e92f19d0183c`
- QASM ZIP SHA-256:
  `c9f259d38b24658e303c73c020c0c35e38d26d835061b19afc67626f8b70f51b`
- Upstream estimator SHA-256:
  `57b70e7dc111e27fdce9fc9f3f891ee9a586704ea4ffd4d4f6c00337ec0b44e7`
- Upstream QASM archive loader SHA-256:
  `5b6c19de2104719f950878645378063cad0427d8a95b8b7dff7107046dbac216`
- C134 alias audit SHA-256:
  `f2241d9f83522256b5c25a3435d6a508461184266696ec643f5c095b442e74d4`
- direct DB/QASM report SHA-256:
  `6aaa4d620999f68be53a93b48db13baf6e813cd72e3e3e45a5aa43f980c137d1`

Reproduce the provenance result without fitting a model:

```bash
python3 benchmark_v1/scripts/audit_qonductor_db_qasm_provenance_v3.py \
  --output-dir artifacts/benchmark_v3/qonductor_db_qasm_provenance_reproduction
```
