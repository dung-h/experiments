# S24: dataset semantics and common evaluation panels

Date: 2026-09-27. Status: **authoritative for all new runs and final thesis
reporting**.

This decision removes an ambiguity that appeared in earlier notes: a target,
a split-group identity, and a circuit representation are three different
things. In particular, a QPack `workflow_id` is a leakage-control group. It is
not a "workflow-duration" target.

## 1. Locked vocabulary

| Term | Meaning |
|---|---|
| observation target | The archived or locally measured quantity predicted by a method. |
| observation unit | The entity to which the target belongs, such as one submitted circuit job. |
| split group | Rows that must stay together to prevent leakage. It does not redefine the target. |
| representation view | The pre-execution input exposed to a method: logical QASM/DAG, submitted physical QASM/DAG, or common tabular fields. |
| evaluation panel | A frozen set of observation IDs or circuit hashes on which eligible methods are evaluated. |
| compatibility key | The fields that must agree before numerical method comparison is allowed. |

The required comparison key is:

```text
(evaluation_panel_id, target_id, clock_id, test_row_set_hash, split_id)
```

Rows missing any component cannot enter the same numerical leaderboard cell.
Representation coverage, failure/OOM/timeout rate, and abstention are reported
beside prediction error; they are not removed from the denominator.

## 2. Archived real-QPU source semantics

The canonical envelope contains 8,767 direct one-circuit observations. It is a
shared evaluation envelope, not an assertion that every method can consume
every representation.

| Source | Rows | Observation target and unit | Split group | Representation boundary |
|---|---:|---|---|---|
| Ma--Li | 340 | Mean of three archived IBM `result.time_taken` values for one submitted circuit at 1,024 shots | exact logical-QASM hash | Exact logical, pre-transpilation QASM is present. |
| Qonductor | 4,482 | Archived IBM `Result.time_taken` for a one-circuit job; shot count varies | exact submitted-QASM hash | Exact submitted/post-transpilation QASM is present for every row. |
| QPack MCP | 3,945 | IBM `Result.time_taken` for the single measured QAOA circuit executed in one optimizer evaluation, stored as circuit-execution milliseconds and converted by `/1000` | all iterations from the same one of 46 optimizer workflows | Exact submitted QASM and per-iteration angles are absent. The workflow is only a group identity. |

QPack IC and RH are excluded from this single-circuit envelope. Their VQE
evaluation aggregates multiple measurement circuits, so their label has no
single-circuit referent. Queue, QJob, optimizer, and total-workflow durations
are provenance/other targets and must never substitute for MCP circuit
execution time.

All three included sources preserve the IBM provider `time_taken` primitive,
but collection protocols, shots, hardware dates, and representation stages
differ. Primary error metrics therefore remain source-stratified. A pooled
row-weighted score is diagnostic only; source-balanced and QPack
workflow-cluster-balanced summaries are mandatory.

## 3. Qonductor logical reconstruction boundary

Qonductor did not lose the submitted circuit: `circuits.zip` contains the
exact circuit returned by `qiskit_job.circuits()` for every canonical row.
The separately linked original logical circuit was not archived.

The current bounded audit over 4,482 rows assigns:

| Tier | Rows | Permitted wording |
|---|---:|---|
| `version_pinned_logical_candidate` | 3,065 | regenerated logical candidate from the public MQT Bench recipe/version evidence |
| `structural_candidate_only` | 605 | structurally usable candidate; not original logical QASM |
| `structural_recipe_only` | 661 | recipe-level structural identity only |
| `unavailable_named_source` | 151 | no defensible logical candidate |

None of these tiers may be called byte-exact recovery of the original
pre-transpilation QASM. The public dependency declares a lower bound rather
than an exact collection lock, so regenerated candidates require a pinned
environment and retain their tier. Exact submitted-physical and regenerated-
logical views are separate method inputs and separate scorecard strata.

## 4. QPack reconstruction boundary

QPack MCP constructs one regular-graph MaxCut QAOA circuit and calls the IBM
device once per optimizer evaluation. The adapter maps its duration to IBM
`result["time_taken"]`; the QPack output stores this under `Circuit execution
durations [ms]`.

A structural-DAG adaptation may be promoted only when reconstruction is pinned
to LibKet/QPack stable revision
`9beaf65e951e01181b1e324cbb1b227af10eef96` and validates recorded problem,
size, `P`, qubit count and depth. This representation is suitable only for a
method that does not require exact per-iteration angles or submitted physical
routing. Until that validation passes, the method-row cell is `unavailable`,
not silently dropped. Exact-QASM or compiled-routing methods remain
unavailable on QPack even if the structural candidate passes.

## 5. Frozen real-QPU panels

| Panel ID | Rows | Purpose |
|---|---:|---|
| `rqpu_common_tabular_v2` | 8,767 | Same archived observation rows for methods whose declared pre-execution inputs exist across all three sources. Reports source-stratified primary metrics plus diagnostic pooled/transfer views. |
| `rqpu_logical_structural_v1` | coverage-dependent | Ma--Li exact logical QASM plus only validated Qonductor/QPack structural candidates. Coverage and tier are part of every result. |
| `rqpu_submitted_physical_v1` | 4,482 core | Exact Qonductor submitted physical QASM. Cross-source reconstructed/compiled views are separately named adaptations, never silently mixed into this core. |

"Same dataset" therefore means that every registered method is evaluated on
the same frozen panel wherever its input contract is available, and every
ineligible row remains visible in the eligibility/coverage matrix. It does not
mean inventing missing QASM or treating logical and physical lifecycle stages
as identical.

## 6. Frozen simulator panel up to 16 qubits

The common simulator source is the exact Ma--Li/MQT-Bench independent-QASM
corpus under `Quantum-Execution-Time-Prediction/data/quantum_circuits` selected
by `*_indep_qiskit_<n>.qasm` with `2 <= n <= 16`.

`sim_common_q16_v1` contains **204 exact QASM files**, 22 families, and all
available widths 2--16. The ordered-member digest is:

```text
SHA256(SHA256(file_bytes) + two spaces + basename + newline, basenames sorted)
= d756e36d5c8af2ab582761c2556d44ff1ce6fbd3d479c6450c1571bc6e9a3d49
```

The existing corpus is intentionally non-rectangular; no circuit is invented
to make a family-by-width grid look complete.

- `sim_common_q16_v1/core`: the 162 source circuits with q <= 9, used for the
  complete feasible accuracy core where an engine can finish under its frozen
  resource contract.
- `sim_common_q16_v1/frontier`: the 42 source circuits with q = 10--16, used
  to measure scaling, timeout, OOM, planner failure and quality-constrained
  coverage.

The earlier Aer run used a 149-circuit pre-screened q<=9 manifest. It remains
valid evidence for that historical subpanel, but it is **not** relabelled as a
complete 162-row `sim_common_q16_v1/core` run. When mapped to the new panel it
has at most 149/162 circuit coverage, subject to an exact-hash join; the 13
unattempted panel circuits remain visible as unmeasured/unsupported rather
than disappearing.

Every simulator receives the same ordered circuit hashes, converted only to
its method-specific IR. A result remains on its own target and clock:
compile/build, path search, first execution, warm execution, extraction and
end-to-end are not interchangeable. Dense sampling, scalar tensor-network
contraction and quality-constrained MPS are different target strata. Failed,
timed-out, OOM and quality-failed rows remain in coverage denominators.

No new simulator timing run is authorized until a materialized panel manifest
pins all 204 per-file hashes, IR conversion version, host/resource profile,
timeout and clock boundaries.

## 7. Supersession and immutability

This decision corrects target wording in `REAL_QPU_BENCHMARK_CONTRACT_V1.md`,
`S23_METHODOLOGY_AUDIT_20260927.md`, the v1 archived-data limitation text and
old QPack runner/plan names. Those files and existing artifacts remain as
historical provenance; raw data, splits and completed predictions are not
rewritten.

For all new code, cards, tables and thesis prose:

- use `QPack MCP direct one-circuit IBM Result.time_taken` for the target;
- use `QPack optimizer-workflow group` only for splitting/cluster weighting;
- say `Qonductor regenerated logical candidate` with its tier, never
  `recovered original logical QASM`;
- use `sim_common_q16_v1` as the shared simulator circuit panel and report the
  q<=9 core separately from the q10--16 frontier.

The machine-readable authority is
`protocol/common_evaluation_panels_v2.json`. Any future change requires a new
contract/panel ID, a supersession note and new hashes; it must not mutate this
decision in place after measurements depend on it.

## 8. Primary-source anchors

- Ma--Li paper and public implementation:
  <https://arxiv.org/abs/2411.15631> and
  <https://github.com/mooselab/Quantum-Execution-Time-Prediction>.
- Qonductor paper and public repository:
  <https://arxiv.org/abs/2408.04312> and
  <https://github.com/manosgior/Qonductor-SC25>.
- QPack paper and the pinned LibKet/QPack source revision:
  <https://arxiv.org/abs/2205.12142> and
  <https://gitlab.com/libket/qpack/-/tree/9beaf65e951e01181b1e324cbb1b227af10eef96>.
- MQT Bench circuit-source project: <https://github.com/cda-tum/mqt-bench>.
