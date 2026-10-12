> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Corrected findings

| Claim | Assessment | Correct wording |
|---|---|---|
| Frozen replay | SUPPORTED | All completed neural/formula/statistical replay comparisons match within declared tolerance. This is subset identity, not full replay coverage. |
| Graph versus MLP B1 | SUPPORTED WITH QUALIFICATION | Graph requires more warm host-return time on shared selected requests with frozen ensemble and input contracts. Not pure graph-architecture causal cost or production latency. |
| Dense overhead versus execution | SUPPORTED WITH QUALIFICATION | 27/30 and 28/30 Graph, 20/30 and22/30 MLP B1 medians exceed historical CUDA-Q warm sample32 labels, in different software/time environments with matched GPU model/driver. |
| Full preprocessing cost | INCONCLUSIVE | P1 shows saved-input preparation adds cost and tail variability; complete raw QASM B2 and isolated per-stage costs remain unmeasured. |
| Practical estimator-selection benefit | INCONCLUSIVE | Lower MLP compute cost is measured; accuracy requirements, strongest classical costs, hardware portability and scheduler gains are not established. |
| Native Qiskit slow/inaccurate | UNSUPPORTED | A setup AST parse hit cap before native inference. No native timing or accuracy qualification resulted. |
| Standalone cold start / incremental B1 memory | UNSUPPORTED | Shared-bundle setup/absolute process peaks do not identify these quantities. |

## Native parse evidence

Manual receipt: INCOMPLETE_REQUEST_TIMEOUT, 30 s signal alarm per setup stage, 120 s process bound. Failure path is `parse_frozen_compiled -> restore_saved_qasm3_hardware_wires -> openqasm3.parse`; no parse/transpile/timing rerun in this audit. The current path loads already compiled nominal physical circuits, so fresh transpilation was never intended. estimate_duration and frozen affine bridge are after all 50 parses and were not reached.

Per-observation private QPU example omitted from this publication view; original evidence is retained locally.

## Historical statements requiring clarification

1. accuracy_cost_analysis.md: add the category B software/time qualifier to execution ratios.
2. failures_and_exclusions.md: explain Maestro's missing output is frozen unsupported target/normalized metadata, not a file-retention loss.
3. validation_report.md/provenance_manifest.json intermediate storage counts differ. Both are historical non-final observations; final immutable payload size is audited here, not inferred from those fields.
4. Correct the summary interpretation of total duration: execution_manifest.wall_seconds covers the corrected coordinated run and excludes earlier attempts/manual native reconciliation. resource_closure is later full predeclaration elapsed. Neither is a pure estimator compute sum.

No historical source or raw evidence overwritten.
