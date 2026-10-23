# QPU target semantics audit

## Scope and provenance

This is a read-only post-freeze audit at repository commit `146886cff0d3f329291197be0d742b4743735c88`.  The authoritative qualified analysis contract is [`benchmark_v1/protocol/qualified_logical_benchmark.json`](../../../benchmark_v1/protocol/qualified_logical_benchmark.json#L2-L50), not the historical 7,350-row materialization.  It declares 4,515 rows: Ma–Li 340, Qonductor 230, and QPack MCP 3,945, and explicitly calls the result a retrospective, reconstruction-qualified grouped-OOF comparison rather than 4,515 exact historical logical instances (lines 2–13).

The immutable inputs used for this audit are [`panel.csv`](../../../artifacts/real_qpu/qualified_logical_panel/panel.csv) (SHA-256 `b4ade914955d7807e466cf5cf0c5265d21b39a81fdcee8e91dc2fb6300eb978a`), [`manifest.json`](../../../artifacts/real_qpu/qualified_logical_panel/manifest.json) (`687c3357cda8523d988641935bd6c5c99d5fed164e85ef6e649bed6e9556d2cb`), and the saved OOF predictions cited below.  No dataset, split, prediction, or metric artifact was modified.

## Verified source routes

| Source | Original target field / canonical conversion | Observation unit | Shots in qualified cohort | Boundary evidence | Status and unknowns |
|---|---|---|---:|---|---|
| Ma–Li | Mean of three archived IBM `Result.time_taken` values; already seconds | Archived circuit/context observation summarized over three measurements | 1,024 for all 340 | [`docs/methodology.md`](../../methodology.md#L151-L155); [`strict_logical_benchmark.json`](../../../benchmark_v1/protocol/strict_logical_benchmark.json#L9) | **VERIFIED_FROM_DOCUMENTATION.** Exact logical QASM is retained. Individual repetitions, job IDs, dates, queue inclusion, compilation inclusion, reset/measurement/repetition-delay accounting and service internals are **UNKNOWN**. |
| Qonductor | Archived IBM one-circuit job `Result.time_taken`; already seconds | One-circuit archived job | 1,000 (116), 4,000 (82), 8,000 (14), 12,000 (12), 16,000 (3), 20,000 (3) | [`docs/methodology.md`](../../methodology.md#L153-L161); source inventory [`docs/data_and_feature_dictionary.md`](../../data_and_feature_dictionary.md#L109-L119) | **VERIFIED_FROM_DOCUMENTATION.** Exact submitted *physical* QASM and job association are available, but original logical circuit is not. Queue/service decomposition, compilation, reset/measurement and delay accounting are **UNKNOWN**. |
| QPack MCP | `circuit_execution_duration_ms / 1000` | One measured QAOA circuit per optimizer evaluation, not workflow total | 4,096 for all 3,945 | [`docs/methodology.md`](../../methodology.md#L154-L155), [`docs/methodology.md`](../../methodology.md#L170-L173), target code [`evaluate_qpack_workflow_secondary.py`](../../../benchmark_v1/scripts/evaluate_qpack_workflow_secondary.py#L34-L59) | **VERIFIED_FROM_CODE / DOCUMENTATION.** Workflow ID governs grouping only. Exact optimized angles, submitted routing/QASM, compilation lifecycle, queue inclusion, reset/measurement/repetition-delay and provider-service decomposition are **UNKNOWN**. |

The canonical target clock is deliberately named `archived_observed_service_execution_time`, in seconds, and is defined as a provider-recorded execution/service boundary—not pulse duration or a local wall clock ([`docs/methodology.md`](../../methodology.md#L142-L149)).  This name is an interoperability label, not evidence that all three sources share an identical internal timing boundary.

## Construction, grouping, and joins

The target-ID construction maps Ma–Li to `ibm_result_time_taken_mean3_seconds` and all other source routes to `ibm_result_time_taken_seconds`; grouping is exact logical hash (Ma–Li), exact submitted hash (Qonductor), and optimizer workflow ID (QPack) ([`build_canonical_eligibility_and_splits_v3.py`](../../../benchmark_v1/scripts/build_canonical_eligibility_and_splits_v3.py#L28-L37), #L124-L141).  The 4,515 panel then uses five outer and four inner source-balanced, target-blind grouped folds ([`qualified_logical_benchmark.json`](../../../benchmark_v1/protocol/qualified_logical_benchmark.json#L15-L19)).

Qonductor’s resource-estimator data and scheduler/JCT material must not be silently joined: the repository map records that the public circuit/job database has no stable join key to the estimator CSV ([`benchmark_v1/PAPER_REPO_EXPERIMENT_MAP_2026-09-21.md`](../../../benchmark_v1/PAPER_REPO_EXPERIMENT_MAP_2026-09-21.md#L27)).  This audit therefore makes **no** scheduler/JCT attribution and inserts no inferred overhead into labels.

## What is, and is not, established

**VERIFIED FACTS.** All labels are seconds; QPack is explicitly converted once from milliseconds; Qonductor records variable shots; and QPack’s workflow is not its timing unit.  The source collection routes differ materially.

**INTERPRETATION.** Comparing a nominal scheduled duration to these service/execution observations is scientifically useful as a cross-clock diagnostic, but it is not a measurement of a single physical overhead term.

**OPEN QUESTIONS.** Whether `Result.time_taken` includes queue time, transpilation, simulator/service initialization, sampling/result extraction, teardown, or retry handling cannot be established on a row-by-row, source-comparable basis from the retained artifacts.  These fields remain **UNKNOWN**, not zero.
