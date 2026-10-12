# Results by research question

The canonical writing layer is [thesis synthesis](../docs/thesis/README.md).
Retain population, clock, support and uncertainty. The two tracks are not one
leaderboard; negative findings remain results.

| RQ | Track | Canonical evidence | Cohort/context | Status |
| --- | --- | --- | --- | --- |
| RQ1 Accuracy | A | [Main accuracy](thesis/tables/b_qpu_main_accuracy.csv), [paired comparison](real_qpu/shared_row_comparisons.csv) | Qualified 4,515; source/group weighting separate | ANSWERED WITH QUALIFICATIONS |
| RQ1 Accuracy | B | [Context table](thesis/tables/d_simulator_contexts.csv), [simulator scores](simulator/method_comparison.csv) | Separate engine/API/precision/quality contexts | ANSWERED WITH QUALIFICATIONS |
| RQ2 Generalization | A | [Cases](generalization/full_grid/case_status.csv), [seeds](generalization/full_grid/seed_metrics.csv), [ensembles](generalization/full_grid/median_metrics.csv), [paired evidence](generalization/full_grid/paired_comparisons.csv) | 9 family, 10 backend-context, 2 size holdouts; case overlaps not pooled | PARTIALLY ANSWERED, NON-BLOCKING |
| RQ2 Generalization | B | [Transfer synthesis](../docs/thesis/rq2_generalization.md), [MPS/context results](thesis/tables/d_simulator_contexts.csv) | Evaluated family/width evidence, not universal engine/hardware transfer | PARTIALLY ANSWERED, NON-BLOCKING |
| RQ3 Failures | A | [Failure taxonomy](thesis/tables/f_failure_modes.csv), [QPU holdouts](thesis/tables/c_qpu_generalization.csv) | QNN/Random, collisions, tails, source sensitivity | ANSWERED WITH QUALIFICATIONS |
| RQ3 Failures | B | [MPS safety](analysis/mps_selection_safety.csv), [interpretation](../docs/thesis/rq3_failures.md) | c44 64/140 per selector; coverage/resource failures separate | ANSWERED WITH QUALIFICATIONS |
| RQ4 Accuracy–cost | A | [Matched summary](accuracy_cost/qpu/accuracy_latency_summary.csv), [paired interpretation](accuracy_cost/qpu/accuracy_cost_tradeoff.md) | Exact45, warm B2, no QPack; not full-cohort overhead | ANSWERED WITHIN MEASURED SCOPE |
| RQ4 Accuracy–cost | B | [Cost table](thesis/tables/e_accuracy_cost.csv), [Dense interpretation](accuracy_cost/dense/accuracy_cost_tradeoff.md) | Exact30 each precision; not full150 or operational break-even | ANSWERED WITHIN MEASURED SCOPE |
| RQ5 Fidelity | A | [Taxonomy](thesis/tables/a_method_taxonomy.csv), [QCEC qualification](reproducibility/qcec_report.md) | Native components/adaptations/proxies; no historical recovery claim | ANSWERED WITH QUALIFICATIONS |
| RQ5 Fidelity | B | [Applicability](../docs/thesis/rq5_fidelity_applicability.md), [input replay](../docs/thesis/reproducibility_findings.md) | Endpoint/quality and environment-aware compatibility | ANSWERED WITH QUALIFICATIONS |

RQ4 is coded ANSWERED WITH QUALIFICATIONS in the [closure CSV](thesis/tables/g_rq_closure.csv).
Current [completion decision](research_questions/closure/experiment_completion_decision.md)
requires no additional experiment. Analytical raw, shot-scaled and calibrated
outputs remain distinct. [Historical packages](../docs/historical_packages.md)
explain old pilot/closure statuses; they are not active queues.
