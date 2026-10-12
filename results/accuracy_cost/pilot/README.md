> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 controlled overhead pilot

**PARTIAL SUCCESS.** Frozen inference replay and B1 prediction measurements completed for the saved subset. No train/refit or quantum execution. Full B2 from original QASM is not claimed.

Read [replay](replay_validation.md), [timings](timing_summary.md), [accuracy–cost](accuracy_cost_analysis.md), [exclusions](failures_and_exclusions.md) and [decision](scientific_decision.md). Predeclaration: selected_variants.csv, selected_workloads.csv, selection_manifest.json. Native context exceptions are recorded, not hidden.

Raw estimator seconds are derived from integer perf_counter_ns intervals; predicted_seconds in timing records is the estimator output, not measured latency. Graph/MLP denotes median-of-three frozen seeds. Pilot is neither a complete estimator leaderboard nor production latency certification. All saved accuracy values remain original.

Selected workloads: 140 = QPU50 + dense FP32/FP64 30 each + Maestro30. Counts refer to context requests, not unique circuits across tracks. Narrow sampling does not cover every family/backend stratum; uncovered strata are explicit in the manifest. Runtime-estimator contexts are not comparable across targets.

Completed-worker RSS high-water 1.3469 GiB and CUDA reserved peak 0.2012 GiB, each below 8 GiB. Failed manual native parse peak RSS was not measured. Storage and wall totals in execution/validation receipts. No commit/push or presentation edit.
