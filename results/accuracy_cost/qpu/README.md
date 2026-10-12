> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 Phase 4 matched QPU accuracy–cost

SUCCESS. Gates 0–3 passed: Graph and MLP replay 45/45, with 2,700 matched full-B2 requests (45 × 2 × 30), no feature cache and explicit CPU-to-CUDA transport.

| method | n | mae_seconds | rmse_seconds | r2 | median_request_median_B2_ms |
|---|---|---|---|---|---|
| graph | 45 | 0.724176697021094 | 0.9475399431446588 | 0.8976867320714272 | 332.557293 |
| mlp | 45 | 0.8125209634576435 | 1.2575724544385458 | 0.819780020637797 | 149.5139715 |


The paired interval crosses zero; no stable accuracy superiority is established on these45 IDs.

Read matched_b2_comparison.md, matched_accuracy_analysis.md, accuracy_cost_tradeoff.md and validation_report.md. Exact inputs, models, protocol and source pins are in predeclaration.json. Fresh evidence: replay_validation.csv, raw_b2_timings.csv, paired_observation_results.csv and stage_timing_summary.csv. Interpretation is restricted to this source-biased 45-ID subset and local bridge environment. No training, refit, quantum execution, dependency change, leaderboard change or commit/push.
