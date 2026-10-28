# IBM usage formula: descriptive evaluation on frozen Track A

Declared before calculation on 2026-10-27. No fitting, selection, calibration or new circuit execution.

Population: all 4,515 rows / 165 structural groups in `artifacts/real_qpu/qualified_logical_panel/panel.csv`. Preserve provider `Result.time_taken` labels, seconds, original groups and source aggregation. This is a new diagnostic; frozen scores remain unchanged.

1. `IBM_QUICK`: the IBM-documented quick formula, `2 + 0.00035 * shots`, applied to one circuit and one sub-job per observation, without advanced mitigation. This uses a typical instruction length rather than each circuit's saved duration.
2. `IBM_DURATION_DEFAULT_DELAY`: `2 + shots * (saved_nominal_single_shot_duration_seconds + 0.00025)`. Uses the saved snapshot duration and IBM's typical 250 microsecond delay. Assumes one sub-job; no separate initialization term is added because historical initialization duration/options are unknown. This is an explicitly configured application of the baseline formula, not the remote service estimate or a full replay of historical options.

Ma–Li mean-of-three labels remain means: do not multiply shots or overhead by three. No rounding to whole seconds: evaluate continuous estimates. No parameters are inferred from target labels. Count missing/nonfinite inputs as failures rather than replace with zero. Abort on unexpected row count, group count, source counts, duplicate IDs or mismatched joins.

Primary descriptive metric: group-balanced MAE (equal mean of within-group absolute errors). Also report pooled MAE, source-macro MAE, RMSE, raw R², bias, median/P90/P99/max absolute error, and source slices. Positive bias means overprediction. There is no trained model, so no train/test fitting or OOF selection is needed; all rows are scored using the fixed formula. No confirmatory superiority claim or CI is introduced.

Formula source: https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time . The official service `job.usage_estimation` is a separate server-returned estimate: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-job-v2 .
