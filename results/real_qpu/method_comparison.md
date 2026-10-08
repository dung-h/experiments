# Qualified real-QPU benchmark results

Status: completed and independently checked against saved evidence.

This comparison assigns the same 4,515 archived observations to every method under fresh grouped outer folds. It contains 32 variants, not 32 paper methods. Simulator results remain a separate engine/clock comparison; no simulator measurement was changed by this run.

## Data and protocol

| Source | Included observations | Conservative groups | Input qualification |
| --- | ---: | ---: | --- |
| Ma–Li | 340 | 148 | Exact source logical QASM; historical compiled layout not recovered |
| Qonductor | 230 | 11 | Archive-supported MQT recipe; exact submitted physical QASM, not proven original logical DAG |
| QPack MCP | 3,945 | 6 | Six source-backed QAOA structures; historical optimizer angles and submitted routing absent |

The input filter is independent of runtime and prediction error: Ma–Li 340 is retained, Qonductor 4,482 is reduced to 230 by archive-supported recipe eligibility, and QPack MCP 3,945 retains its explicit structural qualification. The 4,252 excluded Qonductor rows comprise 2,835 unverified recipe candidates and 1,417 rows without qualified logical extraction. Reconstruction supplies inputs, never new runtime labels.

All methods share 165 conservative circuit/workflow/model-input groups and five outer/four inner folds. Neural fits use inner folds 1–3 and inner fold 0 for epoch selection; regressors tune across all four inner folds and refit outer-train. Preprocessing is fitted only inside the corresponding training partition. These are not backend-held-out or family-held-out splits.

Graph and matched MLP use the upstream 51-global/178-node schema, three CUDA seeds (42, 1234, 31415), 500 epochs per cell and per-observation median predictions. The saved fit-only mask retains 40 globals in each fold; it is not an ad hoc seven-feature model. They do not receive shots or backend/source IDs. Qonductor-style regressors preserve the upstream CX/CZ/ECR counter named `swap`, depth, active width, shots and circuit_count=1; their bounded inner search is disclosed rather than called exhaustive reproduction.

The evaluation target is archived provider execution/service time. Ma–Li averages three `Result.time_taken` measurements at 1,024 shots; Qonductor supplies one-circuit job labels with recorded shots; QPack supplies one-circuit duration in milliseconds converted once to seconds. Nominal snapshots match backend names, not proven job-day calibration. Method output clocks are separate columns in the result table.

## Learned methods on the same 4,515 test observations

| Method | Coverage | MAE (s) | R² | Source-balanced MAE (s) | p99 error (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ma–Li-style logical graph + global features | 100% | 0.834 | 0.468 | 0.898 | 2.224 |
| Matched global-feature MLP | 100% | 0.893 | 0.363 | 0.976 | 2.597 |
| Qonductor-style inner-selected regressor family | 100% | 0.887 | 0.411 | 0.879 | 2.690 |
| Qonductor-style Extra Trees | 100% | 0.966 | 0.362 | 1.095 | 2.724 |
| Qonductor-style Random Forest | 100% | 1.027 | 0.102 | 1.012 | 3.562 |
| Qonductor-style Gradient Boosting | 100% | 1.017 | 0.070 | 1.008 | 3.719 |
| Qonductor-style AdaBoost | 100% | 1.034 | 0.270 | 1.081 | 2.826 |
| Qonductor-style Histogram Gradient Boosting | 100% | 0.972 | 0.324 | 0.907 | 2.746 |
| Qonductor-style Polynomial Regression | 100% | 0.918 | 0.402 | 1.008 | 2.477 |

## Analytical variants and coverage

Raw scheduled duration, effective cost and throughput estimates are evaluated against observed service time but keep their original meaning. Affine/log-affine calibration is an added model fitted on successful outer-train rows only; it is not a claim about a physical launch-overhead coefficient. All raw and calibrated variants are retained, with no test-driven choice of the best variant.

| Method family | Successful / assigned | What remains unavailable |
| --- | ---: | --- |
| QCRE source unitary duration | 4,515 / 4,515 | None; terminal measurement is explicitly omitted |
| Qiskit nominal schedule | 4,515 / 4,515 | None; measurement follows Qiskit's duration convention |
| Hyb-HANAS-style raw r2 cost | 4,175 / 4,515 | 148 Kyoto zero-survival cases and 192 Osaka exponential overflows |
| Hyb-HANAS-style log-cost Ridge | 4,367 / 4,515 | 148 Kyoto zero-survival cases; log representation can retain the 192 Osaka cases |
| Scholten-style nominal ordinary-depth throughput | 3,399 / 4,515 | 1,116 rows lack eligible throughput input; snapshot gate data does not establish CLOPS |
| Scholten original paper | 0 / 4,515 | Original kernel n / template D required for effective depth are not established |

Missing values and failures are never imputed as zero. Polynomial regression has three negative predictions, retained in raw-seconds metrics; only the declared log-error calculation projects them to zero. This is a limitation, not a post-hoc reason to repair predictions.

## Findings and limits

- MLP minus graph MAE: +0.059 s; 95% group-bootstrap interval [-0.485, +0.329] s. The interval includes zero.
- Graph minus inner-selected regressor MAE: -0.053 s; 95% group-bootstrap interval [-0.318, +0.458] s. The interval includes zero.
- Graph has the lowest pooled learned-method MAE point estimate, but these intervals do not establish a robust advantage over the matched MLP or inner-selected regressors. The selector has slightly lower source-balanced MAE.
- QPack accounts for 87.4% of observations but only six conservative groups. Repeated contexts do not provide 3,945 independent circuit structures. Report pooled and source-balanced results together.
- Lower partial-coverage MAE is not a full-cohort win. Use each method pair's identical successful test rows; the pair table reports both MAEs/R² and its row/group count.
- Scheduled timing and observed service labels have different boundaries. Calibration improvement shows a learnable mapping in this cohort, not equivalence of the clocks or validation of historical snapshot accuracy.
- This run does not isolate individual feature effects, prove historical logical-instance recovery for reconstructed inputs, or establish backend/family OOD performance. Such claims require separate evidence.

## Files and reproduction

- [method_comparison.csv](method_comparison.csv): all 32 variants, fidelity/claim boundary, separate clocks, coverage and error metrics.
- [shared_row_comparisons.csv](shared_row_comparisons.csv): all 465 pairs, both methods' scores on identical successful test rows, observed differences and saved grouped-bootstrap intervals.
- [dataset_profile.json](dataset_profile.json): per-source width/depth/gates/shots/runtime distributions, reconstruction counts and saved feature masks.
- [seed_metrics.csv](seed_metrics.csv): every neural seed, pooled and by source; no best-seed selection.
- [failure_counts.csv](failure_counts.csv): method/status/reason/source/backend counts.
- [source_and_backend_metrics.csv](source_and_backend_metrics.csv): source, backend and reconstruction-tier slices of the same unified fits.
- [Result provenance](../../provenance/real_qpu.json): source hashes and scope of the independent local validation.

The original per-observation outputs, checkpoints and frozen receipts remain
outside this public checkout. Tables above are aggregate exports from their
validated reporting supplement, not a new experiment.

From this checkout:

```bash
python3 -B scripts/check_qpu_results.py
```

This checks coverage, aggregate accounting, identical-row comparisons and
imported source hashes. It cannot independently recalculate MAE from excluded
targets and predictions. The actual producing runner, analytical executor and
finalizer are retained under `methods/real_qpu/`; their recorded input paths
require the external/local evidence described in the
[reproduction guide](../../docs/reproduction.md). No source refit or simulator
measurement is part of a publication check.
