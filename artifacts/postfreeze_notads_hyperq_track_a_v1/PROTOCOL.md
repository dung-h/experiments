# NoTaDS / HyperQ fixed analytical adaptations on Track A

Declared 2026-10-27 before evaluating any new method outcome. This is a separate post-freeze application, not a replacement of published/frozen scores.

## Fixed population and inputs

All 4,515 canonical observations, 165 structural groups, original five outer folds, shots and provider Result.time_taken targets in seconds. Source counts 340 Ma–Li, 230 Qonductor, 3,945 QPack. Reuse the previously saved native-operation DAGs, exact graph hash manifest and backend-matched frozen snapshot registry. No retranspilation, circuits executed, simulator/QPU jobs or neural/tree model training. One CPU process, graph-at-a-time, 8 GiB RSS stop guard; no GPU worker.

Use the same saved compiled/native representation for all new structural scores so the paper/code comparison is not confounded by different circuit representations. This is an adaptation of NoTaDS to the saved uncut native circuits, not execution of its circuit-cutting scheduler. Ma–Li/QPack physical views are frozen reconstructions, Qonductor has exact submitted physical views. All timing data is nominal snapshot timing, not execution-day timing.

## Locked estimators

1. `NOTADS_PAPER`: remove measurement and barrier/snapshot from the timing view; all retained native operations must have arity 1 or 2. Assign ASAP unweighted dependency levels. Score kappa1+10*kappa2, where kappa2 counts levels with any 2Q operation and kappa1 counts remaining levels.
2. `NOTADS_CODE`: same unitary operation sequence; `depth(filter_1Q)+10*depth(filter_2Q)`. Noncounted operations still synchronize dependency wires, matching installed Qiskit depth semantics. This transcribes NoTODS.py at commit ea1303e545bb4c94bcbda1020a6d845ada6a9506; source/paper discrepancy remains explicit.
3. `HYPERQ_PAPER`: duration-weighted longest dependency path on the saved native operations, including terminal measurements, using per-operation/per-qubit snapshot Target durations. Barrier/snapshot are omitted as in the saved graph. Any unsupported or missing duration fails explicitly; do not impute from runtime. No reset/delay appears in the selected native-operation histograms; if encountered, abort and review. This reimplements the paper (OSDI 2025 sections 3.3–3.4), not the public depth-based scheduler.

Retain HyperQ unitary-only duration as an implementation check against existing saved Qiskit nominal single-shot values. The full-measurement version is not assumed identical to that existing boundary. Preserve all measurement operations, including repeats; do not deduplicate.

## Raw and adapted outputs

NoTaDS scores are in arbitrary time units: save raw score and shots*score, but do not compute a seconds MAE on unconverted scores. Do not invent a gate duration of one second or an arbitrary 60ns scale.

HyperQ raw path is in seconds. Report raw and shots*path separately. Neither includes repetition delay or service/sub-job overhead. Do not call multiplication by shots an explicit original-paper job-time formula; it is a declared benchmark adaptation.

For all three scalar routes, evaluate both raw-input and shots-scaled-input **outer-train affine calibration**, matching the frozen analytical calibration procedure: OLS `target_seconds ~ 1 + scalar`, output max(0,a+b*scalar). Exactly two coefficients per fold; no HPO, log-affine alternatives, nonlinear model or model selection. Fit only outside the current held-out fold; each observation gets one held-out prediction. No full-data fit is scored. Calibrated models are learned adaptations, not raw analytical predictions. Save coefficients/train-test ID hashes and disjoint-group receipts. Preserve mean-of-three Ma–Li labels; do not multiply shots by three.

## Scoring and uncertainty

Primary descriptive metric: group-balanced MAE. Also pooled/source-macro MAE, RMSE, raw R², bias, median/P90/P99/max absolute error, source and fold slices. Do not choose a winning variant after scoring.

Predeclared paired comparisons (negative delta favors first method):

- NoTaDS paper shots-affine minus NoTaDS code shots-affine (same representation/calibration, paper/code sensitivity).
- HyperQ shots-affine minus NoTaDS code shots-affine (calibrated routes, not pure topology attribution).
- HyperQ shots-scaled minus existing Qiskit shots-scaled duration (measurement-boundary diagnostic).

Use 10,000 paired whole-structural-group bootstrap draws, seed 42; group-balanced differences. Descriptive 95% percentile intervals; no confirmatory promotion/multiplicity-adjusted discovery claims.

## Integrity and output

Abort on cohort/group/fold/source mismatch, duplicate IDs, nonpositive/invalid targets/shots, changed graph/snapshot hashes, nonfinite predictions or unsupported gates. Cache features by saved graph plus snapshot identity, not row ID. Feature extraction must not use target values. Save feature hashes before scoring. Validate the fast resource-path kernel independently with explicit DAG traversal on toy cases and saved graph canaries, compare filtered depths against Qiskit, replay calibration coefficients and metrics from saved outputs. Preserve all frozen artifacts and valid tail rows. No commit or push.

Upstream claim/code audit: `docs/audits/notads_hyperq_estimator_20261027/REPORT.md`.
