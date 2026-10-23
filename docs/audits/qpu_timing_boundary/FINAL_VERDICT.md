# Final verdict — QPU runtime target-boundary audit

## Verified

1. The canonical qualified comparison is a 4,515-row, five-fold grouped OOF panel with original source labels retained; it is not a homogeneous single-clock experiment.
2. Raw QCRE and Qiskit are nominal **single-shot scheduled-duration** estimators.  They are almost numerically identical under the frozen Target and both strongly underpredict archived provider execution/service labels.
3. Explicit multiplication by shots improves Ma–Li substantially but does not close the Qonductor or QPack discrepancy.  H1 is therefore **PARTIALLY_SUPPORTED**, not a complete explanation.
4. Calibration is fold-safe but label-trained.  Its large MAE reduction establishes a useful learned bridge on this panel, not correctness of a raw analytical runtime formula.
5. The source target construction and lifecycle evidence differ in documented ways.  H3 is **SUPPORTED**.

## Interpretation

H2 is **PARTIALLY_SUPPORTED** only in the limited sense that a scheduled critical path omits one or more contributors represented in the archived service/execution label.  The artifacts do not identify measurement, reset, shot repetition, service handling, initialization, compilation, queueing, or teardown separately; none may be called the measured dominant overhead.

The thesis conclusion should therefore be qualified as follows:

> On the qualified 4,515-row archived-QPU panel, raw QCRE/Qiskit nominal scheduled durations are cross-clock diagnostics rather than observed-service-time predictors.  Shot multiplication reduces error for the fixed-1,024-shot Ma–Li stratum but is insufficient for the variable-shot Qonductor and fixed-4,096-shot QPack strata.  Outer-train calibration yields a substantially more useful predictive bridge, while retaining the limitation that it learns the archived source-specific execution/service label rather than validating a physical duration formula.

## What remains unknown

- A row-level decomposition of observed service time into gate execution, measurement/reset, repetition delay, initialization, compilation, result extraction, queue, and teardown.
- Exact job-day instruction-duration/calibration state for every archived source observation.
- A stable Qonductor join from resource-estimator CSV to scheduler/JCT traces.
- A matched saved train-only median/constant baseline for the analytical calibrated subset.

## Required thesis qualification

Current conclusions need **wording qualification, not a metric rewrite**: never describe raw QCRE/Qiskit as failed physical gate-duration formulas for a single shared QPU runtime, and never describe affine calibration as recovering a measured overhead.  Preserve source-stratified results and call the calibrated variants learned, fold-safe adaptations.

## Audit validation

- No models trained, tuned, or rerun.
- No new QPU/simulator measurement was executed.
- No canonical data, target, split, metric, or frozen artifact was modified.
- Only this audit directory was created.
- Repository-level `git diff --check` was run after creation and passed.
