# Frozen prediction error decomposition audit

## Data and calculation

All calculations below read only saved OOF rows from [`summary/oof_predictions.csv`](../../../artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv), SHA-256 `c46634505e9d84f83c65f28c3097c6dbb964ee7d3932e85862d2ed4953ea41bc`.  For each predicted row, `error = prediction_seconds - actual_seconds`; MAE is mean absolute error; bias is mean signed error.  Ratios are `prediction / actual` only for positive targets (all retained rows in these summaries).  No calibration, baseline, split, or prediction was rerun or fitted.

The saved method table independently reports the same full-panel raw MAEs: QCRE raw 6.115988 s, QCRE shot-scaled 5.671859 s, Qiskit raw 6.115983 s, and Qiskit shot-scaled 5.654227 s ([`method_metrics.csv`](../../../artifacts/real_qpu/qualified_logical_panel/summary/method_metrics.csv)).

## QCRE saved OOF predictions

| Source | n | QCRE raw MAE / bias (s) | Raw median ratio | Shot-scaled MAE / bias (s) | Scaled median ratio | Affine-calibrated MAE / bias (s) |
|---|---:|---:|---:|---:|---:|---:|
| All | 4,515 | 6.115988 / -6.115988 | 0.00000239 | 5.671859 / -5.671859 | 0.009776 | 1.061287 / +0.014483 |
| Ma–Li | 340 | 8.450184 / -8.450184 | 0.000554 | 3.602003 / -3.602003 | 0.567411 | 0.955486 / +0.242357 |
| Qonductor | 230 | 3.514240 / -3.514240 | 0.000000876 | 3.501525 / -3.501525 | 0.002089 | 2.689499 / +2.386931 |
| QPack | 3,945 | 6.066501 / -6.066501 | 0.00000234 | 5.976783 / -5.976783 | 0.009589 | 0.975478 / -0.143474 |

The raw QCRE signed-error distribution on all 4,515 saved rows has P05/median/P95 `-8.071204 / -6.600809 / -3.626918 s`; the shot-scaled equivalent is `-7.106063 / -6.202472 / -3.150699 s`.  Thus the negative bias is not a small number of outliers.

## Qiskit cross-check

Qiskit repeats the pattern almost exactly: full-panel raw MAE/bias `6.115983 / -6.115983 s`, scaled `5.654227 / -5.654227 s`, and affine-calibrated `1.060995 / +0.014479 s`.  Source-level scaled MAE is Ma–Li `3.600569 s`, Qonductor `3.483978 s`, and QPack `5.957751 s`.  This agreement is expected from the declared frozen-Target equivalence, not independent confirmation of a physical mechanism.

## Coverage and comparison limits

QCRE/Qiskit routes cover 4,515/4,515 rows for all raw, scaled, and affine forms.  Scholten nominal proxy covers 3,399/4,515 only; its raw MAE is 610.288137 s and log-affine calibrated MAE 0.726600 s on that *different covered subset*.  It must not be ranked directly against full-coverage methods.

No saved `outer_train_median` / train-only constant prediction method exists in this 32-method `summary/oof_predictions.csv`; therefore a matched calibrated-versus-median comparison is **UNAVAILABLE** in this audit.  Constructing one would be a new baseline calculation and is prohibited by scope.

## Hypothesis assessment

### H1 — single-shot versus repeated-shot scaling mismatch: **PARTIALLY_SUPPORTED**

**Support.** The frozen code confirms that raw values are single-shot scheduled durations and scaled values multiply by ledger shots.  MAE falls by 4.848181 s on Ma–Li, where every row is 1,024 shots; the median prediction/observed ratio moves from 0.000554 to 0.567411.

**Contradiction/limit.** On Qonductor the MAE changes only 0.012715 s, and its scaled median ratio remains 0.002089.  On QPack the MAE changes only 0.089718 s and its scaled median ratio remains 0.009589.  Therefore a missing shots multiplier explains part of the direction and a large part of the Ma–Li reduction, but cannot explain the magnitude of the cross-source discrepancy.

### H2 — missing execution/service overhead: **PARTIALLY_SUPPORTED**

**Support.** Raw scheduled clocks strongly underpredict the service/execution labels for every source; the method cards independently declare this clock mismatch.  This is compatible with omitted components such as measurement, repetition handling, initialization, or provider service handling.

**Limit.** No artifact measures any named component independently, and the source records do not establish a common inclusion/exclusion policy for queue, compiler, sampling, reset, or teardown.  Hence no residual may be quantitatively assigned to “overhead,” and no causal decomposition is identified.

### H3 — heterogeneous source target semantics: **SUPPORTED**

**Support.** The three documented source routes differ in repetition aggregation (Ma–Li mean-of-three), observation form (Qonductor one-circuit job), native unit (QPack milliseconds), shots (1,024 / variable / 4,096), source circuit evidence, and grouping.  The protocol explicitly warns that same-row controls do not harmonize source-specific collection boundaries ([`docs/data_and_feature_dictionary.md`](../../data_and_feature_dictionary.md#L146-L155)).

**Limit.** Heterogeneity does not by itself prove it caused every source difference; nominal snapshot mismatch and reconstructed analytical inputs are also live alternatives.
