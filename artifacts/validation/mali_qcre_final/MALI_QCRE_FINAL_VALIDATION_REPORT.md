# Ma–Li / QCRE final validation pass

This report validates the committed `OUR_PROXY` feature artifact; it does
not claim historical calibration recovery or a universal QPU estimator.

Rows: **340**; unique logical QASM hashes: **170**; grouped folds: **5**, seed **1234**.

## Split and duration integrity

Repeated Osaka/Kyoto copies are grouped by SHA-256 (`134` duplicate logical groups), so no logical circuit appears in both train and test.
Duration coverage: **complete**; unsupported operation rows: **0**; zero-duration rows: **0**.
Policy: native target durations are used for gates, measurement and reset; barrier/directive instructions are skipped; delay uses its own duration; unknown operations are listed in qcre_unsupported_duration_ops

## Grouped out-of-sample single-feature calibration

All rows below use the same grouped folds and fit log1p(target) only on
the training groups of each fold.

| Feature | MAE (s) | MedAE (s) | RMSE (s) | R² | log-R² |
|---|---:|---:|---:|---:|---:|
| `logical_depth` | 1.4867 | 1.1399 | 1.9410 | -0.0346 | -0.0281 |
| `logical_two_qubit_depth` | 1.4105 | 1.0923 | 1.8353 | 0.0750 | 0.0370 |
| `physical_depth` | 0.6763 | 0.5344 | 0.8788 | 0.7879 | 0.8241 |
| `physical_two_qubit_depth` | 0.7062 | 0.5657 | 0.9090 | 0.7731 | 0.8121 |
| `qcre_weighted_critical_path_seconds` | 0.8276 | 0.6594 | 1.0976 | 0.6692 | 0.7321 |

Grouped median baseline: MAE 1.4342 s, MedAE 1.1489 s, R² -0.0122.

## Incremental ablation (same grouped folds)

| Feature set | MAE (s) | MedAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|---:|
| `physical_depth` | 0.6763 | 0.5344 | 0.8788 | 0.7879 |
| `physical_depth_plus_physical_2q_depth` | 0.6712 | 0.5386 | 0.8732 | 0.7906 |
| `physical_depth_plus_weighted_path` | 0.6850 | 0.5652 | 0.8724 | 0.7910 |
| `physical_depth_plus_physical_2q_depth_plus_weighted_path` | 0.6778 | 0.5770 | 0.8649 | 0.7946 |

## Shots-scaled affine calibration

The upstream execution helper uses `shots=1024`; this test fits only
the training groups of each fold:
`T_obs = alpha + beta * (1024 * weighted_path)`.

MAE 0.7209 s; MedAE 0.5659 s; RMSE 0.9473 s; R² 0.7536.

## Per-backend grouped out-of-sample result

The weighted-path prediction below is produced by the same grouped
calibration, then sliced by backend for diagnostics.

| Backend | n | Target median (s) | Prediction median (s) | MAE (s) | MedAE (s) | R² |
|---|---:|---:|---:|---:|---:|---:|
| `kyoto` | 148 | 8.7236 | 8.4658 | 0.8170 | 0.6594 | 0.6697 |
| `osaka` | 192 | 8.5839 | 8.3541 | 0.8358 | 0.6594 | 0.6666 |

## Leave-one-backend-out: the two transfer directions

These are the two domain-transfer directions, each trained on one
backend and evaluated on the other. They are not five iid folds.

| Train backend | Held-out backend | n | MAE (s) | RMSE (s) | R² |
|---|---|---:|---:|---:|---:|
| `osaka` | `kyoto` | 148 | 0.8132 | 1.0757 | 0.6705 |
| `kyoto` | `osaka` | 192 | 0.8201 | 1.0911 | 0.6798 |

## Reading the result

The grouped scores are the defensible numbers for this artifact. If
weighted duration fails to improve the physical-depth baseline, the
claim should remain that compiled physical structure carries the
transferable signal; duration is then a calibrated proxy rather than
an independently validated hardware clock.
