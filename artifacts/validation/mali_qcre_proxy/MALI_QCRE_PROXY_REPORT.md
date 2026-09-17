# Ma–Li / QCRE gate-aware proxy validation

This is an `OUR_PROXY` analysis, not exact historical hardware validation.
The observed target is Ma–Li's device-labelled `time_taken`; the circuit
is transpiled with the current FakeOsaka/FakeKyoto target and a
backend-target weighted critical path is used as the gate-aware proxy.

Rows: **340**; seed: **1234**; transpile level: **1**.

## Overall log-scale association and five-fold calibration

| Feature | Pearson | Spearman | MAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|---:|---:|
| `logical_depth` | 0.0610 | 0.2354 | 1.4841 | 1.9261 | -0.0188 |
| `logical_two_qubit_depth` | 0.2355 | 0.3155 | 1.3941 | 1.8234 | 0.0870 |
| `physical_depth` | 0.9096 | 0.8175 | 0.6793 | 0.8854 | 0.7847 |
| `physical_two_qubit_depth` | 0.9030 | 0.7975 | 0.7107 | 0.9158 | 0.7697 |
| `qcre_weighted_critical_path_seconds` | 0.8607 | 0.8000 | 0.8251 | 1.1017 | 0.6667 |

## Leave-one-backend-out calibration

Each row below trains the one-feature log calibration on the other
backend and scores the held-out backend. This is a domain-transfer
diagnostic, not a claim that two backends are iid.

| Feature | Held-out backend | n | MAE (s) | RMSE (s) | R² |
|---|---|---:|---:|---:|---:|
| `logical_depth` | kyoto | 148 | 1.4819 | 1.9111 | -0.0402 |
| `logical_depth` | osaka | 192 | 1.4685 | 1.9113 | 0.0173 |
| `logical_two_qubit_depth` | kyoto | 148 | 1.4046 | 1.8175 | 0.0593 |
| `logical_two_qubit_depth` | osaka | 192 | 1.3859 | 1.8233 | 0.1057 |
| `physical_depth` | kyoto | 148 | 0.6640 | 0.8824 | 0.7782 |
| `physical_depth` | osaka | 192 | 0.6862 | 0.8761 | 0.7935 |
| `physical_two_qubit_depth` | kyoto | 148 | 0.6792 | 0.9065 | 0.7660 |
| `physical_two_qubit_depth` | osaka | 192 | 0.7254 | 0.9094 | 0.7775 |
| `qcre_weighted_critical_path_seconds` | kyoto | 148 | 0.8132 | 1.0757 | 0.6705 |
| `qcre_weighted_critical_path_seconds` | osaka | 192 | 0.8201 | 1.0911 | 0.6798 |

## Interpretation

The table compares association and a held-out one-dimensional log
calibration. It does not establish that the reconstructed physical
circuit or calibration snapshot equals the one used for the original
Ma–Li runtime label. A positive result supports transfer of a timing
proxy; it does not support a universal runtime estimator or compiler
selection claim.

Per-backend coefficients are in `mali_qcre_proxy_summary.json`; every
row-level feature and calibrated prediction is in the CSV beside this
report. The same JSON records leave-one-backend-out calibration; that
is the relevant domain-transfer check and is not an iid random split.
