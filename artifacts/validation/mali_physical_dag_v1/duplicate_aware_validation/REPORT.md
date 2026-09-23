# Ma--Li duplicate-aware fitting

All tests remain original observed rows. Only outer-training rows are collapsed by exact `(QASM hash, backend)` cell; noise weights are calculated from outer-training duplicate spread.

| Evaluation | Method | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|
| grouped-QASM | `cell_mean_equal` | 0.4923 | 0.8917 | 0.8473 |
| grouped-QASM | `cell_mean_noise_weighted` | 0.4948 | 0.8906 | 0.8454 |
| grouped-QASM | `row_weighted` | 0.4937 | 0.8895 | 0.8432 |
| strict backend+QASM | `cell_mean_equal` | 0.5200 | 0.8765 | 0.8325 |
| strict backend+QASM | `cell_mean_noise_weighted` | 0.5156 | 0.8799 | 0.8361 |
| strict backend+QASM | `row_weighted` | 0.5230 | 0.8755 | 0.8306 |

A gain would indicate that repeated labels were distorting training weight. If row-weighted remains best, duplicate handling should be retained as uncertainty evidence rather than a model-selection method.
