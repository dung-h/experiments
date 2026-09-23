# Ma--Li label noise and uncertainty calibration

The duplicate audit groups rows by exact raw-QASM SHA-256 and backend.
It finds 40 duplicated cells (80 rows).
The duplicate spread is a lower-bound warning for prediction precision:
median absolute difference 0.4171 s,
P90 1.2775 s,
maximum 1.8158 s.

Intervals use split-conformal calibration in log1p(seconds). The calibration
groups are disjoint from both the fit and test QASM groups. Coverage is an
empirical property of this dataset/proxy protocol, not an IBM hardware-wide
confidence guarantee.

| Evaluation | Target coverage | Empirical coverage | Median width (s) | P90 width (s) |
|---|---:|---:|---:|---:|
| Grouped-QASM | 90% | 87.6% | 2.1283 | 2.6605 |
| Grouped-QASM | 80% | 78.8% | 1.4794 | 1.9709 |
| Strict backend+QASM | 90% | 86.5% | 2.1991 | 2.7912 |
| Strict backend+QASM | 80% | 76.8% | 1.5360 | 2.2369 |

The interval artifact is intended for a runtime-estimator deployment gate: return a point estimate plus an uncertainty band, and flag QWalk-like support gaps rather than silently presenting a precise scalar.
