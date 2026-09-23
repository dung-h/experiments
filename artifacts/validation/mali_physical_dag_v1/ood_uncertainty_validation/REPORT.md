# Ma--Li support-aware uncertainty validation

Global conformal and a two-band support-stratified variant use the same QASM-disjoint calibration group. The latter is diagnostic because each band is small.

| Evaluation | Method | Nominal | Coverage | Median width (s) | QWalk coverage | QWalk median width (s) |
|---|---|---:|---:|---:|---:|---:|
| grouped-QASM | `global` | 90% | 87.6% | 2.1283 | 0.0% | 2.1888 |
| grouped-QASM | `support_stratified` | 90% | 87.6% | 2.0337 | 0.0% | 2.7141 |
| grouped-QASM | `global` | 80% | 78.8% | 1.4794 | 0.0% | 1.7136 |
| grouped-QASM | `support_stratified` | 80% | 77.6% | 1.4600 | 0.0% | 1.8106 |
| strict backend+QASM | `global` | 90% | 86.5% | 2.1991 | 0.0% | 2.3261 |
| strict backend+QASM | `support_stratified` | 90% | 87.6% | 2.2637 | 0.0% | 2.6024 |
| strict backend+QASM | `global` | 80% | 76.8% | 1.5360 | 0.0% | 1.8450 |
| strict backend+QASM | `support_stratified` | 80% | 78.5% | 1.5857 | 0.0% | 1.9034 |

QWalk is consistently in the high-support-distance band, but neither global nor two-band intervals cover it. Support distance is therefore useful to trigger abstention/additional calibration, not yet sufficient to create a reliable automatic tail interval. It is not a hardware-wide confidence guarantee.
