# Real-QPU results

Assigned dataset: **4,515 observations** (Ma–Li 340; Qonductor 230; QPack 3,945).
All method families share the fresh five-fold outer test assignments.

**Status: fresh-fit execution and final QA pending.** Error scores and paired
intervals are not yet published for this dataset. An empty score is not zero,
unavailable method support or evidence that a method failed.

| Method family | Assigned observations | Required evaluation |
| --- | ---: | --- |
| Ma–Li-style logical graph | 4,515 | Three-seed grouped OOF |
| Matched global MLP | 4,515 | Three-seed grouped OOF |
| Qonductor-style six regressors | 4,515 each | Nested grouped OOF |
| QCRE schedule and calibrated variants | 4,515 each | Explicit coverage and fresh outer-train calibration |
| Qiskit duration and calibrated variants | 4,515 each | Explicit coverage and fresh outer-train calibration |
| Hyb-HANAS-style cost variants | 4,515 each | Explicit coverage and fresh fold-local calibration |
| Scholten-style nominal throughput | 4,515 each | Nominal adaptation; explicit missing support |

See the [dataset report](../../docs/data_preprocessing.md),
[QPU protocol](../../docs/real_qpu_benchmark.md) and
[execution summary](../../protocol/real_qpu_common_panel.json).
A populated numerical CSV will be added only after the complete run is checked.
