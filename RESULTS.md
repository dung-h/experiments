# Results and failure analysis

## Comparative conclusion

The four studies measure different target semantics. Their scores are evidence
about their own estimator/target pairs; they are not entries in one leaderboard.

## Packaging verification (2026-09-17)

The capsule was tested from a fresh temporary work directory rather than from
the existing checkouts. `scripts/verify_artifacts.py` passed; bootstrap checked
out all four upstream revisions at the locked SHAs, applied the Ma–Li/Qonductor
overlays, copied 15 QASM inputs, and materialized all six CDAA duration
snapshots. Qonductor's public-artifact JSON matched the committed fixture
byte-for-byte, and its fake scheduler smoke completed without credentials.

For Ma–Li, a ten-sample fake-snapshot build plus one-epoch DAG and ten-fold
pretrained/from-scratch adaptation smokes completed on CPU. The full 340-row,
500-epoch numbers below remain the committed experiment outputs; the smoke is
only an execution/path check. For CDAA, all 90 Qiskit and 45 TKET proxy files
were generated in the clean work directory, and a one-circuit SQGM/SABRE run
completed under the pinned old-Terra environment. cuTensorNet was not rerun in
this packaging check because it requires the RTX 5070 Ti/CUDA stack; its
25-row GPU measurements remain explicitly hardware-scoped below.

This distinction is intentional: packaging verification proves that a reviewer
can materialize and execute the relevant paths, while the reported scientific
results retain their original provenance and hardware boundaries.

| Track | Primary result | Supported claim | Boundary |
| --- | --- | --- | --- |
| Ma–Li | pretrained R² 0.5922 ± 0.8317; scratch R² 0.8855 ± 0.0461 | A simulator prior can harm local fake-snapshot transfer | Not the historical paper environment |
| Qonductor | regression MAE 502.359 ms, R² 0.9386; DAG MAE 915.609 ms, R² 0.8849 | Supplied regression predictions beat the numerical DAG baseline | Derived recompute, not live IBM/retraining |
| CDAA/QCRE | 30/45 verification rows agree; maximum delta 3.50e-16 s | QCRE matches the supplied schedule-duration reference | Schedule duration is not observed QPU wall-clock |
| cuTensorNet | warm median actual / RUNTIME_EST = 0.196; QFT-28/30 0.960/0.946 | The pre-run signal is conservative on easy contractions and closer on QFT | One GPU/software/precision setting |

## 1. Ma–Li: seed 1234, fold 10

The pretrained and from-scratch controls use the same 340 recorded Osaka/Kyoto
labels, fake-snapshot features, architecture, optimizer, ten folds, 500 epochs,
seed 1234 and split indices. Initialization is the only treatment.

| Fold set | Pretrained R² | From-scratch R² |
| --- | ---: | ---: |
| Folds 1–9 | 0.7616–0.9343 | 0.8035–0.9363 |
| Fold 10 | **-1.8991** | **0.9358** |

The held-out Osaka row is qwalk-noancilla_indep_qiskit_9:

| Quantity | Value |
| --- | ---: |
| Actual Osaka runtime | 13.6962 s |
| Pretrained prediction | 36.0312 s |
| From-scratch prediction | 12.8724 s |
| Pretrained squared-error share of fold 10 | 498.85 / 508.59 = 98.1% |

The complete retrieval record—QASM SHA-256, label rows, fold, seed and
predictions—is [artifacts/mali/fold10_qwalk_outlier.json](artifacts/mali/fold10_qwalk_outlier.json).
The full comparison is
[artifacts/mali/MA_LI_TRANSFER_BASELINE_COMPARISON.md](artifacts/mali/MA_LI_TRANSFER_BASELINE_COMPARISON.md).

The Kyoto copy is in fold-10 training while the Osaka copy is test. Global and
non-calibration graph features are identical; only four T1/T2 calibration
fields differ. Setting those fields to their standardized mean yields 19.402 s
for both copies from the pretrained model, but 13.213 s from scratch. This
indicates that the pretrained weights react strongly to the reconstructed
calibration domain shift.

The likely interacting factors are simulator-to-QPU scale mismatch (the same
circuit's simulator labels are about 42.43 s and 73.24 s), different
normalization populations, and a 34-row held-out fold dominated by one extreme
sample. Without fold 10, mean R² is 0.8690 pretrained and 0.8799 from scratch.
The aggregate gap is therefore largely one localized failure. Historical IBM
calibration snapshots, the authors' processed tensor/order and a matched
multi-seed baseline are not public, so this remains an OUR_PROXY adaptation.

## 2. Qonductor: regression versus numerical DAG

The public 100-row evaluation file contains author-generated predictions. Our
script recomputed its metrics; it did not retrain the estimator.

| Predictor | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Qonductor regression | 502.359 ms | 1059.620 ms | 0.9386 |
| Numerical DAG critical path | 915.609 ms | 1451.502 ms | 0.8849 |

The learned predictor uses five post-transpile/job features: routing or
two-qubit proxy, depth, qubits, shots and circuit count. The DAG comparator is
a calibrated critical-path sum, not another learned model. The failure boundary
is provenance: this does not establish new-backend transfer, live IBM
execution, queue-time prediction, or a newly trained model. Details and
fixtures are under [artifacts/qonductor/](artifacts/qonductor/).

## 3. CDAA/QCRE: schedule agreement versus hardware truth

QCRE reproduces the supplied Qiskit schedule-duration reference in 30 of 45
Eagle verification rows; the largest absolute difference is
3.4998046127832083e-16 s. The artifact's average gate-aware-depth error
reduction is 63.74× versus traditional depth and 17.80× versus multi-qubit
depth. It identifies the shortest estimated compiler result on five of six
devices; Aachen reaches 80%.

The arXiv abstract says 68×, while detailed bundled statistics give 63.74×
(approximately 64×). Both are retained, not silently merged. Offline
Qiskit/TKET runs use fake-backend topology proxies because the original path
queries a live IBM target; SQGM/SABRE use the artifact driver. No result is a
measured QPU runtime. See
[artifacts/cdaa_qcre/REPLICATION_REPORT.md](artifacts/cdaa_qcre/REPLICATION_REPORT.md).

## 4. cuTensorNet: one pre-run estimate, distinct actual clocks

The RTX 5070 Ti experiment creates scalar tensor networks for GHZ, HEA,
QAOA-cycle, random brickwork and QFT at 16–30 qubits in complex64. It reads one
RUNTIME_EST after TIME_TUNED path optimization, then measures the same
contraction by CUDA events after two warm-ups.

Across 25 rows, no warm contraction exceeds the estimate. Median
actual/estimate is 0.196; mean absolute log error is 1.450. QFT is harder and
more accurately estimated: 0.960 at 28 qubits and 0.946 at 30 qubits.

This does not conflict with a slow first call. Tensor construction, path search,
allocation and Python dispatch are outside the warm CUDA-event target. First
contraction, warm contraction and end-to-end latency are separate columns, not
three cuTensorNet estimates. The detailed result is
[artifacts/cutensornet/CUTENSORNET_RUNTIME_ESTIMATOR_INITIAL_RESULTS.md](artifacts/cutensornet/CUTENSORNET_RUNTIME_ESTIMATOR_INITIAL_RESULTS.md).
