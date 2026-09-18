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
only an execution/path check. The separate 340-row Ma–Li/QCRE proxy was also
run locally with the current FakeOsaka/FakeKyoto target at transpile level 1;
its output is marked `OUR_PROXY` and does not replace the historical Ma–Li
adaptation result. For CDAA, all 90 Qiskit and 45 TKET proxy files
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
The aggregate gap is therefore largely one localized failure. The authors'
processed tensor/order and the exact IBM calibration snapshot used for those
340 labels are not identified or bundled in the public package. Public
candidate calibration captures exist, but none has yet been matched to the
historical tensor or collection timestamp, so this remains an OUR_PROXY
adaptation.

The detailed report-style interpretation of the QWalk circuit structure, the
178-dimensional DAG metadata, the T1/T2 intervention and the separate QCRE
proxy inference path is preserved in
[artifacts/mali/MA_LI_QWALK_AND_QCRE_INTERPRETATION.md](artifacts/mali/MA_LI_QWALK_AND_QCRE_INTERPRETATION.md).
Use that note when presenting why the pretrained model fails on QWalk and why
compiled physical depth is stronger than logical depth in the proxy audit.

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

### 2.1 Qonductor mapping audit

The additional audit confirms why the public Qonductor resource-estimator
numbers cannot currently be joined to individual circuits. The CSV has 100
rows and only `predicted`, `real`, and `dag`; the SQLite database has 7,449
jobs, 166,093 circuits and 17 backend labels. There is no shared
`job_id`/`circuit_id`/`ibm_quantum_id`. Matching by row order or nearest runtime
would therefore be fabricated. The audit and its machine-readable evidence are
[artifacts/validation/qonductor_mapping/QONDUCTOR_MAPPING_AUDIT.md](artifacts/validation/qonductor_mapping/QONDUCTOR_MAPPING_AUDIT.md).

## 2.2 Ma–Li / QCRE gate-aware proxy validation

To test whether compiled, gate-aware structure explains the Ma–Li labels, we
transpiled all 340 Osaka/Kyoto QASM circuits with Qiskit 1.4.1 `FakeOsaka` or
`FakeKyoto` (seed 1234, optimization level 1). Instead of Qiskit's ALAP pass,
which fails on some large inputs in this environment, the script reconstructs
a weighted critical path from the current FakeBackend target durations. This
is a current-target `OUR_PROXY`, not the historical physical circuit or
calibration snapshot used to create the labels.

| Feature | log Pearson | log Spearman | 5-fold calibrated R² |
| --- | ---: | ---: | ---: |
| Logical depth | 0.0610 | 0.2354 | -0.0188 |
| Logical two-qubit depth | 0.2355 | 0.3155 | 0.0870 |
| Physical depth after transpile | 0.9096 | 0.8175 | 0.7847 |
| Physical two-qubit depth | 0.9030 | 0.7975 | 0.7697 |
| Weighted target-duration path | 0.8607 | 0.8000 | 0.6667 |

The same one-feature calibration held out an entire backend at a time: the
weighted path reached R² 0.6705 on Kyoto and 0.6798 on Osaka. The raw scale is
not a hardware runtime: the median observed label is 8.637 s while the median
weighted path is 0.00510 s (about 1,708× smaller). Thus the useful result is
ranking/structure plus an explicitly fitted calibration, not direct equality.
All row-level features, paths, correlations and held-out scores are in
[artifacts/validation/mali_qcre_proxy/](artifacts/validation/mali_qcre_proxy/).

### 2.3 Final leakage, duration and ablation checks

The validation pass groups by logical-QASM SHA-256 before splitting. The 340
rows reduce to 170 logical circuits; 134 hashes occur on both devices, so an
Osaka/Kyoto copy cannot land in different folds. Grouped five-fold calibration
gives physical-depth R² `0.7879` (MAE `0.6763 s`, MedAE `0.5344 s`) and weighted
path R² `0.6692` (MAE `0.8276 s`, MedAE `0.6594 s`). The grouped median baseline
has R² `-0.0122` and MAE `1.4342 s`.

The incremental ablation uses exactly those folds:

| Feature set | R² | MAE (s) | MedAE (s) |
| --- | ---: | ---: | ---: |
| Physical depth | 0.7879 | 0.6763 | 0.5344 |
| + physical 2-qubit depth | 0.7906 | 0.6712 | 0.5386 |
| + weighted path | 0.7910 | 0.6850 | 0.5652 |
| + physical 2-qubit depth + weighted path | 0.7946 | 0.6778 | 0.5770 |

Thus the timing-aware path adds only a small incremental R² (`~0.003`) over
compiled structural features. The defensible claim is that physical compiled
structure carries most of the transferable signal; the duration proxy is
useful for calibration but is not independently dominant. Multiplying the path
by the upstream `shots=1024` and fitting an out-of-sample affine map gives R²
`0.7536` (MAE `0.7209 s`), with fold intercepts around 3.1–3.3 s and slopes
near 1.0. Duration coverage is complete: 0 unsupported operations, negative
or non-finite durations, or zero-duration rows; barriers are skipped and
measurement/reset/delay handling is explicit in the script.

The transpiler-seed sensitivity reran all 340 rows at seeds 1234, 2025 and
31415 under the same Qiskit 1.4.1/FakeBackend/optimization-level-1 protocol and
the same grouped folds:

| Feature | Mean R² | Std | Range |
| --- | ---: | ---: | ---: |
| Physical depth | 0.7847 | 0.0059 | 0.7764–0.7897 |
| Physical 2-qubit depth | 0.7720 | 0.0045 | 0.7661–0.7769 |
| Weighted path | 0.6628 | 0.0138 | 0.6437–0.6756 |

The seed range is narrow enough to retain a central estimate, while still being
reported rather than hidden. Full fold details and row-level seed outputs are
in [artifacts/validation/mali_qcre_seed_sensitivity/](artifacts/validation/mali_qcre_seed_sensitivity/);
the grouped validation is in
[artifacts/validation/mali_qcre_final/](artifacts/validation/mali_qcre_final/).
The stricter two-direction transfer gives R² `0.6705` for Osaka → Kyoto and
`0.6798` for Kyoto → Osaka; these are kept separate from the grouped
five-fold diagnostics.

### 2.4 Public calibration snapshot variability

**Finding recorded:** 2026-09-18.

The snapshot audit compares the frozen Qiskit fake-provider pair against the
public DAQEC-Benchmark `drift_characterization.csv`. The latter contains 42
timestamp snapshots per backend over 14 days (2025-01-15 through 2025-01-28),
with three replicate rows per timestamp. Timestamp-level aggregate variation
is material:

| Backend | T1 range / median | T1 CV | T2 range / median | T2 CV |
|---|---:|---:|---:|---:|
| Osaka | 63.6% | 17.3% | 50.3% | 12.9% |
| Kyoto | 53.0% | 16.8% | 48.6% | 12.8% |

The variation is non-monotonic; fitted slopes are only descriptive. This
supports treating calibration time as a domain variable and running
multi-snapshot sensitivity. It does not identify the exact historical
per-qubit snapshot used by Ma–Li: the DAQEC file contains aggregate means, not
the 127-qubit node-level tensor. The reproducible audit and timestamp rows are
in [artifacts/validation/mali_snapshot_variability/](artifacts/validation/mali_snapshot_variability/).
The raw input and exact download/hash/rerun instructions are in
[artifacts/validation/mali_snapshot_variability/DATASET_README.md](artifacts/validation/mali_snapshot_variability/DATASET_README.md).

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
