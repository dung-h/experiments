# Results and failure analysis

## Comparative conclusion

The five studies measure different target semantics. Their scores are evidence
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
| Ma–Li | direct compiled Ridge: strict backend+QASM-held-out log R² 0.8742, MAE 0.5224 s; adaptation: pretrained R² 0.5922 ± 0.8317, scratch R² 0.8855 ± 0.0461 | Compiled structure supports a source-specific proxy estimator; a simulator prior can still harm fake-snapshot transfer | Current FakeBackend proxy, not the historical paper environment |
| Qonductor | regression MAE 502.359 ms, R² 0.9386; DAG MAE 915.609 ms, R² 0.8849 | Supplied regression predictions beat the numerical DAG baseline | Derived recompute, not live IBM/retraining |
| CDAA/QCRE | 30/45 verification rows agree; maximum delta 3.50e-16 s | QCRE matches the supplied schedule-duration reference | Schedule duration is not observed QPU wall-clock |
| cuTensorNet | warm median actual / RUNTIME_EST = 0.196; QFT-28/30 0.960/0.946 | The pre-run signal is conservative on easy contractions and closer on QFT | One GPU/software/precision setting |
| Azizov et al. independent | 1,192/1,192 screened local rows successful; 149/1,402 circuits eligible | Current FakeWashingtonV2/FakeSherbrooke + Aer pipeline is executable on a documented local subset | Conservative q≤9/ops/depth screen; not the paper's full HPC dataset or exact artifact |

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

### 2.5 Direct source-specific compiled estimator

**Finding recorded:** 2026-09-19.

The preceding proxy work evaluated one-feature calibrations. The direct-estimator
pass asks the more useful operational question: given a logical QASM circuit and
an intended Osaka/Kyoto target, does a model trained only on the Ma–Li observed
`result.time_taken` labels predict better than physical depth alone? It uses the
same 340 labels at fixed 1,024 shots, keeps queue time excluded, and never mixes
Qonductor, QPack, IonQ or simulator targets into the training table.

Features are available before execution but only after target-specific
transpilation: logical width/depth/two-qubit-depth/gate-count; physical
depth/two-qubit-depth/gate-count/two-qubit-count/SWAP count; a reconstructed
weighted target-duration critical path; and backend identity. Numeric counts
are log transformed. The compiled features and duration path are still derived
from the frozen current `FakeOsaka`/`FakeKyoto` protocol, not recovered
historical physical circuits or calibration snapshots.

All evaluation rows are out of fold by QASM SHA-256. Three views are retained:
five-fold grouped logical circuits (within this source domain); paired-backend
diagnostic (the same QASM may appear on the other backend in training); and the
stricter 2 × 5 protocol that holds out both a backend and its logical-QASM
fold. Family-held-out leaves one public circuit family out and does not feed
the family token to the model. The table uses aggregate OOF predictions, not a
mean of small-fold R² values.

| Evaluation | Physical-depth Ridge: log R² / MAE | Compiled Ridge: log R² / MAE | Proxy + compiled residual RF: log R² / MAE |
| --- | ---: | ---: | ---: |
| Grouped logical circuits | 0.8246 / 0.6769 s | 0.8777 / **0.5116 s** | **0.8858** / 0.5303 s |
| Backend + logical circuit held out | 0.8215 / 0.6792 s | 0.8742 / **0.5224 s** | **0.8753** / 0.5565 s |
| Family held out | 0.8155 / 0.6890 s | 0.6734 / 0.6869 s | **0.8610** / **0.5630 s** |

The paired-backend value (`compiled Ridge` log R² `0.9052`) is retained in the
machine-readable result but is not used as primary transfer evidence because
it can see the paired logical circuit on the other device. The strict directions
remain separate: when Kyoto is held out, compiled Ridge reaches log R² `0.8491`
(MAE `0.5554 s`); when Osaka is held out, it reaches `0.8908` (`0.4970 s`).
There are only two backend directions, so this is evidence from a small,
source-specific domain rather than broad device generalization.

Within this proxy, compiled representation improves on depth-only structure.
The experiment does **not** isolate a causal benefit of duration alone: the
direct compiled models add several structural features at once, and the earlier
matched Ridge ablation found only a small incremental gain from weighted path.
No final deployable model is serialized, because selecting one after inspecting
these comparisons would require an explicit frozen selection rule. The script,
OOF predictions, per-fold metrics, strict-direction metrics and provenance are
under [artifacts/validation/mali_direct_estimator/](artifacts/validation/mali_direct_estimator/).

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

## 5. Azizov et al.: independent reproduction status

The paper studies measured Qiskit Aer noisy-simulation execution time after
backend-aware transpilation. Its starting pool is described as 1,510 Ma–Li
circuits, reduced to 1,402 unique circuits in 22 families, with two fake
backends, four optimization levels and 1,024 shots. The authors' code and raw
runtime table are not public yet, so this track is explicitly an independent
reproduction of the protocol rather than an exact numerical replication.

The P0 smoke run uses the pinned public Ma–Li QASM pool, source-hash
deduplication, one representative circuit per family with at most eight logical
qubits, current `FakeWashingtonV2` and `FakeSherbrooke`, and one timed Aer run
after one untimed backend warm-up. It produced 168 successful rows (21
families × 2 backends × 4 optimization levels). `T_transpile` and `T_exec` are
stored separately. The grouped baseline uses a circuit-ID grouped 80/20 split,
not a row-random split.

The committed P0 artifacts are:

- [P0 report](artifacts/azizov_independent/P0_REPORT.md)
- [raw P0 measurements](artifacts/azizov_independent/p0_smoke.csv)
- [environment record](artifacts/azizov_independent/p0_smoke.environment.json)
- [grouped baseline metrics](artifacts/azizov_independent/p0_baselines.json)
- [track instructions](tracks/azizov_independent/README.md)

These local seconds are not comparable as exact values to the paper's 32-CPU,
256-GB HPC measurements. P1 must add all unique circuits, per-run timeout and
censoring records; P2 must add backend-held-out, family-held-out and
transpiler-seed sensitivity tests.

### P1 bounded subset

A checkpointed subset then measured 63 circuits (three smallest eligible
circuits per family under a 16-qubit configuration; the observed maximum was
nine qubits), both fake backends, all four optimization levels and 1,024 shots.
All 504 rows completed successfully. On the same grouped circuit-ID split,
Random Forest reached `R²_log=0.7478`; SVR reached `0.7432`. The feature
ablation is more nuanced: compiled-only Ridge reached `0.6654`, source-only
Ridge `0.0346`, while Random Forest was similar for source (`0.7396`) and
compiled (`0.7360`) features. This supports testing compiled structure but does
not justify a GNN or a universal estimator yet.

The P1 subset is a bounded local pilot, not the paper's full 1,402-circuit
matrix. Its raw rows, environment and metrics are in
[artifacts/azizov_independent/](artifacts/azizov_independent/).

### Local feasible subset

To separate machine feasibility from paper coverage, a conservative screen was
applied to the 1,402-circuit manifest:

```text
logical_width <= 9
logical_ops <= 4000
logical_depth <= 4000
```

This retained 149 circuits (10.6%) and rejected 1,253. The rejected manifest
records the reason for every exclusion; most are width-related. The complete
eligible matrix contains 1,192 successful rows (149 circuits × 2 backends × 4
optimization levels), with zero timeout/error rows. The maximum observed
`T_exec` is 10.7751 s and the largest transpiled DAG has 153,548 nodes.

This is the current **empirically confirmed local subset**, not a claim that
all circuits above nine qubits are impossible. It excludes the unconfirmed
9-qubit QWalk circuit with 6,199 logical operations and every higher-width
circuit. A process-isolated runner can later test those rejected rows without
endangering the main job.

### Feasible-subset split evaluation

The 1,192 successful rows were evaluated with the same four baseline model
families (Ridge, SVR, Random Forest and HistGradientBoosting) on `log1p(T_exec)`.
The source block contains logical width/depth/operation counts plus backend
and optimization level; the compiled block contains post-transpilation width,
depth, operation counts, SWAP count and DAG-node count; the hybrid block joins
both. The full model-by-model table is in
[FEASIBLE_Q9_SPLIT_EVALUATION.md](artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.md),
with machine-readable values in the accompanying JSON.

The headline results (best model selected by log-scale R² within each block)
are:

| Split | Source R² (log / seconds) | Compiled R² (log / seconds) | Hybrid R² (log / seconds) |
| --- | --- | --- | --- |
| Random row | 0.7331 / 0.4763 | 0.7456 / 0.5438 | 0.7635 / 0.5430 |
| Grouped circuit (primary) | 0.8200 / 0.8667 | 0.8135 / 0.8006 | **0.8558 / 0.8230** |
| Family held out | 0.7884 / 0.6326 | 0.7973 / 0.7700 | **0.8375 / 0.7923** |
| Hold out FakeSherbrooke | 0.5845 / 0.5308 | 0.5308 / 0.5295 | 0.5511 / 0.5337 |
| Hold out FakeWashingtonV2 | 0.5511 / 0.3357 | 0.5255 / 0.4965 | **0.6297 / 0.5501** |

These numbers support three bounded conclusions. First, the local Aer runtime
is predictable on this screened domain with ordinary baselines. Second, the
hybrid representation is the strongest block on the grouped and family
hold-outs, so compiled structure adds signal but does not replace source
structure. Third, transfer between the two fake-backend snapshots is clearly
harder (log-R² about 0.53--0.63) than interpolation among circuits from the
same snapshots. The random-row result is optimistic because another row for
the same logical circuit can be present in training; it is not the primary
generalization claim. These are local baselines, not the paper's full
1,402-circuit GNN result.

### Width-10--16 resource boundary probe

The next-width canary was process-isolated and sampled RSS and CPU from
`/proc`, with a 16 GiB RSS cap and 180-second wall cap. It was stopped after
seven of the 41 planned width-10--16 source circuits because several rows were
already long enough to make a full canary disproportionate on this machine.
Four `FakeWashingtonV2` rows (`ae_q10`, `ae_q12`, `dj_q10`, `dj_q11`) hit the
16 GiB cap within about six seconds. Three rows completed: `dj_q16` in
0.681 s at 0.42 GiB peak RSS, `ghz_q10` in 66.367 s at 0.96 GiB, and `ghz_q11`
in 67.936 s at 1.05 GiB. The raw rows and protocol are in
[Q10_Q16_RESOURCE_CANARY_PARTIAL_REPORT.md](artifacts/azizov_independent/Q10_Q16_RESOURCE_CANARY_PARTIAL_REPORT.md).

This is a resource-boundary finding, not a failed reproduction. It shows that
logical width is not a sufficient capacity predictor: a sparse 16-qubit DJ
instance completed while some 10--12-qubit instances exceeded the RSS cap.
The result is specific to the current FakeWashington/Aer/default-parallelism
configuration and is not a claim that those circuits are impossible on another
Aer method, thread policy or machine. The partial probe is kept separate from
the 1,192-row q<=9 training/evaluation table.

Raising the guard to 36 GiB was also tested on `ae_q10` and `ae_q12`. Both
remained around 16.8 GiB RSS but hit the 180-second wall cap. Thus extra RAM
alone does not resolve the frontier bottleneck; a 40 GiB cap would leave only
about 5 GiB for the host and would mainly increase swap/OOM risk. Details are
in [Q10_Q16_HIGHMEM_CHECK_REPORT.md](artifacts/azizov_independent/Q10_Q16_HIGHMEM_CHECK_REPORT.md).
