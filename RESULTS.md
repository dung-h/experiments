# Results and failure analysis

Read the three finding clusters first:
[`docs/FINDINGS_CLUSTERS.md`](docs/FINDINGS_CLUSTERS.md).

This file keeps the older track-level tables. Scores are evidence about their
own estimator/target pairs; they are not a leaderboard.

| Cluster | Sections in this file | Packaged reports |
| --- | --- | --- |
| 1. Ma–Li × Azizov | §§1–2.13 and §5 | `artifacts/validation/mali_*`, `mali_azizov_*`, `artifacts/mali/` |
| 2. Quantum Rings solutions | Quantum Rings / iQuHACK 2026 | `artifacts/quantum_rings/REPORT.md` |
| 3. Family-aware paper | Family-aware paper subsection | `artifacts/quantum_rings/family_aware_paper/` |

The chronological record of what was run from 17 to 22 September 2026 is
[docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md](docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md).
The compact table is [docs/experiment_tracker.csv](docs/experiment_tracker.csv).

## Comparative conclusion

The studies measure different target semantics. The three finding clusters
are summarised in docs/FINDINGS_CLUSTERS.md; the table below still lists
every packaged track. Their scores are evidence
about their own estimator/target pairs; they are not entries in one leaderboard.

The dense-statevector results below are a separately labelled local supplement.
They use one PyTorch reference kernel and are not another QPU, Aer, CUDA-Q or
cuTensorNet target.

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
| cuTensorNet | 70% 55-row: warm actual/EST 0.215. 90% QFT q36–q44: 4.19–5.11. Ma–Li MQT QASM q≤10: 176/176 ok, median actual/EST 0.231; `grover-v-chain_9` 2.227 | `RUNTIME_EST` is a contraction-path cost proxy, not a wall-clock; cheap q≤10 QASM over-predicts, large QFT under-predicts | One GPU/software/precision setting; synthetic 70%, 90% QFT, and Ma–Li QASM tables are not pooled; QPU labels are not used |
| Azizov et al. independent | 1,192/1,192 screened local rows successful; 149/1,402 circuits eligible | Current FakeWashingtonV2/FakeSherbrooke + Aer pipeline is executable on a documented local subset | Conservative q≤9/ops/depth screen; not the paper's full HPC dataset or exact artifact |
| Quantum Rings / iQuHACK | Spirit Sprinters public 144-row log-R² 0.743, MAE 107 s; SoftLocked in-domain R² 0.967 and public-table log-R² −25.5 | Winner is a structural public reference; SoftLocked cannot be compared until clocks match | Contest labels have no host CPU/GPU model; no SDK rerun |

## Quantum Rings / iQuHACK 2026

**Finding recorded: 2026-09-18. Packaged: 2026-09-23.** Full write-up:
[artifacts/quantum_rings/REPORT.md](artifacts/quantum_rings/REPORT.md).

The contest asks for a truncation-threshold rung and a 10,000-shot forward
simulator wall time from OpenQASM plus a CPU/GPU and single/double tag.
The public table has 36 circuits × 4 contexts = 144 rows. Holdout QASM is
private. The Quantum Rings SDK was not rerun.

The public labels do not describe the simulation machine. They expose
`backend`, `precision`, and per-run `peak_rss_mb`. They do not name CPU
model, GPU model, core count, DRAM, VRAM, OS, or the SDK version that
produced the times. CPU versus GPU is a categorical tag, not a device
profile. These rows cannot train a hardware-conditioned estimator of the
kind used for the RTX 5070 Ti cuTensorNet track.

Spirit Sprinters (official winner, commit `14b5ea14`) is an XGBoost
threshold classifier plus an edge-aware Transformer duration model. On the
public 0.99-forward table its log-second R² is 0.743 and MAE is 107 s.
Threshold exact accuracy is 99.3% against 0.75 labels and 86.1% against
0.99 known rows. The duration model still misses large circuits: `dj` 130
qubits, CPU double, is 2,588 s true versus 1,446 s predicted.

SoftLocked (commit `da13c074`) is a ~90-feature GradientBoosting pair.
On its own `training_data.csv` a grouped rerun gives threshold exact
88.62% and seconds R² 0.967, matching the README. The same checkpoint on
the public 144-row table gives log-R² −25.5 because its labels sit at a
median of 74,048 s against a public median of 20 s. Its CLI also ignores
the CPU/GPU tag.

The two artifacts are not a leaderboard. They are a structural reference
and a domain-shift warning. Bloch, Rishivarshil and Hazel are outside
this track.


## Family-aware paper (arXiv:2606.11620)

**Finding recorded: 2026-09-18. Packaged: 2026-09-23.** This is Cluster 3,
not Cluster 2. Cluster 2 scores Spirit Sprinters and SoftLocked checkpoints.
This section reconstructs the family-conditioned paper protocol on the same
144 public rows. Full write-up:
[artifacts/quantum_rings/family_aware_paper/](artifacts/quantum_rings/family_aware_paper/).

Two targets exist and must not be pooled. Paper-style 0.75 uses mirror-sweep
wall time at the first 0.75 crossing (a proxy; the public JSON was generated
at 0.99). Challenge 0.99 uses the 10,000-shot forward wall time.

On circuit-level 5-fold CV, trees are the strongest public runtime baseline:
Gradient Boosting log-R² 0.757 on the 0.75 proxy, ExtraTrees 0.712 on 0.99
forward. The reconstructed family MLP is negative for runtime on both
targets. Fold-local family accuracy is 39%. A frozen public MQT MLP reaches
39/40 on its own MQT holdout and does not beat trees on the contest table.
The paper's reported runtime R² 0.82 is not recovered.

## 0. Local dense-statevector runtime estimator supplement

**Finding recorded: 2026-09-20.** This supplement answers a narrower engineering
question: can a runtime estimator and feasibility guard be calibrated for one
known simulator contract, before claiming transfer across frameworks or
machines? The target is `prepared_execute_reset_state`: allocate a fresh dense
statevector, apply a pre-materialized direct gate schedule and evaluate
`<Z_0>`. Construction, materialization, compilation/transpilation, sampling,
cloud and QPU time are outside the label.

The v1 matrix contains 96 successful rows across GHZ, HEA, QAOA-cycle, random
brickwork and QFT at widths 16/20/24, crossed with CPU/GPU and complex64/
complex128. The v2 extension adds `random_matching` and `random_star`, seeds
17/43/101, widths 16–28 and a GPU frontier. The canonical corpus has 208
rows: 204 duration labels and four explicit `resource_limit` rows. No runtime
is imputed for an OOM attempt.

The primary circuit-group split holds all device/precision variants of an
unseen logical circuit out of training. `logical_hgb` reaches log-R² `0.9794`
on that in-range test, improving on the analytical statevector-bytes plus
gate-work baseline (`0.9633`). The result is evidence that logical gate
composition adds signal within this fixed kernel, not evidence of universal
simulator prediction. Coarse interaction-graph features do not improve the
logical ablation.

Width holdout changes the conclusion: analytical Ridge reaches log-R² `0.8754`
versus `0.7295` for logical HGB. Thus the nonlinear model is strong for
interpolation but is not automatically safe for width extrapolation.

The most operational finding is the GPU memory boundary. The q28 complex128
statevector is 4 GiB and passes the naive `statevector_bytes <= VRAM` check,
but all four attempts hit `resource_limit`. A peak-reserved/statevector
envelope calibrated only through q24 predicts that boundary before execution.
The q28 complex64 envelope predicts feasible and all four observations finish.
This is a machine- and kernel-specific feasibility guard, not a transferable
VRAM law. The protocol, raw rows, OOF predictions and report are in
[`experiments/simulator_runtime_v2/README.md`](experiments/simulator_runtime_v2/README.md)
and `artifacts/simulator_runtime_v2/`.

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

### 2.6 Deep physical-DAG protocol and first model gate

**Finding recorded:** 2026-09-21.

The deeper experiment starts by separating circuit identity from filename and
backend. Raw-QASM SHA-256 grouping yields 170 logical circuits, 130 paired
across Osaka/Kyoto, 300 distinct `(QASM, backend)` physical graphs and 9
family connected components. The strict backend-plus-unseen-QASM manifest has
zero train/test hash overlap in all 10 diagnostic partitions. The split audit
and rationale are in
[artifacts/validation/mali_deep_protocol_v1/split_audit/](artifacts/validation/mali_deep_protocol_v1/split_audit/).

Using the frozen Qiskit 1.4.1 / IBM Runtime 0.36.1 protocol, the compact
current-FakeBackend corpus contains 67,404,145 native-operation nodes and
79,673,830 dependency edges across 300 graphs. No operation lacked a target
duration. The largest graph has 337,142 nodes; QWalk has only 9 active
physical qubits but 107,143 native nodes. These are reproducible physical
proxies, not historical IBM transpilation/calibration records.

The static model gate is intentionally ordered from simple to rich:

| Input | Grouped-QASM log R² / MAE | Strict backend+QASM log R² / MAE | Family-component log R² / MAE |
|---|---:|---:|---:|
| QCRE weighted path | 0.7321 / 0.8274 s | 0.7314 / 0.8229 s | 0.6207 / 0.9157 s |
| Compiled depth/width | 0.8202 / 0.6889 s | 0.8156 / 0.6924 s | 0.8021 / 0.7228 s |
| Physical node/edge summary | 0.8527 / 0.5766 s | 0.8500 / 0.5817 s | 0.8351 / 0.6136 s |
| Rich layer/opcode/timing summary | **0.8896 / 0.4937 s** | **0.8755 / 0.5230 s** | **0.8700 / 0.5752 s** |

The rich summary is still static feature engineering; it does not prove that
message passing or attention is useful. A coarsened 256-bin ordered-DAG GRU
residual was then tested with group-disjoint inner validation. It reached only
pooled log R² `0.7339` (MAE `0.8571 s`) on grouped circuits and `0.6988`
(`0.9022 s`) on strict splits. It reduced the QWalk errors to approximately
5.45/5.24 s, but remained worse overall than the rich static summary. The
current interpretation is therefore: compiled timing/layer/opcode descriptors
carry strong signal, while the first DAG sequence model loses information under
coarsening and is not yet justified as an estimator. Reports, raw manifests,
OOF predictions and the GPU-run protocol are in
[artifacts/validation/mali_physical_dag_v1/](artifacts/validation/mali_physical_dag_v1/).

### 2.7 Feature-block ablation and QWalk tail diagnostic

**Finding recorded:** 2026-09-21.

The next pass keeps exactly the same grouped-QASM, strict backend-plus-QASM and
family-component folds. It adds one interpretable block at a time to the small
log-ridge estimator. The grouped-QASM results are:

| Block | MAE (s) | log-R² | seconds-R² |
|---|---:|---:|---:|
| QCRE weighted path | 0.8274 | 0.7321 | 0.6707 |
| + compiled depth/width | 0.6889 | 0.8202 | 0.7660 |
| + native node/edge count | 0.5766 | 0.8527 | 0.7925 |
| + layer descriptors | 0.5635 | 0.8616 | 0.8022 |
| + duration/criticality descriptors | 0.5109 | 0.8892 | 0.8446 |
| graph size + opcode counts | 0.5563 | 0.8241 | 0.7656 |
| all rich descriptors | **0.4937** | **0.8895** | **0.8432** |

The same ordering holds on strict backend-plus-unseen-QASM (rich log-R²
`0.8755`) and family held-out (rich log-R² `0.8700`). The useful signal is
therefore concentrated in compiled size/layer and especially duration/path
descriptors; opcode counts alone are weaker under family holdout. The edge block
is identical to the compact node/edge row here because the global quantum-edge
fraction is constant for this reconstructed corpus. This is evidence for a
strong static estimator, not evidence that message passing or DAG attention is
needed.

QWalk is retained as a separate forensics artifact. Both rows share one raw-QASM
hash and are held in fold 5. Their nearest standardized training-row distance is
about `3.02--3.12` RMS, while active width and criticality are outside or at the
edge of the training support. The rich static model lowers grouped absolute
error to about `5.91 s` (Osaka) and `5.35 s` (Kyoto); the coarsened DAG-GRU lowers
it further locally to `5.45 s` and `5.24 s`, but loses badly on aggregate
metrics. The defensible conclusion is an OOD/tail limitation: add coverage or
uncertainty handling and test critical-subgraph features before increasing GNN
complexity. Full ablation and feature-support tables are under
`artifacts/validation/mali_physical_dag_v1/feature_ablation/` and
`.../qwalk_forensics/`.

### 2.8 Critical-subgraph and uncertainty pass

**Finding recorded:** 2026-09-21.

High-criticality bottleneck summaries were added to the rich static descriptor
model. They improve log-R² from `0.8895` to `0.9041` on grouped QASM, and from
`0.8755` to `0.8869` on strict backend-plus-unseen-QASM. The MAE changes only
from `0.4937` to `0.4987 s` grouped and from `0.5230` to `0.5306 s` strict;
family-held-out log-R² improves only from `0.8700` to `0.8732` while MAE rises
to `0.5934 s`. This is promising for a residual/uncertainty feature, but not
yet enough to justify a larger critical-subgraph GNN.

The duplicate-label audit finds 40 exact `(raw-QASM hash, backend)` cells with
80 rows. Their median absolute runtime difference is `0.4171 s`, P90 `1.2775 s`,
and maximum `1.8158 s`; a pair-mean RMSE floor is about `0.3541 s`. This puts a
real lower bound on how precise a scalar estimator can be under the recorded
target semantics.

Split-conformal intervals, calibrated on a QASM-disjoint training group, cover
`87.6%` of grouped rows at nominal 90% and `86.5%` under strict backend holdout.
At nominal 80%, coverage is `78.8%` and `76.8%`; median widths are `1.48--1.54 s`
for the 80% bands. These are empirical dataset/proxy intervals, not hardware-wide
confidence guarantees. The intended deployment behavior is therefore a point
estimate plus uncertainty band and an OOD flag for QWalk-like inputs.

Artifacts: `artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/`
and `.../uncertainty_calibration/`.

### 2.9 Retrained topology-control validation

**Finding recorded:** 2026-09-21.

The prior saved-model edge interventions showed that connectivity mattered, but
could not establish whether retrained message passing on the correct DAG was
better than node statistics. A controlled 256-bin experiment therefore held
node descriptors, residual target, optimizer, folds and capacity fixed while
changing only adjacency: node-only, true dependency DAG, reversed DAG and
deterministically shuffled destinations. Three independent seeds were trained
on grouped-QASM folds; 5,000 paired bootstrap replicates resampled the 170 raw
QASM hashes, preserving paired backend rows.

| Variant | Seed-mean MAE (s) | Seed-mean log-R² |
|---|---:|---:|
| Node-only | **0.8145** | **0.7666** |
| True DAG | 0.8448 | 0.7382 |
| Reversed DAG | 0.8967 | 0.6991 |
| Shuffled DAG | 0.8530 | 0.7285 |

For true DAG minus node-only, the log-R² delta is `-0.0284`, with a paired
QASM-group bootstrap interval `[-0.0650, 0.0008]`; only `2.9%` of replicates
favor true DAG. The MAE delta is `+0.0303s`. True DAG remains better than
reversed/shuffled adjacency, so direction contains some information, but the
coarsened message-passing representation does not improve runtime prediction.

This is a useful negative result: do not scale up GAT/GINE/TransformerConv on
the same 256-bin representation. It does not rule out a better native-DAG or
critical-subgraph representation, and it has not yet been repeated under strict
backend-plus-unseen-QASM splits. Raw per-seed OOF rows and the bootstrap are in
`artifacts/validation/mali_physical_dag_v1/topology_controls_*`.

### 2.10 Strict no-real-QASM simulator transfer screen

**Finding recorded:** 2026-09-21.

The earlier Ma--Li adaptation comparison could retain real-circuit identities
in simulator pretraining. A deliberately narrower test removes all 170 raw
QASM hashes in the observed Osaka/Kyoto set from the simulator corpus: 384 of
3,020 simulator rows are removed, leaving 2,636 rows and 1,232 simulator QASM
hashes. It uses a common logical-QASM feature MLP rather than the unavailable
full Graph Transformer, so it tests circuit-inductive transfer without claiming
an exact paper reproduction.

| Three-seed mean model | MAE (s) | log-R² |
|---|---:|---:|
| Scratch on observed-runtime training folds | **0.6831** | **0.3360** |
| Frozen simulator encoder | 1.9152 | -0.2533 |
| Fine-tuned simulator encoder | 0.6956 | 0.0816 |

Grouped-QASM bootstrap gives fine-tuning minus scratch Δlog-R² in
`[-0.8351, 0.0000]` and ΔMAE in `[-0.0354, 0.0572] s`; only 2.6% of resamples
favor fine-tuning in log-R². Frozen transfer is decisively worse in MAE. Thus,
after removing exact circuit overlap, this representation does not support a
reliable simulator-pretraining benefit. It neither tests nor disproves a strict
Graph Transformer experiment with historical calibration snapshots.

Artifacts: `artifacts/validation/mali_strict_transfer_logical_v1/` and its
`evaluation/` subdirectory.

### 2.11 Duplicate-aware fitting and support-aware uncertainty

**Finding recorded:** 2026-09-21.

Each test target remains an original observed row. Only rows inside the outer
training partition are collapsed by exact `(raw-QASM hash, backend)` cell. On
grouped-QASM evaluation, equal-weighted cell means improve MAE/log-R² from
`0.4937 s`/`0.8895` to `0.4923 s`/`0.8917`; on strict backend-plus-QASM holdout,
noise-weighted cell means improve them from `0.5230 s`/`0.8755` to
`0.5156 s`/`0.8799`. These are modest robustness gains, not nested model
selection results or a way to erase observed runtime noise.

For the same rich static estimator, nearest standardized circuit-feature
distance is calculated using fit rows only and a QASM-disjoint calibration fold
sets global or two-band conformal residual quantiles. QWalk is consistently in
the high-distance band. At nominal 90%, global intervals cover `87.6%` of
grouped and `86.5%` of strict rows, but cover neither QWalk row. The
support-stratified band widens QWalk intervals (grouped median `2.7141 s` versus
global `2.1888 s`) without covering it. Therefore an estimator may expose an
OOD flag and request abstention/additional calibration; it must not present
these intervals as automatic tail or hardware-wide guarantees.

Artifacts: `artifacts/validation/mali_physical_dag_v1/duplicate_aware_validation/`
and `.../ood_uncertainty_validation/`.

### 2.12 Family-OOD protocol: missing MQT families versus unseen hardware families

**Finding recorded:** 2026-09-22.

The earlier family-held-out log-R² of about `0.87` on the 340 Osaka/Kyoto rows
is a statement about the 10 hardware-selected MQT families. It does not score
the 12 MQT families that never entered that table. Those questions are now
separated.

Experiment A compares structure, not runtime. The local MQT extract has 1,510
independent-Qiskit QASMs. The hardware table uses 170 hashes from 10 families;
549 circuits from GHZ, W-state, GraphState, DJ, AE, VQE, QAOA, Grover and the
remaining walk/portfolio variants have no `result.time_taken` here. Selected
families are dense (median two-qubit count 4,606, interaction density 1.0).
Missing families are sparse (81 two-qubit gates, density 0.034). Leave-one-family
nearest-group assignment is 20/22. The exceptions are `ae`, which sits nearer
the selected/QPE region, and `qwalk-noancilla`, which sits nearer the missing
region. Inside the 10 selected families the hardware hashes are still a size
bias: sampled QASMs have median width 91 versus 70 for unsampled members of
those same families. Physical FakeOsaka/FakeKyoto proxy rows cover all 1,510 QASMs on both
backends (3,020/3,020 ok). Compiled structure widens the split: median
physical depth 17,040 versus 275, routing-depth ratio 62.2 versus 7.6,
centroid distance 5.87 versus 4.46 logically. The same two families are
mis-assigned (`ae`, `qwalk-noancilla`). Rows are under
[artifacts/validation/mali_family_ood_v1/feature_space_audit/](artifacts/validation/mali_family_ood_v1/feature_space_audit/).

Historical Experiment B remains blocked: no Osaka/Kyoto `result.time_taken`
for the 12 missing families, and both machines are retired. The unblocked
sibling is B2, which keeps C's QPU splits and puts the held family back into
simulator pretraining. Scratch T0–T3 match C exactly. Logical T0 `incl` is
worse than `excl` (log-R² `-0.377` vs `-0.282` excluding QWalk). Compiled-proxy
T2 is the best sim arm (`0.248` / `1.363 s`) and still far behind scratch T1
(`0.859` / `0.624 s`). The weak T2 pooled score is ansatz-driven; QFT/QPE/QNN
remain large-negative. Seeing a family on FakeWashington/Sherbrooke does not
replace compiled QPU-scratch features.

Live-QPU collection for the 12 missing families was removed on 2026-09-22.
It required IBM credentials that are not available here, and no jobs were
submitted. External public extracts (QPack, Qonductor, IonQ, simulators) also
do not supply trusted QASM-joined hardware labels for those MQT instances.
See [docs/REMOVED_AND_OUT_OF_SCOPE.md](docs/REMOVED_AND_OUT_OF_SCOPE.md).
The day-by-day record is
[docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md](docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md).

Experiment C holds out each of the 9 hardware connected components from both
simulator pretraining and QPU training. Simulator pretraining stays on logical
T0; Washington/Sherbrooke compilation is not transferred as Osaka/Kyoto
structure. Excluding the two QWalk rows, QPU-scratch logical T0 has log-R²
`-1.283` (MAE `1.259 s`). Compiled T1 jumps to `0.859` / `0.624 s`. Adding the
duration-weighted path (T2) is `0.860` / `0.612 s`. The rich static summary
(T3) reaches `0.893` / `0.546 s`, and the all-component T3 value `0.8700`
matches the earlier family-component rich-summary gate. Simulator T0 plus
affine QPU calibration is `-0.282` / `1.696 s` and does not replace compiled
scratch. Per-family, logical T0 collapses on QNN and `random`; compiled T1
restores them; T3 is not uniformly best (`random` prefers T1/T2).

Artifacts: [artifacts/validation/mali_family_ood_v1/](artifacts/validation/mali_family_ood_v1/).


### 2.13 Compiled transfer (Ma–Li protocol, Azizov representation)

**Finding recorded: 2026-09-22.** Cluster 1 treats Ma–Li and Azizov as one
object: keep Ma–Li's two clocks and 340 Osaka/Kyoto labels, replace the
logical DAG with the circuit after `transpile` to the target backend.

Compiled Ridge on Washington/Sherbrooke simulator `time_taken` (n=3020):
logical T0 log-R² 0.272, compiled T1 0.313, compiled T2 0.435. The same
Ridge trained from scratch on the 340 hardware rows: T0 0.720, T1 0.863,
T2 0.875. Frozen simulator T2 plus an affine QPU head reaches 0.353
(`incl`) and 0.349 (`excl`). Seeing the test QASM on the simulator does
not matter. QPU scratch remains the ceiling.

The coarsened native DAG (256 bins) is stronger on the simulator clock
(log-R² 0.779 versus Ridge T2 0.435) and only slightly stronger on QPU
scratch (0.905 versus 0.875). Affine DAG transfer collapses to log-R²
−2.63 because two QWalk rows predict thousands of seconds. Fine-tune from
the simulator DAG recovers 0.875, close to scratch. Ignore
`qpu_scratch_pilot/` (`softplus` decode bug).

Reports:
[mali_azizov_compiled_transfer_v1](artifacts/validation/mali_azizov_compiled_transfer_v1/)
and
[mali_azizov_transpiled_dag_v1](artifacts/validation/mali_azizov_transpiled_dag_v1/).
English reading path: [docs/FINDINGS_CLUSTERS.md](docs/FINDINGS_CLUSTERS.md).

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

### 4.1 Feasibility frontier and warm calibration

The same runner was later expanded to a timeout-bounded 55-cell grid (GHZ,
QFT, HEA 2/4/8, QAOA-cycle p=2/3/6, brickwork 4/8/16, widths 16/20/24/28/30).
All 55 cells completed. On that frontier the warm median actual/EST is 0.215.
Family-held-out calibration, using only pre-contraction planner and circuit
features, reproduces the earlier warm-B1 decision: a train-fold median-ratio
correction has log MAE **0.131**, ahead of Ridge 0.247 and HGB 0.253. Ridge
is stronger on unseen widths (0.089) and should not be selected from that
split alone. Artifact:
[CUTENSORNET_B1_CALIBRATION_RESULTS.md](artifacts/cutensornet/feasibility_frontier_v1/CUTENSORNET_B1_CALIBRATION_RESULTS.md).

### 4.2 QFT width frontier at 90% workspace (23 September 2026)

The 25-row and 55-row tables used a 70% workspace policy. A later QFT-only
sweep on the same GPU changed that cap to 90% and pushed width from 28 to 44
qubits. It is a new measurement table, not a replacement of §4.1.

`90%` is a feasibility cap of about 14.95 GB, not a request to occupy 90% of
VRAM. q36 remains unsliced with a ~8 GiB largest intermediate. From q40 the
planner switches to a ~4 GiB intermediate and 2, then 4, then 8 slices.

| QFT width | slices | `RUNTIME_EST` | warm actual | actual / estimate |
|---:|---:|---:|---:|---:|
| 28 | 1 | 5.617 ms | 5.114 ms | 0.910 |
| 32 | 1 | 10.612 ms | 20.418 ms | 1.924 |
| 36 | 1 | 185.939 ms | 778.812 ms | 4.189 |
| 40 | 2 | 0.726 s | 3.641 s | 5.015 |
| 42 | 4 | 2.229 s | 10.402 s | 4.668 |
| 44 | 8 | 5.796 s | 29.634 s | 5.113 |

Three facts matter for the estimator:

1. Graph plus VRAM cap plus architecture 12 is enough to choose a feasible
   plan. It is not enough to predict the exact kernel/layout time of that plan.
2. Slicing is not the only residual. q36 is still one slice and already 4.19×
   slow. q44 first contraction (29.614 s) and warm contraction (29.634 s) are
   almost identical, so the 5.11× gap is not cold-start.
3. The high-level optimizer is not empirically unique. Three plan-only q40
   repeats with the same seed, sample budget and 90% cap returned 2, 2 and 4
   slices (`RUNTIME_EST` 0.417 s, 0.415 s, 0.687 s).

`EFFECTIVE_FLOPS_EST` on this run equals `RUNTIME_EST × 2.00e13`. That is the
NVIDIA time-objective definition, not measured throughput.

The 70% 25-row QFT q16–q24 rows over-predict (actual/estimate 0.25–0.32). The
90% large-QFT rows under-predict. Do not apply the 25-row median ratio 0.196
to this table.

Artifact: [runtime_est_workspace90_frontier_v1_20260923/REPORT.md](artifacts/cutensornet/runtime_est_workspace90_frontier_v1_20260923/REPORT.md).

### 4.3 Ma–Li MQT Bench QASM, q≤10 (23 September 2026)

The 25-row and 55-row grids used generated GHZ, HEA, QAOA-cycle, random
brickwork and QFT tensors. They did not load the public Ma–Li MQT Bench
OpenQASM pool. This run does: 1,510 `*_indep_qiskit_*.qasm` files, 22
families, still on the local GPU contraction clock, not Osaka/Kyoto
`result.time_taken`.

Exact contraction of the full pool is not feasible on one 16 GB GPU. The
executable subset is all 22 families with 2–10 qubits and at most 2,500
gates: 176 timed, 1,331 skipped for width, three deferred heavies
(`grover-noancilla_8`, `qwalk-noancilla_8`, `qwalk-noancilla_9`). All 176
selected circuits finished.

Median warm actual / `RUNTIME_EST` is 0.231. That matches the cheap-kernel
70% synthetic over-prediction, not the 90% QFT q36–q44 under-prediction,
even though this table also used a 90% workspace cap. Family medians sit
between 0.194 and 0.322 except `grover-v-chain` (0.430), whose q9 file is
the only reversal: warm 52.319 ms versus estimate 23.497 ms (2.227), with a
~2 GiB intermediate. The only sliced plan is `qwalk-noancilla_6` (32
slices, actual/estimate 0.438). `qwalk-noancilla_7` spent 108.7 s in path
search for a 4 ms warm contraction; that end-to-end cost is not
`RUNTIME_EST`.

Do not apply 0.231 to the QFT-44 row, and do not treat 176 circuits as the
full 1,510-file pool.

Artifact: [mali_qasm_qle10_v1_20260923/REPORT.md](artifacts/cutensornet/mali_qasm_qle10_v1_20260923/REPORT.md).

### 4.4 local follow-ups, not packaged

The 2026-09-22 multi-target calibration, the paired CUDA-Q / cuTensorNet IR,
and the 2026-09-23 runtime-estimate trace remain in the working tree. They are
not part of this packaging round.

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
