> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../docs/thesis/README.md).

# RQ4 final report

Status: complete with explicit limitations. Final Scientific Freeze v2.
This report closes the fixed grid of nine family, ten backend-context and two
QPack size cases. All 21 CPU cases and 126 neural method–seed cells completed.
The cohort remains 4,515 observations and 165 conservative structural groups.

## Estimators, populations and uncertainty

The protocol-defined primary neural estimator is a **median-of-three prediction
ensemble** of seeds 42, 1234 and 31415. Each observation's three predictions
are reduced before scoring. This is neither a chosen seed, mean seed score nor
an extra test-selected model. Individual seed scores and dispersion remain
available. Epoch selection used inner validation, followed by fresh full
outer-train refitting in this RQ4 study. The parent grouped OOF neural study
has its own retained-fit procedure and is not silently relabeled.

Within each case, train/test observation IDs are disjoint and test IDs unique.
Structural overlap is intentional in backend-context transfer. Across cases,
6,684 test assignments cover 4,515 unique IDs. They are different estimands,
not a pooled OOF sample. Family, backend and size tables remain separate.

Intervals are saved 10,000-replicate structural-group bootstrap intervals,
seed 42, conditional on fitted predictions. Fewer than ten shared groups means
no interval. Seed spread does not provide independent test replication.
An equal-case mean is descriptive, not an interval over independent hardware
domains. R² is never averaged. All point deltas below are observed Graph MAE
minus MLP MAE, not the bootstrap mean. Negative favors Graph.

CPU selector and outer-train global/source median references use the same
case labels. Source median falls back to train global median if that source
is absent. References are secondary analyses, not new primary methods.
CPU analytical coverage/failures remain explicit in the complete method table.
No method or seed is selected by test performance.

## Complete per-case comparison

MAE and intervals are seconds. Graph/MLP are the median-of-three ensembles.
Train-source median requires known source provenance at deployment.

### Family holdouts

| Case | Test rows / groups | Graph MAE / R² | MLP MAE / R² | CPU selector MAE | Train global / source median MAE | Graph−MLP delta [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qft | 46 / 26 | 0.563 / 0.532 | 0.513 / 0.573 | 0.528 | 2.494 / 0.939 | 0.050 [-0.089, 0.183] |
| qftentangled | 54 / 27 | 0.743 / 0.841 | 0.552 / 0.907 | 0.788 | 2.570 / 0.792 | 0.191 [-0.014, 0.417] |
| qnn | 26 / 14 | 36.272 / -2197.298 | 50.764 / -4318.143 | 0.434 | 2.658 / 4.751 | -14.492 [-15.774, -13.172] |
| qpeexact | 40 / 23 | 2.202 / -0.580 | 0.604 / 0.847 | 0.876 | 3.064 / 1.512 | 1.599 [1.227, 2.009] |
| qpeinexact | 36 / 22 | 1.575 / -3.975 | 0.713 / -0.096 | 0.978 | 3.181 / 1.423 | 0.862 [0.564, 1.160] |
| random | 30 / 16 | 145.708 / -13158.490 | 188.815 / -22124.320 | 1.431 | 3.627 / 1.722 | -43.106 [-47.517, -38.917] |
| realamprandom | 40 / 22 | 0.636 / 0.426 | 0.952 / -0.203 | 0.433 | 1.121 / 1.496 | -0.316 [-0.404, -0.232] |
| su2random | 38 / 21 | 0.589 / 0.541 | 1.465 / -1.444 | 0.378 | 1.168 / 1.306 | -0.876 [-1.121, -0.620] |
| twolocalrandom | 40 / 22 | 1.194 / -0.463 | 1.259 / -0.658 | 0.548 | 1.396 / 1.311 | -0.065 [-0.171, 0.042] |

### Backend-context holdouts

| Case | Test rows / groups | Graph MAE / R² | MLP MAE / R² | CPU selector MAE | Train global / source median MAE | Graph−MLP delta [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ibm_belem | 24 / 3 | 0.970 / -1.165 | 0.492 / 0.211 | 0.504 | 3.848 / 0.666 | 0.478 not estimated (<10 groups) |
| ibm_jakarta | 408 / 8 | 2.061 / -6.417 | 1.245 / -2.062 | 1.449 | 0.633 / 0.791 | 0.817 not estimated (<10 groups) |
| ibm_kyoto | 148 / 117 | 0.664 / 0.796 | 0.541 / 0.854 | 0.807 | 2.430 / 1.385 | 0.123 [0.056, 0.190] |
| ibm_lagos | 368 / 5 | 0.752 / -2.406 | 0.287 / 0.004 | 0.347 | 0.731 / 0.695 | 0.465 not estimated (<10 groups) |
| ibm_lima | 41 / 4 | 1.705 / -4.750 | 0.681 / -0.332 | 0.783 | 4.357 / 1.127 | 1.024 not estimated (<10 groups) |
| ibm_manila | 367 / 8 | 0.588 / 0.579 | 0.831 / 0.495 | 0.885 | 0.657 / 0.359 | -0.244 not estimated (<10 groups) |
| ibm_nairobi | 1565 / 9 | 1.484 / -21.657 | 1.073 / -10.025 | 1.317 | 1.282 / 1.397 | 0.410 not estimated (<10 groups) |
| ibm_osaka | 192 / 148 | 0.797 / 0.734 | 0.430 / 0.914 | 0.645 | 2.265 / 1.481 | 0.367 [0.274, 0.463] |
| ibm_perth | 351 / 9 | 1.983 / -22.197 | 1.499 / -12.161 | 1.544 | 2.190 / 2.048 | 0.484 not estimated (<10 groups) |
| ibm_quito | 1051 / 7 | 1.958 / -6.125 | 1.517 / -3.559 | 1.575 | 1.969 / 1.842 | 0.442 not estimated (<10 groups) |

### QPack size probes

| Case | Test rows / groups | Graph MAE / R² | MLP MAE / R² | CPU selector MAE | Train global / source median MAE | Graph−MLP delta [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train_through_4 | 1321 / 3 | 0.727 / -0.072 | 0.641 / 0.063 | 0.877 | 0.845 / 0.845 | 0.086 not estimated (<10 groups) |
| train_through_5 | 498 / 2 | 0.474 / -4.685 | 0.789 / -13.060 | 0.396 | 0.867 / 0.867 | -0.314 not estimated (<10 groups) |

## Scientific findings and failure mechanisms

The equal-family descriptive MAE is Graph 21.054 s, MLP 27.293 s, CPU selector
0.710 s. Equal-backend MAE is 1.296/0.860/0.986 s. MLP has lower MAE in 9/10
backend cases and 14/21 total cases. Neither count establishes a universal
winner. Backend/source/workload composition does not isolate a causal
hardware-generation or topology effect.

QNN (26 rows, 14 groups) and Random (30 rows, 16 groups) have severe all-row
overprediction by both neural ensembles. Actual runtime ranges are 3.107–7.376
and 7.981–12.751 s. Graph/MLP MAE is 36.272/50.764 s and 145.708/188.815 s.
Negative R² remains visible above. CPU selector MAE is 0.434/1.431 s on these
same cases. Graph's relative advantage is not practically useful prediction.
No outlier or bad seed is discarded. QNN feature counts exceed fitted support;
Random loses unseen fields under train-only masks and retains extreme scaled
counts. These are diagnostics, not isolated causal feature interventions.

QPack `train_through_4` tests 1,321 rows/3 groups and `train_through_5` tests
498/2. They are low-support descriptive probes, not a broad scaling law.
QPack contributes 87.4% of parent observations but six structures. Parent
pooled, source-balanced and equal-group conclusions remain distinct and
unchanged. The full grid does not establish universal generalization.

### Kyoto, Osaka and Quito

No observation-ID overlap was found within these cases. Complete Graph tensor
inputs have zero exact fit collisions in all three. Graph topology may repeat
while nominal calibration inputs differ. MLP omits backend, calibration and
shots. Exact MLP collisions therefore limit the claim, not imply ID leakage.

| Context | Test observations / groups | Previously seen / new-group rows | MLP fit collisions | Test sources |
| --- | ---: | ---: | ---: | --- |
| ibm_kyoto | 148 / 117 | 148 / 0 | 148/148 | mali_real_qpu: 148 |
| ibm_osaka | 192 / 148 | 149 / 43 | 149/192 | mali_real_qpu: 192 |
| ibm_quito | 1051 / 7 | 1051 / 0 | 1051/1051 | qonductor_single_circuit_ibm: 42, qpack_mcp: 1009 |

Kyoto is known-circuit backend-context transfer, not unseen-circuit or proven
hardware adaptation. Osaka mixes known/new structures. Quito is predominantly
repeated QPack structures and also contains Qonductor. Source-stratified scores
and support scopes must accompany their pooled case values.

Some frozen input-support summaries contain a malformed scalar identity.
Some reader rows retain the historical `pending_transform_validation` wording.
These strings are not silently repaired or used as authoritative identity
objects. The component/aggregate audit binds row IDs, hashes and final-refit
transforms separately. See source receipts and the data-quality audit ledger.

## Execution provenance and historical reconciliation

The ten-case report has 1,950 matching shared metric rows. The twelve-case
targeted report has 50 matching nonempty neural rows and 22 empty scopes with
no score. There are zero shared metric mismatches. They remain historical
snapshots, superseded for current coverage, not invalidated or overwritten.

Twolocalrandom was **unauthorized at execution**. Its six historical cells
were later independently reviewed and retrospectively admitted under the
completion contract. This admission does not rewrite authorization history.
Belem and recovery/adopted cells retain their own source-map routes and
checkpoint identities. No old queue is resumed by reporting consolidation.

## Exact numerical and provenance sources

The following files are linked rather than duplicated. Every current entry
and SHA-256 is in `evidence.json`. Tables include denominators, status, units,
coverage, all applicable metrics and method-output clocks.

- method_metrics.csv (historical source; not bundled): Every CPU/neural method by case and known/new support scope.
- seed_metrics.csv (historical source; not bundled): All 126 individual seed results, including tails and R².
- median_metrics.csv (historical source; not bundled): All 126 ensemble case/method/support-scope metric rows.
- seed_dispersion.csv (historical source; not bundled): 13,368 observation/model seed-dispersion rows.
- paired_comparisons.csv (historical source; not bundled): Exact shared-ID paired comparisons and conditional uncertainty.
- train_references.csv (historical source; not bundled): Train-only reference metrics and fallback counts.
- source_metrics.csv (historical source; not bundled): Source-stratified metrics.
- group_metrics.csv (historical source; not bundled): Structural-group metric support.
- input_support.csv (historical source; not bundled): Consumed-input collisions and train-only masks.
- distribution_shift.csv (historical source; not bundled): Train/test target and feature distributions.
- cell_lineage.csv (historical source; not bundled): 126 selection/refit/checkpoint identity routes.
- comparison_with_partial_reports.csv (historical source; not bundled): 10/12-case exact numerical reconciliation.
- manifest.json (historical source; not bundled): Producing source map, component pins, bootstrap and data-quality audits.
- independent_validation.json (historical source; not bundled): Full-grid independent numerical validation.
- ARTIFACT_INDEX.json (historical source; not bundled): Existing full-grid package inventory.

Historical status and supersession links: historical_reports.json (historical source; not bundled).
Completion of this fixed grid closes the experimental scope. New experiments require a new research amendment.
