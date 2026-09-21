# Current handoff: Quantum runtime estimator replications

Last updated: 2026-09-21
Repository: `https://github.com/dung-h/experiments.git`
Local checkout: `/home/server/Documents/quantum-runtime-estimator-replications`
Branch: `main`
Revision: inspect `git log -1 --oneline` and `git status --short` before a rerun;
this handoff intentionally does not hard-code a commit SHA.

## What is complete

The repository is a reproducibility capsule for five separate tracks. Targets
are not pooled into one universal runtime column.

1. Ma–Li DAG replication and fake-snapshot pretrained/from-scratch adaptation.
2. Qonductor public-artifact metric recomputation and mapping audit.
3. CDAA/QCRE schedule-duration reproduction.
4. cuTensorNet `RUNTIME_EST` versus separate GPU clocks on the RTX 5070 Ti.
5. Azizov et al. transpilation-aware Aer noisy-simulator runtime (independent
   reproduction; P0, bounded P1 and local-feasible split validation complete;
   full paper-scale P1 remains pending).
6. Matched dense-statevector runtime supplement: v1 four-context matrix and v2
   structural/feasibility extension. This is a fixed local PyTorch kernel, not
   a pooled simulator or QPU target.

The additional Ma–Li/QCRE proxy validation is complete for all 340 Osaka/Kyoto
rows. It uses current Qiskit 1.4.1 `FakeOsaka`/`FakeKyoto` targets, optimization
level 1 and `seed_transpiler=1234`; it is labelled `OUR_PROXY`, not historical
calibration recovery.

The follow-on direct source-specific estimator evaluation is also complete. It
uses the proxy's pre-run compiled features to predict the same observed Ma–Li
labels; it does not pool any other source or save a post-hoc selected deployment
model.

The dense-statevector v2 validation is complete. The canonical corpus has 208
rows (204 durations and four retained CUDA resource limits). Circuit-group
log-R² is 0.9794 for the logical HGB estimator; width holdout favors the
analytical Ridge (0.8754 versus 0.7295). A q24-calibrated peak-memory envelope
correctly rejects q28 complex128 while q28 complex64 completes. These numbers
are specific to the recorded RTX 5070 Ti/PyTorch contract.

## Numbers to remember

### Ma–Li/QCRE grouped validation

- 340 rows, 170 unique logical QASM hashes.
- 134 logical hashes are repeated across Osaka/Kyoto and are kept in one fold.
- Grouped five-fold, fit on train groups only:
  - logical depth R²: `-0.0346`;
  - physical depth R²: `0.7879`;
  - weighted target-duration path R²: `0.6692`;
  - physical depth + 2Q depth + weighted path R²: `0.7946`.
- `1024 × weighted_path` with grouped affine calibration: R² `0.7536`.
- Leave-one-backend-out:
  - Osaka → Kyoto: R² `0.6705`;
  - Kyoto → Osaka: R² `0.6798`.
- Median observed label: `8.637 s`; median raw weighted path: `0.00510 s`.
  The path needs calibration and is not a direct QPU wall-clock estimate.

### Ma–Li direct source-specific estimator

Finding recorded: `2026-09-19`.

- Same target: 340 observed Osaka/Kyoto `result.time_taken` rows at 1,024
  shots; queue excluded; 170 logical-QASM SHA-256 groups.
- Candidate inputs: logical structural features, post-transpile physical
  structure, weighted target-duration path and backend label. Inputs are
  reconstructed from current FakeBackends, so this is still `OUR_PROXY`.
- Aggregate OOF, grouped logical-QASM five-fold:
  - physical-depth Ridge: log R² `0.8246`, MAE `0.6769 s`;
  - compiled Ridge: log R² `0.8777`, MAE `0.5116 s`;
  - duration-proxy + compiled residual RF: log R² `0.8858`, MAE `0.5303 s`.
- Strict backend plus unseen-logical-circuit 2 × 5 evaluation:
  - physical-depth Ridge: log R² `0.8215`, MAE `0.6792 s`;
  - compiled Ridge: log R² `0.8742`, MAE `0.5224 s`;
  - residual RF: log R² `0.8753`, MAE `0.5565 s`.
- The paired backend-only diagnostic reaches `0.9052` for compiled Ridge, but
  is not primary evidence because it permits that logical QASM's other-backend
  copy in training.
- Family-held-out OOF is retained, not averaged per small family: residual RF
  log R² `0.8610`, MAE `0.5630 s`. Duration alone is not isolated as the cause
  of the improvement; earlier matched ablation found only a small incremental
  weighted-path gain.

### Calibration snapshot variability

Finding recorded: `2026-09-18`.

- Public DAQEC series: 42 timestamps/backend over 14 days, three rows per
  timestamp.
- Osaka aggregate T1 range/median: `63.6%`; T2: `50.3%`; CVs `17.3%`/`12.9%`.
- Kyoto aggregate T1 range/median: `53.0%`; T2: `48.6%`; CVs `16.8%`/`12.8%`.
- This is a 2025 aggregate series, not the exact per-qubit Ma–Li tensor.
- Report, raw CSV, manifest and rerun instructions:
  `artifacts/validation/mali_snapshot_variability/`.

### Seed sensitivity

All 340 rows were rerun at seeds `1234`, `2025`, `31415`, using the same
grouped folds and Qiskit/FakeBackend configuration.

- Physical depth R²: mean `0.7847`, std `0.0059`, range `0.7764–0.7897`.
- Physical 2Q depth R²: mean `0.7720`, std `0.0045`.
- Weighted path R²: mean `0.6628`, std `0.0138`, range `0.6437–0.6756`.

### Ma–Li transfer outlier already documented

The seed-1234 fold-10 pretrained adaptation fails on
`qwalk-noancilla_indep_qiskit_9` (Osaka): actual `13.6962 s`, pretrained
prediction `36.0312 s`, from-scratch prediction `12.8724 s`, fold R² `-1.8991`.
The exact retrieval fixture is `artifacts/mali/fold10_qwalk_outlier.json`.

### Qonductor limitation

The public resource CSV has 100 rows with only `predicted`, `real`, `dag`.
The separate database has 7,449 jobs and 166,093 circuits, but no shared
`job_id`, `circuit_id` or `ibm_quantum_id`. Do not join by row order or nearest
runtime. This is a reproducibility limitation, not a failed model experiment.

## Where to read first next session

1. `RESULTS.md` — consolidated findings and claim boundaries.
2. `REPRODUCTION.md` — clean-clone and rerun commands.
3. `artifacts/validation/mali_qcre_final/MALI_QCRE_FINAL_VALIDATION_REPORT.md` — grouped split, duration audit, affine calibration and ablations.
4. `artifacts/validation/mali_direct_estimator/MALI_DIRECT_COMPILED_ESTIMATOR_REPORT.md` — direct estimator, split boundaries and full OOF metrics.
5. `artifacts/validation/mali_qcre_seed_sensitivity/MALI_QCRE_SEED_SENSITIVITY_REPORT.md` — three-seed stability.
6. `artifacts/validation/mali_snapshot_variability/SNAPSHOT_VARIABILITY_REPORT.md` — public temporal calibration audit.
7. `experiments/README.md` — commands for the validation scripts.
8. `artifacts/azizov_independent/P0_REPORT.md` — first independent
   transpilation-aware Aer reproduction and feature-block smoke ablation.
9. `artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/REPORT.md`
   — fixed-kernel estimator, extrapolation and GPU-feasibility findings.

For next-week reporting, read
`artifacts/mali/MA_LI_QWALK_AND_QCRE_INTERPRETATION.md` immediately after
`RESULTS.md`. It records the QWalk circuit structure and fold-10 failure,
exactly how Ma–Li's DAG/T1/T2 metadata enters inference, the standardized-mean
intervention, and the distinction between the Ma–Li GNN and the separate QCRE
compiled-feature proxy. Do not merge the two inference paths in the report.

The machine-readable outputs are:

- `artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv`;
- `artifacts/validation/mali_qcre_final/mali_qcre_grouped_validation_rows.csv`;
- `artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_rows.csv`;
- `artifacts/validation/mali_direct_estimator/mali_direct_estimator_oof_predictions.csv`;
- `artifacts/validation/mali_direct_estimator/mali_direct_estimator_summary.csv`;
- `artifacts/validation/mali_direct_estimator/mali_direct_estimator_strict_backend_metrics.csv`;
- the corresponding `*_summary.json` files.

## Exact rerun commands

From the capsule root, after `bash scripts/bootstrap_upstreams.sh work` and a
Qiskit 1.4.1 environment:

```bash
python3 scripts/verify_artifacts.py
python experiments/mali_qcre_validation.py --mali-root work/mali
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --seeds 1234,2025,31415 --optimization-level 1
python experiments/qonductor_mapping_audit.py --qonductor-root work/qonductor
python3 -m venv .venv-mali-direct
.venv-mali-direct/bin/pip install -r experiments/requirements-mali-direct-estimator.txt
.venv-mali-direct/bin/python experiments/run_mali_direct_estimator.py
```

The seed command retranspiles 340 rows per seed and is the slow step. To check
only report generation, pass the committed `--features-csv` or `--rows-csv`
artifacts as documented in `experiments/README.md`.

## Azizov et al. P0 status

The independent track uses the public Ma–Li QASM pool and current
`FakeWashingtonV2`/`FakeSherbrooke` snapshots. P0 has 168 successful rows
(21 small families × 2 backends × 4 optimization levels), with 1,024 shots and
separate transpile/execution clocks. The grouped Ridge smoke score is
`R²_log=0.758`; compiled global features outperform source-only features for
Ridge on this small split (`0.8249` versus `0.7764` in log-R²). These are local
diagnostics, not the paper's exact HPC metrics.

Rerun commands and artifact boundaries are in
`tracks/azizov_independent/README.md`. Full P1 (all unique circuits with
explicit 900-second censoring) and repeated-seed robustness remain pending. A
bounded P1 subset is archived: 504 successful rows from 63 small circuits,
with grouped baseline and source/compiled ablation. The larger local-feasible
subset has a separate split evaluation covering grouped, family-held-out and
backend-held-out tests.

The local-feasibility pass is also complete: 149/1,402 circuits (10.6%) passed
the conservative `width<=9`, `logical_ops<=4000`, `logical_depth<=4000` screen.
All 1,192 resulting backend/optimization rows completed successfully. The
eligible and rejected manifests plus raw runtime table are under
`artifacts/azizov_independent/`. A follow-up split evaluation now covers
random-row, grouped-circuit, family-held-out and both fake-backend-held-out
protocols; the grouped-circuit result is the primary local estimate. Do not
describe the 10.6% as a hardware capacity theorem; it is the subset confirmed
under this Aer/noise/version protocol.

A process-isolated width-10--16 resource canary was attempted after the q<=9
matrix. It retained seven exploratory rows before being stopped: four
FakeWashington rows hit a 16 GiB RSS safety cap, while DJ q16 and GHZ q10/q11
completed. Keep this partial probe separate from model-training data; it is
evidence that circuit structure and Aer parallelism matter beyond logical
width. The report and raw rows are under
`artifacts/azizov_independent/Q10_Q16_RESOURCE_CANARY_PARTIAL_REPORT.md`.
Raising the safety cap to 36 GiB on `ae_q10`/`ae_q12` still produced wall-time
censoring at about 16.8 GiB RSS, so do not equate more RAM with a larger
feasible noisy-Aer frontier. The targeted check is recorded in
`artifacts/azizov_independent/Q10_Q16_HIGHMEM_CHECK_REPORT.md`.

## Claims to preserve

- Do not call the FakeOsaka/FakeKyoto proxy historical calibration truth.
- Do not call the Ma–Li fake-snapshot adaptation a reproduction of the paper's
  exact historical `total_ibm_standardization.npy`/calibration snapshot. Public
  Osaka/Kyoto candidate captures exist, but they have not been matched to the
  authors' tensor or label-collection timestamp.
- Do not call QCRE/CDAA schedule duration observed QPU wall-clock.
- Do not call the Ma–Li direct estimator a historical-QPU model or a universal
  runtime estimator: its compiled inputs use the current FakeBackend proxy and
  its two backend directions are not broad hardware validation.
- Do not call cuTensorNet `RUNTIME_EST` a prediction of first-call or complete
  end-to-end latency; first, warm and end-to-end clocks are separate.
- Do not pool simulator, QPU, cloud workflow and service-runtime targets.
- Do not present the dense-statevector v1/v2 scores as a cross-framework or
  cross-GPU simulator estimator; they are a calibrated local-kernel study.

## Next recommended work

Freeze the Ma–Li model-selection rule before fitting any operational model, then
seek a second real-QPU source with circuit-level join keys or historical
calibration snapshots. Keep Qonductor as a separate job-estimator track until a
public stable join key is available. Simulator studies, including bond-dimension
work, remain controlled pretraining/ablation tracks rather than substitutes for
the observed-QPU estimator.
