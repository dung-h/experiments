# Current handoff: Quantum runtime estimator replications

Last updated: 2026-09-18
Repository: `https://github.com/dung-h/experiments.git`
Local checkout: `/home/server/Documents/quantum-runtime-estimator-replications`
Branch: `main`
Latest local commit: `3bc2d23 Add dated Osaka Kyoto calibration drift audit`

## What is complete

The repository is a reproducibility capsule for five separate tracks. Targets
are not pooled into one universal runtime column.

1. Ma–Li DAG replication and fake-snapshot pretrained/from-scratch adaptation.
2. Qonductor public-artifact metric recomputation and mapping audit.
3. CDAA/QCRE schedule-duration reproduction.
4. cuTensorNet `RUNTIME_EST` versus separate GPU clocks on the RTX 5070 Ti.
5. Azizov et al. transpilation-aware Aer noisy-simulator runtime (independent
   reproduction; P0 complete, P1/P2 pending).

The additional Ma–Li/QCRE proxy validation is complete for all 340 Osaka/Kyoto
rows. It uses current Qiskit 1.4.1 `FakeOsaka`/`FakeKyoto` targets, optimization
level 1 and `seed_transpiler=1234`; it is labelled `OUR_PROXY`, not historical
calibration recovery.

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
4. `artifacts/validation/mali_qcre_seed_sensitivity/MALI_QCRE_SEED_SENSITIVITY_REPORT.md` — three-seed stability.
5. `artifacts/validation/mali_snapshot_variability/SNAPSHOT_VARIABILITY_REPORT.md` — public temporal calibration audit.
6. `experiments/README.md` — commands for the validation scripts.
7. `artifacts/azizov_independent/P0_REPORT.md` — first independent
   transpilation-aware Aer reproduction and feature-block smoke ablation.

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
`tracks/azizov_independent/README.md`. P1 (all unique circuits with explicit
900-second censoring) and P2 (backend/family hold-outs and seed sensitivity)
remain pending. A bounded P1 subset is already archived: 504 successful rows
from 63 small circuits, with grouped baseline and source/compiled ablation.

The local-feasibility pass is also complete: 149/1,402 circuits (10.6%) passed
the conservative `width<=9`, `logical_ops<=4000`, `logical_depth<=4000` screen.
All 1,192 resulting backend/optimization rows completed successfully. The
eligible and rejected manifests plus raw runtime table are under
`artifacts/azizov_independent/`. Do not describe the 10.6% as a hardware
capacity theorem; it is the subset confirmed under this Aer/noise/version
protocol.

## Claims to preserve

- Do not call the FakeOsaka/FakeKyoto proxy historical calibration truth.
- Do not call the Ma–Li fake-snapshot adaptation a reproduction of the paper's
  exact historical `total_ibm_standardization.npy`/calibration snapshot. Public
  Osaka/Kyoto candidate captures exist, but they have not been matched to the
  authors' tensor or label-collection timestamp.
- Do not call QCRE/CDAA schedule duration observed QPU wall-clock.
- Do not call cuTensorNet `RUNTIME_EST` a prediction of first-call or complete
  end-to-end latency; first, warm and end-to-end clocks are separate.
- Do not pool simulator, QPU, cloud workflow and service-runtime targets.

## Next recommended work

Build the multivariate runtime-estimator pilot using the same grouped and
backend-held-out protocols: compare median, ridge/log-linear, random forest and
histogram gradient boosting on the compiled features, and report original-scale
and log-scale metrics. Keep Qonductor as a separate job-estimator track until a
public stable join key is available.
