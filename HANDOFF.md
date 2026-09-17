# Current handoff: Quantum runtime estimator replications

Last updated: 2026-09-17
Repository: `https://github.com/dung-h/experiments.git`
Local checkout: `/home/server/Documents/quantum-runtime-estimator-replications`
Branch: `main`
Latest local commit: `f4ca6db Report explicit backend transfer directions`

## What is complete

The repository is a reproducibility capsule for four separate tracks. Targets
are not pooled into one universal runtime column.

1. Ma–Li DAG replication and fake-snapshot pretrained/from-scratch adaptation.
2. Qonductor public-artifact metric recomputation and mapping audit.
3. CDAA/QCRE schedule-duration reproduction.
4. cuTensorNet `RUNTIME_EST` versus separate GPU clocks on the RTX 5070 Ti.

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
5. `experiments/README.md` — commands for the validation scripts.

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

## Claims to preserve

- Do not call the FakeOsaka/FakeKyoto proxy historical calibration truth.
- Do not call the Ma–Li fake-snapshot adaptation a reproduction of the paper's
  missing historical `total_ibm_standardization.npy`/calibration snapshot.
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
