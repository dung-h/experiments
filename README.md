# Quantum Runtime Estimator: replication studies

This repository is a reproducibility capsule. It records code, patches,
expected outputs and failure analysis for distinct runtime-estimation
studies. It does not merge their targets into one training table.

Read the work as **three finding clusters**, then as older capsules.

The cluster report is [`docs/FINDINGS_CLUSTERS.md`](docs/FINDINGS_CLUSTERS.md).
Track-level tables remain in [`RESULTS.md`](RESULTS.md).
[`REPRODUCTION.md`](REPRODUCTION.md) is the clean-clone workflow.
[`HANDOFF.md`](HANDOFF.md) is the next-session status.

## Finding clusters

| Cluster | Estimator under study | Evidence | Central finding |
| --- | --- | --- | --- |
| Ma–Li × Azizov | Compiled / transpiled prediction of Ma–Li Osaka/Kyoto `result.time_taken` | Logical DAG reproduction, current-FakeBackend compiled proxy, native DAG, family-OOD, compiled Ridge and coarsened-DAG transfer | Logical depth is not enough. After transpile, compiled structure carries the signal. Simulator pretraining plus an affine head does not beat training on the 340 hardware rows. QWalk is the failure tail. Current FakeOsaka/Kyoto snapshots are not historical job-day calibration. |
| Quantum Rings solutions | Spirit Sprinters graph-Transformer; SoftLocked tabular GradientBoosting | Public-artifact audit on the 144-row 0.99-forward table | Winner log-R² 0.743. SoftLocked is in-domain on its own table (R² 0.967) and a domain-shift failure here (log-R² −25.5). The contest does not publish the simulation machine. |
| Family-aware paper ([arXiv:2606.11620](https://arxiv.org/abs/2606.11620)) | Family-conditioned residual/FiLM MLP reconstructed on the public contest table | Circuit-level 5-fold CV, tree baselines, frozen public MQT classifier | Trees are the strongest public runtime baseline (0.75 log-R² 0.757; 0.99 ExtraTrees 0.712). The reconstructed family MLP is not competitive for runtime. Paper R² 0.82 is not recovered. 0.75 mirror-sweep time and 0.99 forward time must not be pooled. |

Do not pool Ma–Li `result.time_taken`, contest 0.75 mirror time, contest 0.99
forward time, SoftLocked's 74k-second table, or local cuTensorNet GPU times.

Cluster paths:

- Cluster 1 reports: [`artifacts/validation/mali_azizov_compiled_transfer_v1/`](artifacts/validation/mali_azizov_compiled_transfer_v1/), [`mali_azizov_transpiled_dag_v1/`](artifacts/validation/mali_azizov_transpiled_dag_v1/), [`mali_family_ood_v1/`](artifacts/validation/mali_family_ood_v1/), [`mali_physical_dag_v1/`](artifacts/validation/mali_physical_dag_v1/)
- Cluster 2 report: [`artifacts/quantum_rings/REPORT.md`](artifacts/quantum_rings/REPORT.md)
- Cluster 3 report: [`artifacts/quantum_rings/family_aware_paper/`](artifacts/quantum_rings/family_aware_paper/)

Day-by-day record: [`docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md`](docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md).
Upstream pins: [`upstream/upstream.lock.json`](upstream/upstream.lock.json).

## Other capsules

These remain in the repository and are not part of the three-cluster
reorganisation:

| Capsule | Target | Note |
| --- | --- | --- |
| Qonductor | One-job execution estimate, queue excluded | Public regression predictions beat the numerical DAG baseline on the supplied 100-row CSV. No stable join from the resource-estimator CSV to the circuit/job database. |
| CDAA/QCRE | Compiled-circuit schedule duration | Gate-aware depth follows the artifact's schedule reference. That reference is not measured QPU wall-clock. |
| cuTensorNet | RTX 5070 Ti contraction `RUNTIME_EST` | 70% 25-row over-predicts cheap kernels; 90% QFT q36–q44 under-predicts 4.19–5.11×. Plan-cost proxy, not wall-clock. |
| Azizov independent Aer screen | Local noisy Aer `T_exec` after transpile | 149/1,402 circuits pass the conservative local screen; 1,192 tested rows complete. Not the unpublished paper table. |
| Dense-statevector / CUDA-Q | Fixed local simulator kernels | Separate clocks. Not pooled with QPU or Quantum Rings labels. |

## Scope

- Ma–Li × Azizov: simulator `time_taken` during pretraining; recorded Osaka/Kyoto `result.time_taken` during hardware evaluation. Compiled features are a current FakeBackend proxy.
- Quantum Rings solutions and family-aware paper: public contest labels. CPU/GPU is a tag, not a host profile. No SDK rerun.
- Qonductor: archived one-job execution estimate.
- CDAA/QCRE: analytical duration from archived instruction-duration snapshots.
- cuTensorNet: local GPU scalar contraction; warm CUDA-event time is distinct from first-call end-to-end latency. The 90% QFT frontier is a later workspace policy, not a replacement of the 70% 25-row/55-row tables.

The repository excludes live QPU jobs, cloud ETL, virtual environments,
checkpoints, coarsened native-DAG tensors, and upstream source trees.
Upstream sources are cloned on demand at pinned commits.

## Repository layout

```text
docs/FINDINGS_CLUSTERS.md   three-cluster reading path
tracks/                     overlays, clone scripts, vendored family-aware code
artifacts/                  committed reports, metrics and fixtures
experiments/                Ma–Li / Azizov / Qonductor validation scripts
scripts/                    bootstrap and integrity verification
upstream/                   pinned revisions and data boundaries
RESULTS.md                  track-level tables
REPRODUCTION.md             clean-clone procedure
```

```bash
python3 scripts/verify_artifacts.py
```

That check uses the Python standard library. It verifies the committed
capsule. It does not rerun GPU work, retrain models, submit a cloud job or
recreate historical QPU calibration state.
