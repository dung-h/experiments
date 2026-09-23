# Current handoff: Quantum runtime estimator replications

Last updated: 2026-09-23
Repository: `https://github.com/dung-h/experiments.git`
Local checkout: `/home/server/Documents/quantum-runtime-estimator-replications`
Branch: `main`

Inspect `git log -1 --oneline` and `git status --short` before a rerun.
This handoff does not hard-code a commit SHA.

## What a reviewer should read

1. [`docs/FINDINGS_CLUSTERS.md`](docs/FINDINGS_CLUSTERS.md) — three clusters
2. [`README.md`](README.md) — layout and other capsules
3. [`RESULTS.md`](RESULTS.md) — track-level tables
4. [`REPRODUCTION.md`](REPRODUCTION.md) — clone and rerun
5. `python3 scripts/verify_artifacts.py`

The three clusters are:

1. **Ma–Li × Azizov** — compiled / transpiled prediction of Osaka/Kyoto
   `result.time_taken`. Logical depth fails; compiled structure carries the
   signal; simulator affine transfer does not beat QPU scratch; QWalk is the
   tail. Current FakeBackend snapshots are not historical calibration.
2. **Quantum Rings solutions** — Spirit Sprinters public log-R² 0.743;
   SoftLocked in-domain R² 0.967 and public-table log-R² −25.5. No machine
   profile in the contest labels.
3. **Family-aware paper** — public reconstruction of arXiv:2606.11620.
   Trees beat the reconstructed family MLP on runtime. Paper R² 0.82 is not
   recovered. 0.75 proxy and 0.99 forward are different clocks.

Do not push research commits to `iQuHACK/2026-Quantum-Rings`. That remote is
read-only for this project.

## What is packaged in the three clusters

Ma–Li / Azizov artifacts under `artifacts/validation/mali_*` and
`artifacts/mali/`, including compiled transfer, transpiled DAG (without
`coarsened/`), family-OOD A/B2/C, physical-DAG ablations, and the original
seed-1234 QWalk fixture.

Quantum Rings solution audit under `artifacts/quantum_rings/`.

Family-aware reports, JSON and frozen MQT classifier under
`artifacts/quantum_rings/family_aware_paper/`. Scripts are vendored in
`tracks/quantum_rings/family_aware_paper/experiments/`. Public JSON/QASM are
cloned with `bash tracks/quantum_rings/clone_upstreams.sh work`.

## Other capsules already in the repository

Qonductor, CDAA/QCRE, cuTensorNet 25-row, 55-row 70% frontier and 90% QFT width frontier, Azizov
independent Aer screen, dense-statevector v1/v2, CUDA-Q dense and MPS
pilots. These are not part of the three-cluster reorganisation.

## Packaged in this cuTensorNet capsule update

- `artifacts/cutensornet/runtime_est_workspace90_frontier_v1_20260923/`

## Intentionally not in this packaging round

Working-tree follow-ups that remain local:

- `artifacts/simulator_runtime_v2/dense_statevector_cpu_eval_20260922/`
- `artifacts/cutensornet/multi_target_calibration_v1/`
- `artifacts/cutensornet/runtime_est_trace_v1_20260923/`
- `artifacts/paired_simulator_v1/`
- `artifacts/cudaq_runtime/launch_scaling_v1_20260922/`

Live IBM jobs and other credential-gated plans were removed. See
[`docs/REMOVED_AND_OUT_OF_SCOPE.md`](docs/REMOVED_AND_OUT_OF_SCOPE.md).

## Boundaries that must not be blurred

- FakeOsaka / FakeKyoto / FakeWashingtonV2 / FakeSherbrooke snapshots used
  here are not the historical job-day calibrations of the Ma–Li CSVs.
- Ma–Li `result.time_taken` is not contest 0.99 forward time and is not
  SoftLocked's 74k-second table.
- `mali_azizov_transpiled_dag_v1/qpu_scratch_pilot/` used a `softplus`
  decode; ignore it. The full eval used `expm1`.
- Cluster 3 does not recover the paper's private 0.75 forward labels or the
  authors' family-pretraining circuits.
