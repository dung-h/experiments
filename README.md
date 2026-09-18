# Quantum Runtime Estimator: five replication studies

This repository is a reproducibility capsule for five distinct runtime-
estimation studies. It records our code, patches, expected outputs and failure
analysis; it does not merge their targets into one training table or claim a
universal estimator.

| Track | Estimator under study | Evidence class | Central finding |
| --- | --- | --- | --- |
| Ma–Li | Graph Transformer over circuit DAG | Local reproduction plus fake-snapshot and grouped QCRE proxy validation | The simulator-pretrained initialization fails catastrophically on one held-out QWalk fold, while compiled physical structure remains stable across transpiler seeds in the separate proxy check. |
| Qonductor | Job/circuit regression in a cloud scheduler | Metric recomputation plus local fake-backend smoke | The public regression predictions outperform the numerical DAG baseline on the supplied 100-row evaluation CSV. |
| CDAA/QCRE | Gate-aware compiled-circuit duration estimate | Bundled-artifact reproduction plus offline compiler proxy | Gate-aware depth closely follows the artifact's schedule-duration reference, but the reference is not measured QPU wall-clock. |
| cuTensorNet | NVIDIA pre-run contraction-path `RUNTIME_EST` | Local RTX 5070 Ti measurement | The estimate is conservative for easy warm contractions and much closer for hard QFT contractions. |
| Azizov et al. | Transpilation-aware Aer noisy-simulation runtime | Independent P0/P1 plus screened local-feasibility matrix; paper code/data not yet public | 149/1,402 circuits pass the conservative local screen and all 1,192 tested backend/optimization rows complete. |

The canonical comparative report is [`RESULTS.md`](RESULTS.md). It gives the
metric, provenance class, failure boundary and interpretation for every track.
[`REPRODUCTION.md`](REPRODUCTION.md) documents the clean-clone workflow. The
next-session status and handoff are kept in [`HANDOFF.md`](HANDOFF.md). The
exact upstream revisions are pinned in
[`upstream/upstream.lock.json`](upstream/upstream.lock.json).

## Scope

Each track has its own target semantics:

- Ma–Li: simulator labels during pretraining and recorded Osaka/Kyoto
  `result.time_taken` labels during adaptation;
- Qonductor: Qonductor one-job execution estimate, excluding queue waiting and
  workflow job-completion time;
- CDAA/QCRE: analytical duration derived from compiled circuits and archived
  IBM instruction-duration snapshots;
- cuTensorNet: local GPU scalar tensor-network contraction, with warm CUDA
  event time distinct from first-call end-to-end latency.
- Azizov et al.: Qiskit Aer noisy-simulator execution time after transpilation,
  with transpile time and simulator setup kept separate.

The repository intentionally excludes QPU/cloud ETL, Aer CPU resource-aware
P0–P7 experiments, generated visualization HTML, virtual environments,
checkpoints and upstream source trees. Upstream sources are cloned on demand
at pinned commits and overlaid only with the changes recorded here.

## Repository layout

```text
tracks/                 code, overlays and track-specific instructions
artifacts/              committed reports, metrics and expected fixtures
experiments/            cross-track validation scripts (Qonductor/Ma–Li)
scripts/                bootstrap and integrity verification
upstream/               upstream revisions, licenses and data boundaries
RESULTS.md              report of findings and failure modes
REPRODUCTION.md         clean-clone reproduction procedure
```

The first reproducibility check requires only the Python standard library:

```bash
python3 scripts/verify_artifacts.py
```

It verifies the committed capsule. It does not rerun GPU work, submit a cloud
job or claim to recreate historical QPU calibration state.
