# Quantum Runtime Estimator: five replication studies

This repository is a reproducibility capsule for five distinct runtime-
estimation studies. It records our code, patches, expected outputs and failure
analysis; it does not merge their targets into one training table or claim a
universal estimator.

| Track | Estimator under study | Evidence class | Central finding |
| --- | --- | --- | --- |
| Ma–Li | Graph Transformer over circuit DAG; direct compiled-feature regression | Local reproduction plus fake-snapshot, grouped QCRE proxy and direct source-specific validation | Simulator-pretrained initialization fails on one held-out QWalk fold; a separately evaluated direct compiled-feature estimator improves on physical depth under QASM-grouped tests, but remains a current-FakeBackend proxy. |
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

An additional, separately labelled simulator-data-generation supplement covers
VQCSim, the independent Azizov reconstruction, and the credential-free local
part of the Zero-Setup artifact. Start with
[`experiments/simulator_papers/README.md`](experiments/simulator_papers/README.md)
and its [initial report](artifacts/simulator_papers/INITIAL_LOCAL_REPRODUCTION_REPORT_2026-09-20.md).
The completed estimator-validation pass, raw-artifact map and failure analysis
are in [`SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md`](artifacts/simulator_papers/SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md).
Its [`deep-analysis companion`](artifacts/simulator_papers/DEEP_ANALYSIS_2026-09-20.md)
records the observed residual patterns and the bounded technical explanations.
It is not a sixth pooled runtime target.

The separately generated [matched dense-statevector matrix](experiments/simulator_runtime_v1/README.md)
adds 96 actual local execution labels across CPU/GPU and complex64/complex128
under one PyTorch reference-kernel contract. Its [report](artifacts/simulator_runtime_v1/torch_dense_statevector_v1/REPORT.md)
records the target boundary, raw measurements, baseline OOF scores and the
remaining device/precision transfer failure mode. It is not pooled with Aer,
CUDA-Q or cuTensorNet.

The [structural v2 follow-up](experiments/simulator_runtime_v2/README.md)
extends that one kernel with seeded random topologies and a q26/q28 GPU-memory
frontier. Its [final report](artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/REPORT.md)
keeps 4 CUDA out-of-memory observations as `resource_limit` rather than
inventing duration labels. It finds strong in-range circuit-group prediction,
weaker width extrapolation, and a q24-calibrated peak-memory envelope that
correctly rejects q28 complex128 despite raw statevector bytes fitting VRAM.

The literature and public-artifact audit is [`docs/PAPER_REPO_EXPERIMENT_MAP_2026-09-21.md`](docs/PAPER_REPO_EXPERIMENT_MAP_2026-09-21.md).
It records which papers have usable code/data, what timing contract each one
defines, and why a candidate is or is not selected for a new experiment. The
first CUDA-Q matched simulator pilot is under
[`artifacts/cudaq_runtime/cudaq_matrix_20260921/`](artifacts/cudaq_runtime/cudaq_matrix_20260921/).
It keeps CPU QPP FP64, GPU FP32 and GPU FP64 `sample` wall-clock labels
separate, with first-call and warm-call timing reported independently.
The follow-up [CUDA-Q MPS pilot](artifacts/cudaq_runtime/cudaq_mps_20260921/REPORT.md)
adds configured bond-cap, observed tensor-bond and fidelity diagnostics while
keeping post-run bond observations out of the static estimator feature block.

## Scope

Each track has its own target semantics:

- Ma–Li: simulator labels during pretraining and recorded Osaka/Kyoto
  `result.time_taken` labels during adaptation/direct source-specific regression;
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
