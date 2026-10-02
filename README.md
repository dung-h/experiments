# Quantum Runtime Estimation Benchmark

This project compares methods for predicting quantum circuit runtime. It
uses recorded QPU execution times and simulator measurements collected on
one local machine. Results are reported separately because these timings
measure different parts of execution.

## Start here

- [Slides and figure assets](presentations/README.md): PowerPoint, PDF,
  source-backed charts and presentation sources on this branch.
- [Methodology](docs/methodology.md): datasets, circuit reconstruction,
  train/test splits, and differences from the original papers.
- [Results](docs/results.md): comparisons, findings, and unfinished evaluations.
- [Method-by-dataset table](artifacts/benchmark_v3/results/reader_method_dataset_table.csv): compact cross-track view with target/output clocks, coverage, and method status.
- [Simulator predictor aggregate](artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json): separate Aer and fixed-MPS OOF scores, coverage, paired intervals, and method cards.
- [Reproduction](docs/reproduction.md): commands to rebuild the tables from
  the included predictions and measurements.
- [Numerical tables](artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md):
  detailed results and links to their source files.

| Domain | Data | Evaluation so far |
| --- | --- | --- |
| Real QPU | 8,767 recorded observations: Ma–Li 340, Qonductor 4,482, QPack 3,945 | Unified polynomial and graph adaptations use the same grouped train/test split. Analytical estimates are also compared with recorded service time. |
| Simulator | 204 circuit entries, 191 distinct QASM hashes, 2–16 qubits | Local measurements cover Aer, CUDA-Q, and cuTensorNet. Predictor OOF evaluations now cover a 150-hash q≤9 Aer core and a separate fixed CUDA-Q MPS cell; broader paper-wide coverage remains incomplete. |

## Main findings and limitations

The evaluated V3-large graph-and-metadata model has lower MAE than the
polynomial model on their shared test rows in each QPU source. Both have negative R² on
QPack. Three very large circuits still produce substantial errors, even
after extending the graph representation's resource limit. These results
describe our adaptations, not exact reproductions of the original models.

Scheduled circuit duration does not match recorded QPU service time.
Simulator results also depend on which stage is timed, precision and, for
approximate methods, output quality. The Maestro calibration pilot failed
its timing-stability checks and has no accepted prediction score. The V4
control-flow representation has not been trained.

On the local Aer core, an Azizov-style transpiled-view GNN adaptation reached
0.3008 s MAE, while a transpiled-view XGBoost baseline reached 0.2970 s; their
paired interval does not resolve a difference. On the separate fixed-MPS
cell, a source-DAG graph adaptation reached 0.3024 s MAE over 144 finite
targets (142 passed the quality gate). These are configuration-specific
adaptation results, not cross-engine rankings or paper-exact reproductions.
The [results page](docs/results.md#local-simulator-predictors) explains the
targets, coverage and uncertainty.

QPack circuit structure is reconstructed; its original optimized angles and
submitted circuits were not recovered. Nominal backend snapshots are not
historical job-day calibration. [Methodology](docs/methodology.md) explains
these assumptions and their consequences.

## Reproduction and publication status

The complete review package at commit `860829a` passed a fresh local clean-clone
table rebuild with a newly installed pinned verification environment. The
validation record distinguishes that check from retraining or repeating
hardware measurements: original-source extraction and every paper's training
are not reproduced. The complete original source circuit collections are not
included. See the [validation record](docs/reproduction_validation.md).

The current E1–E6 package is on `presentation-review`; `benchmark-review`
contains the earlier review package. Branch sharing is at the author's request.
Three large CSVs are stored as row-aligned parts smaller than 48 MiB. One
command restores the originals and checks their hashes; Git LFS is not
required. The reproduction guide includes the clone commands. This is not a
tagged release or a claim that every planned
benchmark is complete. Repository licensing and citation metadata remain
unselected; complete external source archives remain excluded. The
[earlier release decision](benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md)
records the remaining licensing and provenance questions. Publishing this
review branch does not change `CURRENT.json` or grant new data licenses. See
[third-party notices and unresolved rights](docs/third_party_notices.md)
before redistributing bundled datasets, snapshots, or derived artifacts.

The [scientific review and routine-agent handoff](docs/scientific_review.md)
lists the current method matrix, supported findings and remaining release work.

## Earlier studies

Earlier replication studies are retained for traceability. They use their
own datasets and timing definitions; their scores are not additional entries
in the current comparison.

- [Study findings](docs/FINDINGS_CLUSTERS.md) and [track results](RESULTS.md).
- [Historical reproduction guide](REPRODUCTION.md).
- [Work log, 17–22 September](docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md).
- [Pinned upstream revisions](upstream/upstream.lock.json).

Code is in `benchmark_v1/`, `tracks/` and `experiments/`; measurements,
predictions and reports are in `artifacts/`. Start with the documents above
rather than the historical task notes.
