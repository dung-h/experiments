# Quantum Runtime Estimation Benchmark

Systematic comparison of quantum circuit runtime estimators, with explicit
runtime semantics, input qualifications and comparable held-out evaluations.
The contribution is a benchmark and its findings, not a universal Graph model.

**Scientific verdict: SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS.**
Experiments are closed. Start with [thesis synthesis](docs/thesis/README.md),
[canonical tables](docs/thesis/canonical_tables.md) or [results by RQ](results/README.md).

## Tracks and runtime semantics

| Track | Target and evaluation population |
| --- | --- |
| A: Real QPU | Source-retained circuit-level observed execution/service time. 4,515 qualified observations: Ma–Li 340, Qonductor 230, QPack 3,945; 165 structural groups. |
| B: Simulator | Runtime within a specified engine, hardware, precision, API and quality context. Predictor core: 150 hashes at 2–9 qubits; measured panel: 204 circuit members / 191 hashes through 16 qubits. |

QPU service time is not queue time, scheduled duration, QPU usage or workflow
turnaround. Simulator sample, state extraction, engine-reported time,
contraction and analog emulation have different boundaries. No cross-context leaderboard.

## Research questions

1. **RQ1:** Predictive accuracy and comparative effectiveness.
2. **RQ2:** Generalization beyond training/calibration distributions.
3. **RQ3:** Error behavior, robustness and failure modes.
4. **RQ4:** Accuracy–cost trade-off and scalability.
5. **RQ5:** Method fidelity, applicability and practical selection.

[Final closure](results/research_questions/closure/final_rq_status.md) keeps
RQ2 partially answered and non-blocking; RQ4 is answered within measured scope.

## Strongest validated findings

- QPU pooled MAE: Graph 0.8341 s, MLP 0.8934 s, selector 0.8867 s.
  Competitive paired uncertainty is unresolved; weighting changes rankings.
- Full generalization grid: 21 CPU and 21 neural cases, 126 seed cells.
  QNN/Random overprediction and Kyoto input collisions limit transfer claims.
- QPU exact45: Graph warm full-B2 costs about 2.22× MLP without an
  established paired accuracy advantage.
- Dense FP32/FP64: MLP has lower MAE and B2 on exact30 subsets,
  not a universal advantage or replacement for full150 accuracy.
- Each frozen MPS c44 selector violates the fidelity gate on 64/140 assessed
  choices. Prediction accuracy alone does not establish safe approximation.

## Repository map

| Location | Use |
| --- | --- |
| [docs](docs/README.md) | Canonical thesis, dataset/methodology guides, historical-package index |
| [results](results/README.md) | Comparisons and supporting evidence by RQ |
| [data](data/README.md) | Own simulator evidence; no private QPU rows |
| [methods](methods/README.md) | Implementations and historical support with prerequisites |
| [provenance](provenance/README.md) | Source/published hashes and export boundary |
| [tests](tests/README.md) | Public saved-evidence checks, not independent refits |

## Public reproduction boundary

```bash
python3 -B scripts/verify.py
python3 -B scripts/check_qpu_results.py
python3 -B scripts/check_simulator_results.py
python3 -B scripts/check_completed_science.py
python3 -B -m unittest discover -s tests
```

These checks validate files, QPU aggregates and retained simulator scores/timings.
They do not train models, run inference or execute simulators. Private QPU
labels, row derivatives, splits, checkpoints and source QASM are not bundled.
Historical runners need external inputs/environments; see [reproduction](docs/reproduction.md)
and [data availability](docs/distribution.md).

Published scientific baseline: `08ae5ac593a217678ec1015e093a78b59ac1a8af` on
`benchmark-review`. Navigation changes are pending review.
[Metadata TODO](docs/metadata_todo.md) lists unresolved license/citation fields.
