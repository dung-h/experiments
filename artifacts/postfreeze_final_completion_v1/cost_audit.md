# Accuracy–cost completion audit

## Existing evidence

The strongest saved timing artifacts are:

- `artifacts/am1_phase4/v1/`: full raw-QASM-to-output B2 timing for 45 QPU IDs, Graph and MLP, with matched accuracy read from frozen predictions.
- `artifacts/am1_phase5_dense_b2/v1/`: matched Dense FP32/FP64 Graph and MLP timing for the declared exact30 population.
- `artifacts/am1_overhead_pilot/v1/accuracy_cost_comparisons.csv`: context-specific Graph/MLP and reference timing summaries, including canonical QCSim/Track-B references.

## Why a classical timing frontier was not run

The existing timing harness has a declared raw-input path for the two Ma–Li neural pipelines.  It does not expose the same parsing, preparation, model-loading and output boundary for fitted Random Forest, XGBoost and Ridge states on the identical exact30/45 populations.  Constructing that path would require a new timing contract and a new estimator-build decision, not a read-only completion of the existing benchmark.  The saved Phase 4 report explicitly says that a complete classical frontier remains unanswered.

Therefore the current thesis-safe claim remains: Graph-vs-MLP prediction cost was measured in a matched local environment, with no stable accuracy superiority established; a complete all-estimator Pareto frontier is not claimed.

## Status

`PARTIAL_BY_AVAILABLE_CONTRACT`

