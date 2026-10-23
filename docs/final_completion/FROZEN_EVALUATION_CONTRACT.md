# Frozen Evaluation Contract

This document is the interpretation contract for final-completion analyses.  It does not change Scientific Freeze v2, existing folds, model hyperparameters, or historical results.

## Track A target and unit

The primary Track-A target is archived observed QPU service/execution time in seconds (`target_seconds`), with the source-specific provider boundary documented in the benchmark reports.  One row is an observation/context record; the primary resampling unit is a conservative structural group, not an arbitrary row.  The frozen panel contains 4,515 rows and 165 groups: Ma–Li 340/148 groups, Qonductor 230/11 groups, and QPack 3,945/6 groups.

## Metric hierarchy

1. **Primary:** group-balanced MAE.  Each structural group contributes equal weight after its row-level absolute errors are averaged.
2. **Secondary:** pooled MAE, source-macro/source-balanced MAE, RMSE, median absolute error, R² where variance is meaningful, P90/P99 absolute error, maximum absolute error, coverage when an interval exists, and signed error for bias diagnostics.

For paired comparisons the primary inferential quantity is `metric(treatment) - metric(control)`, with the sign stated explicitly (negative means treatment is lower).  Group/hash bootstrap is used when the group/hash is the independence unit.  A confidence interval crossing zero is not evidence of equivalence.

## Transfer contract

The A1 transfer comparison uses the same five outer folds, same 4,515 QPU rows/165 groups, same three model seeds (42, 1234, 31415), strict simulator-identity exclusion, `FIXED_RAW_51` for MLP, and `CIRCUIT_ONLY_DAG + FIXED_RAW_51` for Graph+MLP.  Scratch and strict-pretrained treatments share architecture, preprocessing, training budget and held-out test observations; initialization is the intended treatment difference.  Sample-efficiency extensions subsample complete outer-training structural groups only, with nested deterministic subsets and no test-based selection.

## Scope boundaries

- **Material shot-conditioning limitation:** frozen Track-A Graph+MLP/MLP and A1/P1 circuit-only encoders do not receive shots. Mixed-shot OOF accuracy does not establish runtime prediction conditional on query shots or generalization across shots. Identical complete inputs yield the same prediction even if the requested shot count changes. Regressors that receive shots have different information access; their comparison is a pipeline comparison. Evidence: `benchmark_v1/protocol/mali_full_features.json:103`, `artifacts/real_qpu/qualified_logical_panel/summary/README.md:19`, and `docs/track_a_transfer_study/U_GRAPH_V3_INFORMATION_PARITY.md:15`.
- P1 addresses simulator initialization under limited QPU supervision with this fixed input contract. P2 connectivity attribution, if reconciled or executed, remains conditional on the same absence of shot input. Neither study establishes shot-conditioned transfer or connectivity benefit for a fully context-conditioned runtime estimator. No result or running treatment is changed by this reporting qualification.
- Graph+MLP versus MLP-only is a whole-pipeline representation comparison.  It is not a topology-causal comparison because information access, active computation and parameter count differ.
- Topology wording is limited to the existing matched T1–T5 controls in `artifacts/postclosure_topology_ablation/v1/`.
- Track-B contracts (Dense, Aer, MPS, QCSim, cuTensorNet, Pasqal) are not pooled into a universal runtime leaderboard because their timing boundaries differ.
- Raw analytical duration, shot-scaled duration and train-only calibrated predictors are separate estimators; calibration is not called a raw physical law.
- Tail observations remain scored; no hard example is deleted because it worsens a metric.

