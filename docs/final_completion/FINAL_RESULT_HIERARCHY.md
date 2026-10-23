# Final result hierarchy

Frozen historical results remain cited in place. Post-freeze results are recorded separately and do not rewrite Scientific Freeze v2.

## Primary findings

1. Runtime-estimator ranking depends on weighting and execution context; pooled, source-balanced and group-balanced metrics answer different questions.
2. A universal runtime estimator across incompatible execution contracts was not established.
3. Strict simulator-to-QPU pretraining did not establish a 100%-QPU benefit under Amendment A1. Under the completed reduced-supervision extension, it improved MLP group-balanced MAE at 10% and 25% fitted-group fractions in the two-draw protocol, but did not establish Graph+MLP benefit or benefit at 50%/100%.
4. Severe structural distribution shift can produce catastrophic, directionally consistent neural error, while simpler regression remains more stable in the documented holdouts.

## Secondary findings

- Compiled representation improves GNN behavior in the tested Aer configuration, but source-to-transpiled changes are not an edge-only intervention.
- Fixed-MPS Graph+globals outperforms Ridge in its declared contract; this is context-specific.
- The QCSim Maestro-style component improves paired MAE in its canonical 150-hash contract; it is not a universal Maestro selector.
- Graph inference can cost more without a stable universal accuracy advantage.
- Benchmark-scale transpiled/static representation was not robustly superior to source/static representation under matched Ridge/HGB P1 evaluation; this did not justify physical dependency-DAG P2.

## Diagnostics

- QNN/Random support violations and signed neural overprediction.
- Backend-holdout representation collisions.
- Moderate MAE with strongly negative R² under low test variance.
- MPS quality-probability improvement without a safe χ-selection guarantee.
- Raw analytical duration being far from observed service time, with calibration acting as learned adaptation.

## Completion status

The 10/25/50% transfer extension is complete: `aggregation_v2/completion_receipt.json` records 360/360 valid cells and the saved predictions, paired group bootstrap, and replay receipt are available. Its low-fraction MLP result is conditional rather than a replacement for the frozen 100%-QPU A1 conclusion. The benchmark-scale transpiled representation P0/P1 study is also complete and negative at its pre-registered escalation gate; no physical-DAG P2 was run.

