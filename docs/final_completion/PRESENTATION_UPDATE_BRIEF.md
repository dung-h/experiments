# Presentation update brief

The requested early synthesis slide for the Canva deck is specified here because no Canva/browser surface is available in the current execution environment. No presentation file or remote design was modified.

## Slide: Benchmark này so sánh và đo những gì?

**Accuracy**

- Compare: Graph / MLP / regression / analytical controls
- Measure: MAE, R², balanced MAE, tails, coverage

**Transfer & robustness**

- Compare: pretrained vs scratch; ID vs held-out family/backend/size
- Measure: paired ΔMAE + CI, OOD support, fold/source behavior

**Practical usefulness**

- Compare: accuracy vs prediction cost; runtime vs quality configuration
- Measure: latency, fidelity/Brier, failure/abstention

Bottom boundary:

> Only compare methods directly when target, test population and execution contract are matched.

## Slide: Các kết quả bất thường cần giải thích

1. Weighting changes estimator ranking.
2. Simulator transfer is heterogeneous: only the low-fit-group MLP extension established a benefit; full-supervision and Graph benefit were not established.
3. Severe OOD causes catastrophic neural overprediction.
4. Moderate MAE can coexist with strongly negative R².
5. Graph can cost more without a stable accuracy advantage.
6. Better quality probability does not guarantee safe χ selection.

## Status

`PENDING_CANVA_SURFACE`: the local CUA inventory exposed no browser/app surface, so this brief is the reproducible handoff for a later Canva edit. The slide content is intentionally bounded by the frozen evidence and does not introduce a new result.

## Post-freeze representation note

If one sentence is added near the generalization/representation discussion, use:

> On the 4,515-row Track-A panel, the matched source-versus-transpiled P1 study did not establish a representation-wide compiled-structure advantage; this does not establish that physical dependency topology is irrelevant.

Evidence: `artifacts/postfreeze_track_a_transpiled_representation_v1/FINAL_REPORT.md`.
