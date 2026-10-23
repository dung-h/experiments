# Existing Evidence Audit

This audit freezes the evidence inventory before any new final-completion run.  It is read against the saved artifacts; no frozen prediction, split, or Scientific Freeze artifact is rewritten.

## Status key

- **COMPLETE** — the question has a saved, auditable comparison under a coherent contract.
- **PARTIAL** — useful evidence exists, but the proposed claim is broader than the saved contract.
- **MISSING** — no saved evidence answers the stated question.
- **REDUNDANT** — a proposed rerun would duplicate a frozen result without closing a gap.

## Inventory

| Proposed area | Status | Evidence / reason |
|---|---|---|
| Track-A frozen accuracy panel | COMPLETE | `artifacts/final_research/rq4_report.md`; 4,515 observations, 165 structural groups, five frozen outer folds. |
| Study-R historical Ma–Li reconstruction | COMPLETE (historical) | Historical artifacts and `docs/track_a_transfer_study/TOPOLOGY_PREDICTIVE_VALUE_AUDIT.md`; retained as historical, not replaced by Track-A. |
| Strict simulator-to-QPU transfer at 100% QPU labels | COMPLETE | `artifacts/postfreeze_track_a_transfer_v1/FACTORIAL_100_RESULTS.md`; 60/60 cells, 5 folds, 3 seeds, frozen A1 contract. |
| Transfer at 10/25/50% QPU labels | COMPLETE, qualified | `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/`; 360/360 fraction × outer-fold × subset-draw × treatment × model-seed cells validated. MLP pretraining improves group-balanced MAE at 10% and 25% fitted-group fractions in the two-draw design; it is not established at 50%/100%, and no Graph+MLP fraction establishes a benefit. |
| Graph-vs-MLP pipeline comparison | COMPLETE (conditional) | `artifacts/final_research/rq4_report.md` and `docs/research_conclusions.md`; does not isolate topology. |
| Matched topology controls | COMPLETE (bounded) | `artifacts/postclosure_topology_ablation/v1/`; T1–T5, 75 cells, grouped bootstrap. The claim is correctly bounded; no new topology run is required. |
| OOD family/backend/size holdouts | COMPLETE (diagnostic) | `docs/generalization_evaluation.md`, `artifacts/final_research/rq4_report.md`; support-distance interpretation remains non-causal. |
| OOD severity association | PARTIAL | Support violations and signed-error failures are documented, but no single saved association table covers every held-out case. A read-only diagnostic will be generated if inputs are available without refitting. |
| Accuracy–prediction-cost: Graph vs MLP | PARTIAL | `artifacts/am1_phase4/v1/` and `artifacts/am1_phase5_dense_b2/v1/` contain matched subsets and latency evidence, but do not provide a complete classical-baseline cost frontier. |
| Accuracy–prediction-cost: classical baselines | MISSING | Existing cost reports explicitly state that competitive fitted states (RF/XGBoost/Ridge) remain missing from the matched timing population. A small, matched read-only timing benchmark is conditionally feasible. |
| MPS runtime–quality threshold frontier | PARTIAL | `artifacts/simulator_papers/emu_mps_threshold_extension_20260920/` contains threshold labels/sweeps for a separate analog/emulation contract; it is not a frozen Track-B selector evaluation. No new selector run is required unless that claim is made central. |
| Track-B simulator contract separation | COMPLETE | `docs/track_b_finality/TRACK_B_150_CORE_FINALITY_AUDIT.md` and canonicalization report; contracts remain context-specific and are not pooled into a universal leaderboard. |
| Analytical raw vs calibrated estimators | COMPLETE | `docs/research_conclusions.md`, `docs/results.md`; raw, shot-scaled, and train-only calibrated variants are separated. |
| Benchmark-scale source/transpiled representation | COMPLETE, negative gate | `artifacts/postfreeze_track_a_transpiled_representation_v1/`; P0 provenance audit plus matched nested group-safe Ridge/HGB P1. No representation-wide transpiled advantage was established, so physical dependency-DAG P2 was not opened. |

## Decision after completion

The required sample-efficiency comparison is complete and supports a deliberately narrow conclusion: strict simulator initialization improved MLP group-balanced MAE at low fitted-group fractions (10% and 25%) in this two-draw, full-validation protocol, but does not establish a Graph+MLP benefit or a benefit at 50%/100%. Existing topology controls remain sufficient for bounded non-causal wording. The benchmark-scale transpiled-representation study did not clear its pre-registered P1 gate: static physical features were not robustly superior, timing/hybrid behavior was learner- and source-dependent, and physical-DAG P2 is not justified. The cost evidence remains a partial Graph-vs-MLP contract because a matched classical raw-input timing path is unavailable. No model search, new architecture, reverse-size split, crossed backend split, or new simulator/QPU measurement is authorized by this completion record.

