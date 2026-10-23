# Final completion report

This report records the final evidence-completion state without silently treating a partial or incompatible comparison as a scientific result. It records what was already complete, what was newly completed, what was intentionally not run, limitations, thesis claims, artifact paths and reproduction commands.

## Completion status

`FINAL_COMPLETION_POSTFREEZE_EVIDENCE_LOCKED`

The frozen evidence base is auditable and thesis-ready under the bounded claims below. The authorized 10/25/50% transfer extension completed and passed its validation receipt. The benchmark-scale transpiled-representation P0/P1 study also completed; its pre-registered escalation gate did not pass, so physical-DAG P2 was not run. Neither post-freeze study changes Scientific Freeze v2.

## A. What was already complete

See `EXISTING_EVIDENCE_AUDIT.md`.  Frozen Track-A, Study-R, A1 100%, topology controls, generalization stress tests, analytical raw/calibrated separation, and Track-B contract separation were not rerun.

## B. What was newly run

The nested group-subsampled A1 transfer study is complete in `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/`. It reuses the frozen parent model/checkpoint/split contract, samples complete outer-training structural groups, retains full validation labels, and validates all 360 cells. For MLP, pretraining minus scratch group-balanced MAE was -2.976301 s (95% CI [-4.503293, -1.569171]) at 10% and -1.025324 s ([-1.458933, -0.343123]) at 25%; 50% and 100% intervals cross zero. No Graph fraction establishes a benefit. These are conditional two-subset-draw results, not a universal total-label sample-efficiency claim.

The P0/P1 transpiled-representation study is complete in `artifacts/postfreeze_track_a_transpiled_representation_v1/`. P0 reconciled 4,515/4,515 rows and 346 cached physical graph payloads; P1 compared source, transpiled-static, proxy-timing and hybrid views under identical nested group-safe Ridge/HGB protocols. Static transpiled information was not robustly superior, and hybrid/timing patterns differed by learner and source. Consequently, the pre-registered physical-DAG P2 gate was not passed and P2 was not run.

The OOD artifact in `artifacts/postfreeze_final_completion_v1/ood_support_diagnostic.csv` is a read-only reconciliation of saved evidence, not a new model run.

## C. What was intentionally not run

No matched logical-topology rerun, physical-DAG P2, reverse-size split, crossed backend split, MPS selector threshold experiment, new model/HPO search, simulator run, QPU job, or classical timing frontier was run. The current cost harness does not expose a comparable raw-input timing path for fitted RF/XGBoost/Ridge; reporting a new number would create a different contract rather than close the gap.

## D. Remaining limitations

The reduced-supervision result uses only two deterministic subset draws and retains full validation labels, so it is not evidence for a generic total-label learning curve. Non-significant intervals do not establish equivalence. Graph-vs-MLP does not isolate topology. The negative P1 gate does not prove compiled structure or physical-DAG adjacency is irrelevant. Track-B timing boundaries remain context-specific. Existing cost evidence is a Graph-vs-MLP partial tradeoff, not a full Pareto frontier.

## E. Final supported thesis claims

Use the final result hierarchy and claim matrix. The frozen 100%-QPU A1 conclusion remains: no statistically supported benefit was established for either architecture. Add only the qualified extension: MLP simulator initialization improved group-balanced MAE at the 10% and 25% fitted-group fractions in the completed two-draw study; Graph benefit and 50%/100% benefit were not established. Do not claim universal sample efficiency or shot-conditioned transfer.

## F. Final forbidden claims

Do not claim topology causality, universal graph superiority, simulator-pretraining equivalence or universal harm, pure hardware generalization, a universal runtime law, a complete estimator-cost frontier, or a calibrated raw analytical physics formula.

## G. Artifact index

- `docs/final_completion/EXISTING_EVIDENCE_AUDIT.md`
- `docs/final_completion/FROZEN_EVALUATION_CONTRACT.md`
- `docs/final_completion/CLAIM_MATRIX.md`
- `docs/final_completion/EXPERIMENT_STOP_GO.md`
- `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/`
- `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/P1_SAMPLE_EFFICIENCY_REPORT.md`
- `artifacts/postfreeze_track_a_transpiled_representation_v1/FINAL_REPORT.md`
- `research/postfreeze_track_a_transpiled_representation_v1/`
- `artifacts/postfreeze_final_completion_v1/`
- `docs/final_completion/PRESENTATION_UPDATE_BRIEF.md`

## H. Reproduction commands

```bash
/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  research/postfreeze_track_a_transfer_sample_efficiency_v1/run_sample_efficiency.py --all
/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  research/postfreeze_track_a_transfer_sample_efficiency_v1/aggregate_sample_efficiency.py
```

The first command is the declared reproduction command for the bounded extension; the saved completed aggregation, rather than a new rerun, is the reportable result. The 100% parent cells are reused by receipt; the frozen parent namespace is not modified. The transpiled-representation script receipts are recorded under its artifact namespace and should likewise not be rerun solely to seek a different result.

## I. Presentation updates required

Add one early slide, “Benchmark này so sánh và đo những gì?”, with Accuracy / Transfer & robustness / Practical usefulness blocks and the matched-contract boundary. Add one synthesis slide, “Các kết quả bất thường cần giải thích”, covering weighting sensitivity, qualified low-label MLP transfer, OOD overprediction, MAE versus R², graph cost, and MPS safety. Mention that the benchmark-scale transpiled P1 gate did not establish a broad physical-representation advantage; do not present it as evidence that physical topology is irrelevant. A Canva edit could not be performed in this environment because no browser/app surface was available; the exact slide handoff is `docs/final_completion/PRESENTATION_UPDATE_BRIEF.md`. Do not alter the existing narrative order or imply a new statistical result.

