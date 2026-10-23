# Work requested during the sample-efficiency campaign

This navigation record distinguishes completed saved evidence from pending work. It updates the task queue without rewriting historical experiments or receipts.

| Work | Current evidence/status | Location |
|---|---|---|
| Reduced-fit-group simulator transfer | COMPLETE: 360 new cells; validation PASS; 10/25/50% fits plus frozen 100% reuse. Conditional, two-draw analysis with full validation retained. | `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/` |
| Study R raw R² and topology mechanism audit | COMPLETE: saved 340-row/170-raw-QASM-group analysis, 5,000 paired group-bootstrap draws. Kept distinct from Track A. | `artifacts/postfreeze_dag_raw_r2_audit_v1/` |
| QPU target-boundary / analytical-overhead audit | COMPLETE as an audit: source semantics, raw/shot-scaled/calibrated routes, H1/H2 partial support and H3 heterogeneous-source support. Component-level overhead remains unknown. | `docs/audits/qpu_timing_boundary/` |
| Material shots limitation | RECORDED: frozen Graph/MLP and A1/sample-efficiency encoders omit shots; mixed-shot accuracy is not validated query-shot conditioning. | `docs/final_completion/FROZEN_EVALUATION_CONTRACT.md`; `CLAIM_MATRIX.md` |
| Existing logical-topology P2 reconciliation | COMPLETE: reuse saved T1–T5, 75-cell study; no duplicate fit launched. | `artifacts/postfreeze_track_a_transpiled_representation_v1/logical_topology_existing_evidence/REPORT.md` |
| Compiled/transpiled coverage and tabular P1 | COMPLETE: 4,515 observations, 165 groups, five feature views, Ridge/HGB; includes favorable Ridge hybrid contrasts and limited cross-learner consistency. | `artifacts/postfreeze_track_a_transpiled_representation_v1/`; `P1_REPRESENTATION_EXPLAINED.md` |
| Full 4,515-row Azizov-inspired source/transpiled graph comparison | NOT RUN. Tabular P1 is not this graph experiment. | New design required after pilot; see Transformer strategy |
| New Transformer method | DESIGN ONLY. Latest user direction: first Ma–Li 340/148-group pilot, resource canary before choosing complete-operation sparse attention versus coarsening; scale only after pilot review. | `PHYSICAL_DAG_TRANSFORMER_STRATEGY_DRAFT.md` |
| Full Graph/MLP/Ridge/RF/XGBoost accuracy–prediction-cost study | NOT RUN. User has requested matched measurement; inventory fitted states and implement common feature/inference/warm-QASM boundary before timing. No training is implied by this task. | Existing Graph/MLP-only evidence in `artifacts/am1_phase4/v1/`, `artifacts/am1_phase5_dense_b2/v1/`; prior cost gap in `artifacts/postfreeze_final_completion_v1/cost_audit.md` |
| OOD severity | PARTIAL: existing support violations and directional errors reconciled; no new comprehensive row-level association analysis completed. | `artifacts/postfreeze_final_completion_v1/ood_support_diagnostic.csv` |
| Future circuit/QPU encoders and cross-attention | IDEA RECORDED; no implementation or experiment. | `FUTURE_CIRCUIT_QPU_ENCODER.md` |
| Presentation | LOCAL HANDOFF ONLY; no remote Canva edit completed in this campaign. | `PRESENTATION_UPDATE_BRIEF.md` |

## Commit and publication scope

The user explicitly requested commit/push on 2026-10-23. Existing distribution scope excludes third-party QPU observation rows, row-level predictions, source QASM/graphs and group-membership ledgers. Reports, aggregate metrics, producing code, tests and plot figures can be published. Full local evidence is preserved.

The development `main` contains two older unpublished commits; one includes row-level A1 evidence. To avoid publishing those ancestors as a side effect, the scoped report/code package is committed on `postfreeze-track-a-completion`, based on the verified remote `main`. No force push, history rewrite, or deletion is part of this publication. The source worktree and local scientific HEAD remain unchanged.

`scripts/export_postfreeze_track_a_completion.py` defines the explicit file allowlist; the generated `docs/final_completion/PUBLICATION_INDEX.json` records byte hashes. `PUBLICATION_SCOPE.md` describes omitted dependencies. A fresh clone can verify the selected report package; it cannot rerun the private-data experiments without the declared local prerequisites.
