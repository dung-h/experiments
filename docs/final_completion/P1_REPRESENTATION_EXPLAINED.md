# What the completed P1 representation experiment actually measured

Saved-evidence clarification, 2026-10-23. No fitting, inference, simulation, or frozen-artifact change was performed for this clarification. This is the source/transpiled tabular P1, distinct from the A1 sample-efficiency P1.

## Dataset and target

Authority: `artifacts/real_qpu/qualified_logical_panel/panel.csv`, SHA256 `b4ade914955d7807e466cf5cf0c5265d21b39a81fdcee8e91dc2fb6300eb978a`. Existing `outer_splits.csv` and `inner_splits.csv` in that directory define the five outer/four inner group folds. These are real archived QPU labels, not simulator pretraining observations.

| Source | Observations | Structural groups | Shots | Compiled-structure qualification |
|---|---:|---:|---|---|
| Ma–Li | 340 | 148 | 1,024 | Saved frozen reconstruction; historical executed compilation not established |
| Qonductor | 230 | 11 | 1,000; 4,000; 8,000; 12,000; 16,000; 20,000 | Exact archived submitted physical QASM |
| QPack MCP | 3,945 | 6 | 4,096 | Saved structural reconstruction; historical optimizer angles/routing absent |
| Total | 4,515 | 165 | Source-dependent | Mixed provenance |

`target_seconds` is the unchanged provider execution/service-time label. Source boundaries remain distinct: Ma–Li averages three `Result.time_taken` observations, Qonductor uses recorded one-circuit job runtime, and QPack converts recorded milliseconds to seconds. See `artifacts/real_qpu/qualified_logical_panel/summary/README.md`, Data and protocol. The 4,515 rows do not mean 4,515 independent circuits. Ma–Li represents 148/165 groups, whereas QPack dominates the row count.

The cached compiled ledger is `artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv`. The P0 audit joined all observation IDs to 346 distinct graph payloads, or 374 graph-plus-snapshot keys. No new transpilation was performed for P1.

## Exact inputs

"Static" means counts/structural descriptors without instruction-duration values. It does not mean topology-free: depth and structural ratios already summarize dependencies. Neither static view sends adjacency or individual operation tokens to the learner.

**Source block: 51 fields.** The existing Ma–Li-style schema contains 44 exact operation-name counts, allocated qubit width, depth, and five ratios: program communication, critical depth, entanglement ratio, parallelism, liveness. The full gate dictionary is in `benchmark_v1/protocol/mali_full_features.json`, `global_features.ordered_fields`. This P1 uses all 51 slots, including zero/constant slots; it does not use the neural fit-only 40-column mask. Source-input qualification differs by source as documented above.

**Compiled/static block: 36 fields.** Eight saved descriptors (active width, depth, operation count, two-qubit count, measurement count, literal-SWAP count, DAG node count, DAG edge count), 27 fixed opcode counters, and one other-opcode counter. The opcode vocabulary includes `cx`, `cz`, `ecr`, `sx`, `rz`, `measure`, `reset`, `barrier`, and `delay`; unknown names enter `other`. Measurement/SWAP counters overlap their opcode counters, and node count may overlap work count. This is a compact hand-defined summary, not the complete graph or a recomputation of all 51 source descriptors on the compiled circuit. Physical two-qubit depth and routing/expansion descriptors were unavailable and were not input fields.

**Timing block: one field.** `nominal_single_shot_duration_seconds` is copied from successful saved `qiskit_estimate_duration_snapshot__raw` predictions in `artifacts/real_qpu/qualified_logical_panel/analytical/attempts.csv`. It is Qiskit's nominal single-shot duration estimate, not observed runtime, not fitted calibration output, and not multiplied by shots. It was joined by observation ID. P0 calls this field `weighted_critical_path`, but P1 does not recompute a node-weighted path on its cached graph. Graph/circuit/snapshot identity equivalence between the analytical route and each graph payload needs an additional lineage check before using it as graph-bound timing evidence. No node-wise durations or forward/reverse path arrays were used.

**Shared context: one field.** Every view receives `log1p(shots)`. No source/backend category, family, hash, provenance flag, or fold is a model input. This differs from the frozen Ma–Li-style Graph/MLP and A1 circuit-only encoders, which omit shots. Having shots as an input does not itself validate arbitrary-shot extrapolation.

| Saved view | Explicit input | Number of scalar inputs |
|---|---|---:|
| S0 | Source block + shots | 52 |
| P0 | Compiled/static block + shots | 37 |
| P1 | Compiled/static block + nominal duration + shots | 38 |
| H0 | Source + compiled/static blocks + shots | 88 |
| H1 | Source + compiled/static blocks + nominal duration + shots | 89 |

The P0/P1 names above are feature-view identifiers; they also collide with phase names. Prefer the explicit input names in thesis prose.

## Learners and held-out evaluation

Primary learner: Ridge, trained to predict `log1p(target_seconds)`, alpha selected from `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]` by four inner group folds. Secondary learner: HistGradientBoostingRegressor with the two candidates frozen in `p1_secondary_hgb_manifest.json`. This is neither an MLP nor a Graph Transformer.

Counts/depth and nominal duration use log1p; five source ratios retain their natural scale. StandardScaler is fit inside each inner training partition, then refit on full outer-train for the selected candidate. Fitting uses ordinary row-weighted objectives; selection and primary reporting use group-balanced raw-seconds MAE. The code does not apply group-balanced sample weights during fitting. Predictions use `expm1`, without an additional clipping rule.

Each learner/view fits five selected outer models and predicts each of the 4,515 observations exactly once out of fold. There are 22,575 saved prediction rows per learner across five views. The ten learner/view combinations are not ten independent datasets. These fits do not use simulator checkpoints or the three-seed neural ensemble.

## Rechecked numerical results

The clarification independently joined saved predictions back to the panel, checked identical targets and IDs, verified all manifest feature/split/panel hashes, and recalculated both MAEs. All matched the aggregate artifact.

| View | Ridge group-MAE (s) | Ridge pooled MAE (s) | HGB group-MAE (s) | HGB pooled MAE (s) |
|---|---:|---:|---:|---:|
| Source + shots | 0.853614 | 1.153642 | 0.630509 | 0.859529 |
| Compiled/static + shots | 1.189556 | 0.981639 | 0.627485 | 0.864478 |
| Compiled/static + duration + shots | 0.777248 | 0.958265 | 0.630099 | 0.961332 |
| Source + compiled/static + shots | 0.647324 | 0.894772 | 0.648004 | 0.854972 |
| Source + compiled/static + duration + shots | 0.657647 | 0.914899 | 0.611596 | 0.886080 |

Saved paired group-bootstrap examples, 10,000 draws, seed 42:

- Ridge compiled/static minus source: +0.335942 s, 95% CI [+0.160810, +0.510825].
- Ridge hybrid minus source: -0.206289 s, CI [-0.404378, -0.062162]. This is positive evidence of complementarity under Ridge and must not be erased by a blanket negative conclusion.
- HGB compiled/static minus source: -0.003024 s, CI [-0.070772, +0.055686].
- HGB hybrid+duration minus hybrid: -0.036408 s, CI [-0.071769, -0.004819]. This is an incremental nominal-scalar effect, not physical adjacency evidence.

These are conditional paired intervals for saved fits; they do not include new training or dataset-selection uncertainty. Multiple related contrasts were reported; individual intervals are not simultaneous multiplicity-adjusted guarantees.

## Defensible conclusion and limitations

The completed P1 supports Ridge-specific source/compiled complementarity and some learner-dependent nominal-duration effects. Static compiled summaries alone did not consistently improve source summaries. It did not train or reject a physical-DAG Transformer. A missing advantage from this lossy tabular summary is not a general falsification of physical execution information.

Comparisons to the frozen Graph pooled MAE of approximately 0.834 s are descriptive across pipelines: shots, representations, objectives, epoch selection, and fitting populations differ. A new method must beat strong information-matched controls; outperforming only the weaker source Ridge is insufficient.

## Source pointers

- `research/postfreeze_track_a_transpiled_representation_v1/prepare_p1_features.py`: feature extraction and schemas.
- `research/postfreeze_track_a_transpiled_representation_v1/run_p1_representation.py:77`: exact input transforms.
- Same file, line 137: nested fitting and selection; line 158: outer refit; line 168: prediction/selection outputs.
- `research/postfreeze_track_a_transpiled_representation_v1/run_p0_coverage_audit.py:120`: raw analytical scalar selection; line 168: proxy copied into the coverage ledger.
- `artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/{ridge,hgb}/outer_predictions.csv`: original saved predictions.
- Same root, `aggregate/{pooled_metrics,source_metrics,paired_comparisons}.csv`: original aggregate evidence.
- Same root, `validation_receipt.json`: validation PASS for saved identities/folds/predictions; not a receipt that a Transformer was trained.
