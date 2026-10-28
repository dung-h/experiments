# NoTaDS and HyperQ on the frozen Track-A population

## What was evaluated

All **4,515 observations / 165 structural groups**, with the original five group-disjoint outer folds. Sources: Ma–Li 340 rows / 148 groups; Qonductor 230 / 11; QPack 3,945 / 6. Targets are the original provider Result.time_taken labels in seconds, with their original aggregation and unit conversions. This new analysis does not replace frozen results.

The application reuses 346 saved native graph files in 374 graph/snapshot contexts. Qonductor's 230 rows retain exact archived submitted physical circuits. The other 4,285 rows use the already frozen compiled reconstructions; their numerical angles/routing do not establish exact historical execution circuits. Backend durations come from the pinned nominal snapshots, not certified execution-day calibration.

The [protocol](PROTOCOL.md) was declared before scoring. [Amendment 01](PROTOCOL_AMENDMENT_01.md) records two source/clock corrections found during preflight, also before scoring. No runtime data was collected, no circuit retranspiled/executed, and no neural/tree model or architecture was trained/searched. Thirty scalar affine calibration fits were performed on outer-train rows only, as explicitly declared below.

## Method fidelity

**NoTaDS paper:** native unitary operations are assigned dependency levels; each level with a 2Q operation costs 10 units, and other levels cost 1 unit. This is the paper's relative time score. **NoTaDS code:** `depth(1Q)+10*depth(2Q)`, transcribed from the public code; the two filtered depths retain synchronization through excluded gates. These formulas can differ. Both use the same native unitary representation, excluding terminal measurements. Applying them to uncut native circuits is a declared representation adaptation, not a run of the NoTaDS cutting/scheduling workflow.

Neither NoTaDS score is already in seconds. Raw and shots-scaled scores are saved in `features.csv`; no seconds MAE is assigned to arbitrary units. To predict observed seconds, this study applies the same type of two-parameter mapping used by frozen analytical controls: `max(0, a+b*x)`, with x equal to either raw score or shots*score. Each held-out fold uses coefficients fitted exclusively on the other four folds. There is no choice of a best variant or hyperparameter search.

**HyperQ paper reimplementation:** longest dependency path weighted by the per-operation, per-qubit Target duration. It includes all archived terminal measurements, including repeated measurements. The unmodified path predicts single-execution duration. Multiplying by shots is a separate declared workload adaptation; provider overhead and repetition delay are not invented. Affine variants are learned adaptations of these scalar inputs. The public HyperQ depth scheduler was not substituted for the paper algorithm. Source claim/code audit: `docs/audits/notads_hyperq_estimator_20261027/REPORT.md`.

## Results in seconds

MAE is average absolute prediction error. Pooled MAE weights all observations equally; group-balanced MAE gives each structural group equal weight and is the primary descriptive metric here.

| Route | Pooled MAE | Group-balanced MAE | Raw R² |
|---|---:|---:|---:|
| NoTaDS paper score → train-only affine | 1.061807 | 1.127675 | 0.244013 |
| NoTaDS paper score × shots → train-only affine | 1.054150 | 1.122026 | 0.259023 |
| NoTaDS code score → train-only affine | 1.058794 | 1.128705 | 0.250716 |
| NoTaDS code score × shots → train-only affine | **1.042507** | **1.120445** | 0.282618 |
| HyperQ raw path, no calibration | 6.115983 | 7.898773 | −17.132792 |
| HyperQ path × shots, no calibration | 5.654227 | 3.698727 | −14.476318 |
| HyperQ raw path → train-only affine | 1.060995 | 1.135217 | 0.246555 |
| HyperQ path × shots → train-only affine | 1.049696 | 1.127682 | 0.268569 |

The bold row is the lowest observed point value in this bounded table, not a post-hoc selected new method. All variants remain reported.

For the shots-affine variants, pooled MAE by source is:

| Source | NoTaDS paper | NoTaDS code | HyperQ paper |
|---|---:|---:|---:|
| Ma–Li | 0.941860 | 0.943253 | 0.950309 |
| Qonductor | 2.666728 | 2.626663 | 2.649389 |
| QPack MCP | 0.969812 | 0.958702 | 0.964997 |

These are global outer-train calibrations evaluated within each source, not separately source-fitted models. QPack dominates pooled row counts; Ma–Li dominates group counts. Retain both weightings.

## Paired differences and practical scale

Paired whole-group bootstrap, 10,000 replicates, seed 42, group-balanced MAE differences; negative favors the first route. These are descriptive intervals, with no multiplicity-adjusted confirmatory claim.

| First minus second | Δ GB-MAE (s) | 95% percentile CI (s) |
|---|---:|---|
| NoTaDS paper shots-affine − NoTaDS code shots-affine | +0.001582 | [−0.032174, +0.057115] |
| HyperQ shots-affine − NoTaDS code shots-affine | +0.007237 | [+0.000838, +0.013635] |
| HyperQ shots-scaled − saved Qiskit shots-scaled | 0.000000 | [0.000000, 0.000000] |

The paper/code score difference does not establish a clear predictive difference after calibration. The code-based NoTaDS adaptation has a small descriptive advantage over calibrated HyperQ: about **7 milliseconds** of group-balanced absolute error, against a baseline error around 1.1 seconds. This does not establish a substantial new estimator advance.

## HyperQ duplicates an existing analytical route on these inputs

The full weighted-path calculation matches the saved Qiskit nominal duration for **every observation**, with maximum absolute difference **0 s**. Shots-scaled predictions also match exactly. Both independently replayed affine variants match their saved Qiskit counterparts within **1.25×10⁻¹⁴ s**. Thus adding this HyperQ paper reimplementation would duplicate that Qiskit route numerically on this cohort/configuration. Do not count it as independent estimator evidence or a new winner.

This is an equality verified on these saved inputs, not a claim that all HyperQ and Qiskit workflows are universally equivalent. HyperQ's original qVM packing, reset insertion, time multiplexing and systems evaluation were not rerun.

## Preflight findings and limits

1. **82 stale representation-table shots fields.** Frozen panel shots match canonical source rows and native-feature rows on all 4,515 observations, and read-only SQL Job.shots on all 230 Qonductor observations. The older representation producer copies a cached circuit/backend row without refreshing shots (`benchmark_v1/scripts/materialize_compiled_unified_qpu_v1.py` lines 931–941). Static circuit identity matches; this application uses the verified panel shots and stores all discrepancies in `representation_shots_discrepancies.csv`. No historical table was repaired. Effects on other historical consumers are outside this audit.
2. **Measurement boundary correction.** Earlier audit prose implied that the saved Qiskit duration had the same unitary-only boundary as the converted QCRE route. Direct replay proves the saved Qiskit route includes measurements on this cohort. For the first graph, the full duration is 0.006267196 s, while the unitary duration is 0.006265796 s. The existing Qiskit numbers are unchanged; their boundary description needs this qualification. Producing source calls `compiled.estimate_duration` on the whole compiled circuit and the qualified runner reuses those saved values.
3. **Clock and reconstruction limits.** Absolute snapshot durations, relative NoTaDS scores and provider runtime labels are distinct quantities. The affine results learn a bridge using training labels. Their accuracy does not validate uncalibrated physical accounting or identify a particular overhead. Reconstruction qualification and the small number of QPack/Qonductor groups remain relevant.
4. **Implementation preflight history.** Failed development checks were resolved before scoring: the initial native hash check compared byte hashes with a semantic digest; a shots assertion exposed the stale cached metadata; the duration check initially expected the incorrect unitary boundary; parsing integer-valued shots strings required accepting the frozen CSV's `.0` formatting. No frozen invariant was silently changed, no failed scored prediction was dropped, and all final rows are retained.

## Validation

- Six arithmetic tests PASS, covering parallel/mixed layers, serial dependencies, zero-duration gates, repeat measurements, random resource DAGs and rejection of a gate after measurement.
- Eleven saved graph canaries, including the largest graph, PASS an independent explicit-DAG recurrence; ten smallest-per-backend canaries also match Qiskit filtered depth.
- All graph and snapshot pins checked. Extraction peaks at about 1.20 GB RSS; one CPU process, no GPU training.
- Complete 40,635 prediction rows: nine scored routes/reference × 4,515 IDs. Every calibrated row has exactly one outer-test prediction; group overlaps are zero.
- Independent validation: **226,083 checks PASS**, including centered-regression replay, target/fold/group alignment, metrics, deterministic bootstrap and frozen Qiskit prediction equality.
- Original input hashes unchanged. No frozen scores, source QASM, folds, labels or checkpoints overwritten. No commit/push.

## Files and reproduction

Raw scores/provenance: `features.csv`, `graph_scores.csv`, `feature_receipt.json`. Results: `predictions.csv`, `metrics.csv`, `paired_comparisons.csv`, `bootstrap_draws.npz`, `calibration_fits.csv`, `leakage_receipts.csv`. Validation: `independent_validation.json`, `test_environment_receipt.json`, `unit_tests.log`, `completion_receipt.json`.

On a fresh output copy, with the original saved inputs available:

```bash
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_notads_hyperq_track_a_v1/test_core.py
/home/server/Documents/.venv-qonductor/bin/python -u research/postfreeze_notads_hyperq_track_a_v1/run.py --stage extract
/home/server/Documents/.venv-qonductor/bin/python -u research/postfreeze_notads_hyperq_track_a_v1/run.py --stage score
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_notads_hyperq_track_a_v1/validate_saved.py
```

Keep this completed namespace as the execution record. No further model search follows from this application.
