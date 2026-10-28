# Ma–Li shots-input ablation — final report

## Question

Does adding the recorded shot count improve runtime prediction within the existing Ma–Li MLP and Graph+MLP pipelines on the frozen Track-A QPU benchmark?

## Protocol and data

- Frozen Track-A cohort: 4,515 observations, 165 structural groups, five existing outer folds. Source counts per treatment: Ma–Li 340, Qonductor 230, QPack 3,945.
- M0/G0 are the saved no-shots baselines. M1/G1 add one raw-shot input, standardized from each fold's inner-fit rows only. No shot curve, multiplier, source ID, or backend ID was added.
- The 30 baseline cells were reused after checksum, split, and OOF checks passed. The 30 new cells were MLP+shots and Graph+MLP+shots over five folds and three seeds. Each used the locked 500-epoch budget and saved the minimum inner-validation-MSE checkpoint.
- Final OOF prediction for each new treatment is the median of seeds 42, 1234, and 31415. All four treatments cover the same 4,515 rows and 165 groups.

## Main results

Primary metric is group-balanced MAE: calculate MAE within each structural group, then average groups equally. Lower is better. Raw and log1p R² are labeled separately.

| Treatment | Pooled MAE (s) | Group-balanced MAE (s) | Raw-seconds R² | log1p-seconds R² | Max absolute error (s) |
|---|---:|---:|---:|---:|---:|
| M0: MLP, no shots | 0.893423 | 0.580697 | 0.363158 | 0.301486 | 8.063038 |
| M1: MLP + shots | 0.896341 | 0.544851 | 0.424937 | 0.395831 | 6.654804 |
| G0: Graph+MLP, no shots | 0.834133 | 0.633242 | 0.468091 | 0.500307 | 7.329822 |
| G1: Graph+MLP + shots | 0.813429 | 0.603041 | 0.477419 | 0.538851 | 5.935823 |

| Paired contrast (with shots − without) | Δ group-balanced MAE (s) | 95% group-bootstrap CI (s) | 97.5% interval (s) | Point improvement |
|---|---:|---:|---:|---:|
| M1 − M0 | −0.035846 | [−0.093131, +0.006669] | [−0.103400, +0.011245] | 6.17% |
| G1 − G0 | −0.030201 | [−0.063154, +0.001449] | [−0.067570, +0.006045] | 4.77% |

Intervals use 10,000 paired bootstrap replicates of the 165 structural groups, seed 42. Both intervals include zero. The observed improvements are favorable point estimates, but this study does **not** establish a reliable improvement from adding shots. The bootstrap fractions favoring a negative delta (94.42% MLP; 96.90% Graph) are descriptive fractions, not p-values.

Pooled MAE tells a different story for the MLP: it rises by 0.002918 s, while its group-balanced MAE falls by 0.035846 s. The evaluation weighting changes the conclusion because the sources and group sizes are uneven. Do not summarize this as a universal gain.

## Fold and source patterns

The group-balanced fold deltas (shots minus no-shots) were:

| Outer fold | MLP Δ (s) | Graph Δ (s) |
|---:|---:|---:|
| 0 | −0.001927 | −0.075221 |
| 1 | −0.136745 | −0.110368 |
| 2 | −0.033897 | +0.019680 |
| 3 | −0.013795 | −0.026807 |
| 4 | +0.006580 | +0.037311 |

Source-level group-balanced results are descriptive. MLP changes were −0.026162 s on Ma–Li, −0.209197 s on Qonductor, and +0.043070 s on QPack. Graph changes were −0.028753 s on Ma–Li, −0.079316 s on Qonductor, and +0.024123 s on QPack. The improvements are not uniform across sources.

## What the shot feature learned

The raw shot counts differ by source: Ma–Li uses 1,024 shots throughout, QPack uses 4,096 throughout, and Qonductor contains 1,000–20,000 shots. Thus shots can also act as a source/context marker in the pooled benchmark.

As a small observational diagnostic, the OOF predictions were grouped by 13 Qonductor `(source_qasm_sha256, backend)` contexts. There were 25 context/shot-level comparisons above 1,000 shots. The median observed-target ratio relative to the same context at 1,000 shots was 1.319. The median prediction ratio was 1.000 for the no-shots controls, 1.450 for MLP+shots, and 1.199 for Graph+MLP+shots. Support at higher shot levels thins from 13 contexts at 4,000 to one at 20,000. This is not a causal shot-scaling estimate and does not validate a linear shot law; shot levels may also vary with date and workload context.

## Execution and validation

- Training completed: 30/30 new cells; 27,090 per-seed predictions and 18,060 four-treatment median OOF predictions.
- Baseline reuse receipt: PASS. Final validation receipt: PASS, 54,390 checks, including cohort, fold/group isolation, exact held-out IDs, initialization parity, finite outputs, and artifact checksums.
- No QPU jobs, simulator runs, model search, fold changes, or changes to frozen benchmarks were performed.
- Three early startup attempts stopped before training outcomes. A logging-only launcher shim bypassed an argument error in the runner's console-print statement; model, optimizer, batches, seeds, data, and checkpoints were unchanged.
- The root filesystem filled during cell 20's checkpoint write. The existing checkpoint was readable at `next_epoch=331`; training resumed from that checkpoint and all 30 cells subsequently passed validation.
- To restore writable space, 1,240 checkpoint files in the 2026-10-16 backup (9,561,859,468 bytes) were removed only after SHA-256 equality was verified against their counterparts in the main repository. The main-repository copies remain. Deletion receipt: `/mnt/windows/Datamining/checkpoint_cleanup_receipt_20261027.json`.

## Conclusion

Adding raw shot count produces modest favorable group-balanced point estimates in both architectures, but the paired group-bootstrap intervals include zero and source/fold effects vary. The defensible conclusion is: **this experiment did not establish a robust benefit from adding shots as one scalar feature**. It also does not establish that shot count is irrelevant or that runtime scales linearly with shots. Historical baselines and frozen Track-A results remain unchanged.

## Artifacts

- Protocol: `PROTOCOL.md`; machine-readable locks: `protocol.json`.
- Training outputs: `training/fold_*/...`.
- OOF and metrics: `aggregate/oof_predictions.csv`, `aggregate/metrics.csv`, `aggregate/slice_metrics.csv`, `aggregate/paired_comparisons.csv`, `aggregate/group_bootstrap_draws_seed42.npy`.
- Validation: `baseline_reuse_receipt.json`, `validation_receipt.json`.
- Complete artifact hashes: `artifact_manifest.json`.
