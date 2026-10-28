# Ma–Li shots ablation protocol

Status: `LOCKED_PRE_OUTCOME`. This is a bounded post-freeze extension. It does not change Scientific Freeze v2, saved A1 results, targets, or folds.

## Question and comparison

Test whether adding the recorded shot count improves OOF runtime prediction inside the same MLP and Graph+MLP architecture. M0/G0 are reused frozen baselines; only M1/G1 are newly trained.

## Locked data and training

- 4,515 observations, 165 structural groups, five existing outer folds.
- Inner fold 0 selects checkpoints; inner folds 1–3 fit transforms and train. Outer test is untouched until prediction.
- 500 epochs, batch 32, Adam (0.0005, weight decay 0.0001), float32, raw-seconds MSE; minimum validation MSE, earliest tie.
- Seeds 42, 1234, 31415. New cells: 30 (2 architectures × 5 folds × 3 seeds).
- Shots are raw counts standardized using only the fold's inner-fit mean and ddof=1 sample standard deviation. No log transform, multiplier, shot curve, source ID, or backend ID.
- Only `gf_linear1` gains one input weight per output; the new column is zero at initialization and all existing tensors are copied unchanged. Fresh Adam optimizer.

## Primary endpoint

Group-balanced MAE in seconds. Primary paired contrasts: G1−G0 and M1−M0; negative favors adding shots. Use 10,000 paired structural-group bootstrap replicates with seed 42. Report 95% percentile CIs and 97.5% intervals for the two-contrast family. No equivalence claim when an interval includes zero.

## Reuse and provenance

The 30 frozen baseline cells were checksum/split/OOF validated before reuse. Their receipt and predictions are preserved in this output namespace. Detailed immutable inputs, fold transform hashes, and protocol pins are in `protocol.json`.

## Shot-response limitation

Ma–Li and QPack have fixed shot levels; pooled gains may reflect source/context separation. Qonductor has only 11 groups. Within-hash/backend shot diagnostics will use held-out OOF predictions only and will be described as observational, not causal.

## Stop rule

No result-driven architecture or hyperparameter search. Incomplete cells or missing paired predictions block interpretation. No QPU/simulator execution, no frozen artifact modification, no commit or push.
