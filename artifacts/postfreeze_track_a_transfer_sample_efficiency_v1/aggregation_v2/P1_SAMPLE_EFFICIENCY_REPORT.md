# P1 — simulator initialization with reduced QPU fitting groups

## Completion and scope

All360 authorized new training cells completed. Saved-output validation PASS;
three independent replay tests PASS. No training or inference was rerun during
aggregation. The100% A1 prediction bytes, point effects and bootstrap intervals
were reused and independently matched. P1 supplements A1; historical partial
attempt reports and the parent remain unchanged.

New fractions retain10/25/50% of gradient-fit structural groups, with two
deterministic nested subset seeds, five frozen outer folds, four treatments and
three model seeds. Validation labels stay fully available. Thus these fractions
do not describe the total QPU-label budget. Test coverage is4515 observations /
165 groups per treatment/subset. Three-seed median predictions are taken within
each subset; errors are then averaged over the two fixed draws and equally
over structural groups. The100% point has one frozen A1 training design.

## Primary effects: pretrained minus scratch

Negative delta favors simulator initialization. Units: seconds. Percentile95%
intervals use10000 paired whole-group replicates, seed42, reusing A1 draw matrix.

| Fit groups retained | M-S | M-P | MLP delta [95% CI] | G-S | G-P | Graph delta [95% CI] |
|---|---:|---:|---|---:|---:|---|
|10%|5.756831|2.780530|-2.976301 [-4.503293,-1.569171]|3.581593|3.375050|-0.206543 [-0.932432,+0.618565]|
|25%|2.515083|1.489759|-1.025324 [-1.458933,-0.343123]|1.361455|1.496959|+0.135504 [-0.170194,+0.594206]|
|50%|1.274588|1.014043|-0.260545 [-0.579126,+0.222201]|0.877451|0.997774|+0.120322 [-0.182213,+0.653455]|
|100%, frozen A1|0.942722|0.892874|-0.049848 [-0.321792,+0.422907]|0.595000|0.875841|+0.280841 [-0.056831,+0.924418]|

Source: `model_results.csv`, `paired_comparisons.csv`; machine-readable values
retain full precision. New MLP10/25% Bonferroni percentile sensitivity intervals
over six contrasts also remain below zero: [-5.072455,-1.114179] and
[-1.538690,-0.058035]. These are conditional descriptive inferences with finite
bootstrap tail resolution, not retrospective preregistered confirmation.

## Subset sensitivity and weighting

MLP group-balanced improvements have the same direction in both draws:
10% deltas -1.96669/-3.98591s;25% -1.73405/-0.31660s.
Graph10% changes direction across draws (+0.33289/-0.74597s);25% is positive
in both (+0.12253/+0.14848s). Full fold/source/seed metrics and paired strata
are saved; see `subset_metrics.csv`, `fold_metrics.csv`, `source_metrics.csv`,
`seed_metrics.csv`, `stratified_paired_deltas.csv`.

Pooled MAE gives a different MLP conclusion:

| Fit groups | M-S pooled MAE | M-P pooled MAE |
|---|---:|---:|
|10%|1.873146|2.890215|
|25%|1.577916|2.439497|
|50%|1.338083|2.443088|
|100%|0.989635|1.114466|

This reversal must be disclosed. QPack supplies87.4% of rows but only6/165
groups. Unstratified group draws can omit whole sources:8/10 low-fraction
fit plans omit QPack;4/10 contain Ma-Li only. Consequently, fitting-row counts
at10% range18–1746. This is the declared group experiment, not uniform
within-source label reduction. Group-balanced gains do not imply pooled
performance gains or universal cross-source sample efficiency.

## Tail behavior and optimization

All360 histories contain500 sequential epochs, finite validation MSE, and
correct minimum-validation-MSE checkpoint selection. Predictions are finite,
with exact targets/IDs and no missing held-out rows. Tail failures remain:
the median-three-seed output can have errors above100s at10% and pretrained
MLP still has large tails at50%. `tail_diagnostics.csv` retains the worst10
observations for every treatment/fraction/subset; `model_results.csv` includes
P90/P99/max,bias and pooled R2. No tail was excluded or clipped.

There is no saved per-epoch training loss/gradient history, so train–validation
gap and optimization convergence are not fully auditable from histories.
Equal500 epochs do not mean equal optimizer updates across subset fractions.
These conditions are matched within each pretrained/scratch pair.

## Inference limitations

- Bootstrap samples165 complete structural groups; it does not treat repeated
  rows or the two subset draws as independent additional groups.
- Intervals condition on saved training runs and two draws. They do not
  estimate the full variability over possible training subsets or retraining.
- The final analysis specification was locked after training and before
  aggregate treatment effects were computed; it is not a preregistered study.
- Inherited bootstrap sign-tail scores and their Holm-adjusted diagnostics
  are clearly labeled in `paired_comparisons.csv`. They are not claimed as
  calibrated confirmatory frequentist p-values. Original A1 reporting remains
  untouched.
- Inputs lack shots and backend-ID conditioning; this is a circuit-only
  transfer experiment on heterogeneous archived execution/service labels.
- For pretraining provenance, qualified Graph checkpoints can reside under
  a successful attempt subdirectory. Validation followed the original
  receipt-qualified hash resolver rather than an obsolete top-level file.

## Thesis-ready claim

Under the fixed circuit-only transfer contract, simulator initialization
reduced group-balanced error for the MLP at10% and25% of QPU fitting groups
in the two tested subset draws, with conditional group-bootstrap intervals
below zero. This benefit was not established at50% or100%, nor for Graph+MLP
at the tested fractions. Pooled MLP errors increased under pretraining,
demonstrating that the apparent benefit depends on evaluation weighting.
The experiment retains full validation support and does not establish a
general reduction in total QPU-label requirements, shot-conditioned transfer,
or statistical equivalence where intervals cross zero.

## Artifacts and reproduction

Primary receipt: `validation_receipt.json`; group-level effect inputs:
`paired_group_deltas.csv`; replayable draws/distributions:
`bootstrap_distributions.npz`; input provenance: receipt `input_hashes`.
Plots: `transfer_effect.png`, `absolute_performance.png`, `heterogeneity.png`.

```bash
CUDA_VISIBLE_DEVICES='' /home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python research/postfreeze_track_a_transfer_sample_efficiency_v1/test_aggregation_v2.py -v
git diff --check
```

The aggregation entry point is `aggregate_validated_v2.py`; it refuses to
overwrite its existing output directory. Reproduction should use a separate
reviewed output namespace or the saved-output replay command above.
