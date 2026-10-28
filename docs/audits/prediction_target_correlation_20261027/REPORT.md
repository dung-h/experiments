# Saved prediction versus observed-target correlation

Read-only descriptive analysis, 2026-10-27. No fitting or execution. Scientific
Freeze v2 is unchanged. Input hashes and Git HEAD are in `validation.json`.
Reproduce with `/home/server/Documents/.venv-qonductor/bin/python
docs/audits/prediction_target_correlation_20261027/evaluate.py` from the repository.

## Population and measurement

Original saved OOF artifact: `artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv`.
NoTaDS/HyperQ inputs: `artifacts/postfreeze_notads_hyperq_track_a_v1/{predictions,features}.csv`.
Canonical IDs, sources, targets, groups and folds agree across inputs: 4,515
observations / 165 structural groups. Sources: Ma–Li 340, Qonductor 230,
QPack 3,945. Targets retain frozen source-specific observation boundaries;
the common label field does not establish identical physical timing boundaries.

`correlations.csv` contains Pearson linear correlation and Spearman rank
correlation, both observation-weighted, overall and within each source.
Only finite saved predictions with status `predicted` are scored. Coverage is
explicit. Partial methods must not be ranked against full-coverage methods as
though the populations were identical. No CIs, significance or causal claims
are assigned. QPack's 3,945 rows dominate observation weighting; 165 groups
are not 4,515 independent circuits. No new group-balanced primary estimand is
substituted for the frozen benchmark.

## Selected results

| Method | Rows | Pearson | Spearman |
|---|---:|---:|---:|
| QCRE unitary raw | 4515 | 0.526405 | 0.492497 |
| QCRE unitary × shots | 4515 | 0.540523 | 0.508718 |
| Qiskit full raw | 4515 | 0.526749 | 0.535290 |
| Qiskit full × shots / HyperQ paper replay | 4515 | 0.542681 | 0.552843 |
| NoTaDS code raw score | 4515 | 0.529894 | 0.574663 |
| NoTaDS code score × shots | 4515 | 0.552859 | 0.584479 |
| Qiskit full × shots, OOF affine | 4515 | 0.518749 | -0.017257 |
| NoTaDS code × shots, OOF affine | 4515 | 0.531968 | 0.249922 |
| Hyb effective cost raw | 4175 | 0.287300 | 0.446308 |
| Hyb log-cost Ridge adaptation | 4367 | 0.545989 | 0.458360 |
| Frozen Graph+MLP | 4515 | 0.740684 | 0.765037 |
| Frozen MLP | 4515 | 0.642627 | 0.552484 |
| Frozen regression budgeted selector | 4515 | 0.681507 | 0.579015 |

NoTaDS raw scores have abstract units, not seconds. Correlation permits
comparing association despite units; this does not license a seconds MAE.
HyperQ replay duplicates Qiskit's saved values on these inputs and is not
independent evidence.

## Source dependence

For Qiskit duration × shots, Pearson/Spearman are 0.828266/0.815893
on Ma–Li, 0.620597/0.516894 on Qonductor, and 0.408607/0.434858
on QPack. NoTaDS code × shots gives 0.828476/0.819630,
0.570421/0.506530 and 0.424214/0.474433 respectively.
Thus structural scores carry association, but its strength is source-dependent.
Repeated workload/context composition can contribute to association; neither
pooled nor within-source correlation identifies a physical mechanism.

## Calibration and rank diagnostic

Low raw accuracy can coexist with positive correlation: a correct trend can
have the wrong scale/offset. An illustrative prediction y/1000 has perfect
correlation with y while badly missing seconds.

A single positive affine map preserves ranks. Saved OOF calibration uses five
different outer-train fitted maps, not one global map. Fold-specific offsets
can reorder observations across folds. In `fold_rank_check.csv`, raw and
affine Qiskit shot-scaled predictions have identical Spearman correlations
within each of the five test folds, despite the pooled rank drop. This is
consistent with different fold maps and does not by itself establish a bug.
It is a real limitation of global ranking across these OOF outputs. Improved
MAE is not evidence of improved ranking or a recovered physical mechanism.

## Validation

Unique method/observation keys, canonical target/source/group/fold alignment,
finite-value filtering and deterministic replay PASS. Pearson calculations
use positive rescaling to avoid overflow in large effective costs. Existing
frozen metrics, models and splits were not edited. No new models, circuits,
simulator jobs or QPU jobs were run. No commit or push.
