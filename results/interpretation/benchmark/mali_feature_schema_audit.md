> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Ma–Li global-feature schema audit

Interpretation addendum to Final Scientific Freeze v2. Decision **B — reporting fix only**.
No fitted model, prediction, split, circuit, calibration or protocol was changed.

## Raw positions and effective inputs are different quantities

The pinned upstream revision is `32c392a6ece276f1ff046d4e30052d0571ff6dc6`.
`get_openqasm_gates` supplies 44 ordered gate counters. `create_feature_dict`
appends width, depth and five SupermarQ structural summaries: 51 raw positions.
`refine_training_data` retains a position when its sum over the supplied dataset
is positive. It does **not** select the first 41 positions or hard-code a
41-feature list. Counts/structural values in the audited populations are
nonnegative, so positive-sum support and at-least-one-nonzero coincide here.

The [paper, §5.1](https://arxiv.org/html/2411.15631v1) describes 34 gate counters,
width/depth and five structural features, totaling 41. The
[pinned helper](https://github.com/mooselab/Quantum-Execution-Time-Prediction/blob/32c392a6ece276f1ff046d4e30052d0571ff6dc6/data_preparation/helper.py)
explains how a broader extractor yields that supported input. This is source
and data reconstruction of the mechanism, not an assertion that the authors
published their processed training tensor.

| Population/stage | Raw width | Supported/tensor width | Verification |
| --- | ---: | ---: | --- |
| Extractor | 51 | Not yet refined | 44 gate + 2 size + 5 structure |
| Original simulator corpus | 51 | 41 | 1,510 named pinned QASM × two timing contexts = 3,020 rows |
| Published original real-QPU corpus, recomputed support | 51 | 40 | 340 timing rows on Osaka/Kyoto; not proof of the authors' executed tensor |
| Current parent gradient fit, outer folds 0–4 | 51 | 40 in every fold | Exact fit-ID mask, mean, standard deviation and variable mask replay |
| RQ4 final refits | 51 | 11, 22 or 40 | Depends on case train support |
| RQ4 epoch-selection fits | 51 | 11, 17, 22 or 40 | Depends on the smaller inner-fit population |

All 1,510 working QASM bytes and four timing CSVs were compared with their
pinned Git blobs. Static features were re-extracted with the pinned helper;
there was no simulation or timing measurement. The complete file hashes and
support counts are in `original_corpus_support.json`. Modern parser/library
versions and missing author-processed tensors limit this to reproducible
source-feature reconstruction, not independent recovery of the authors' tensor.

### Upstream notebook/default-model inconsistency

The pinned `main.ipynb` cell 16 actually stores the 41 retained raw indices,
matching our simulator reconstruction. Cell48 stores only 12 indices for its
real-QPU example: `[3,11,17,23,31,44,45,46,47,48,49,50]`. That is not the 40
supported positions reconstructed from the released340 timing rows/QASM.
The public Graph constructor and notebook global-only model default to41
inputs; `builder.make_model` does not override that default. This default is
distinct from the support-based refinement function. The active-learning
padding function pads **node rows**, not global width. No published processed
real tensor resolves the 12/40/41 discrepancy. We do not infer which notebook
state produced the paper's real score or silently fabricate a padding fix.

`upstream_pipeline_evidence.json` preserves the exact pinned code/cell outputs,
model default and Git hashes. Ledger `retained_original_real_refinement`
means **recomputed refinement on the released340 subset**, not a verified
author-execution mask. Current40 is directly verified from saved transforms.
Original real executed input width remains unverified. The public simulator
SHAP setup uses the 41-field list; its existence is not local 4515 attribution.

## Exact positions removed

Zero-based raw positions removed from the original simulator population:

| Index | Feature | Nonzero simulator rows / real rows / qualified rows |
| ---: | --- | --- |
| 4 | id | 0 / 0 / 0 |
| 5 | u0 | 0 / 0 / 0 |
| 33 | csx | 0 / 0 / 0 |
| 34 | cu | 0 / 0 / 0 |
| 38 | rc3x | 0 / 0 / 0 |
| 39 | c3x | 0 / 0 / 0 |
| 40 | c3sqrtx | 0 / 0 / 0 |
| 41 | c4x | 0 / 0 / 0 |
| 42 | xx_plus_yy | 0 / 0 / 0 |
| 43 | ecr | 0 / 0 / 0 |

Raw position **37, rccx**, is nonzero in 12 original simulator observations
(six named circuits in two contexts), but zero in all 340 original real rows
and all 4,515 qualified observations. Thus the real/qualified support is
51 − 10 − 1 = **40**. Neither rccx nor the other ten can provide a learned
signal in the current cohort. There is no saved isolated rccx ablation proving
its incremental value in the original simulator population.

`global_feature_ledger.csv` contains exactly 51 rows, including per-source
nonzero counts, original refinement decisions, current transform membership
and held-out-only activation. `masked_feature_concentration.csv` records
source/family/backend support for the eleven suspect positions: all counts
are zero. These eleven positions do **not** explain Random's lost fields;
Random loses additional fields that exist elsewhere in the qualified cohort.

## Every current transform

`effective_input_widths.csv` covers **257** saved transform files: five parent
transforms shared by Graph/MLP and all three seeds, plus 21 cases × two stages
× two methods × three seeds. For each file the exact recorded fit IDs were
used to recompute positive-sum retention and per-feature moments; saved
retained indices, means, sample standard deviations and `std > 1e-6` masks match.

Parent models have one gradient-fit population per outer fold: inner folds
1–3; inner fold 0 selects the epoch. They are not four independently refitted
neural inner models. Regression uses four inner validations and a full outer
refit. RQ4 separately selects an epoch then freshly refits complete outer train.
There is no fabricated four-inner-transform table for neural cells.

| Case | Selection tensor width | Final tensor width |
| --- | ---: | ---: |
| qft, qftentangled, qpeexact, qpeinexact | 22 | 40 |
| qnn, realamprandom, su2random, twolocalrandom | 40 | 40 |
| random | 22 | 22 |
| Jakarta, Lagos, Manila, Perth | 17 | 40 |
| Belem, Kyoto, Lima, Nairobi, Osaka, Quito | 40 | 40 |
| QPack train_through_4 | 11 | 11 |
| QPack train_through_5 | 11 | 11 |

Selections/refits are identical in width across Graph/MLP/seeds within a case.
For QPack train_through_4 only 9 selection and 10 final columns vary; for
train_through_5 there are 10 varying columns in each stage. Positive constant
columns remain tensor positions but are zeroed by the variable mask. Therefore
tensor width is not always the number of varying signals. Parent width40 has
40 varying columns. Node raw width178 and its own mask are separate quantities.

Ledger `retained_transform_row_indices_json` indexes **zero-based CSV data
rows**, excluding the header, in `effective_input_widths.csv`. The first five
are outer folds0–4. Feature-support arrays use zero-based raw 51 indices.

## What fidelity is retained, and what is adapted

Raw schema, gate ordering and positive-support refinement follow upstream;
the original constructor's default 41 is replaced by the fitted supported width
in these fresh local models, not a claim of loading the author's weights.
Current refinement and moments are fit-only; validation/test cannot retain a
field or alter its normalization. Retention is then frozen for that fitted model.
The upstream helper refines before dataset splitting. Its public normalization
also concatenates one-dimensional global vectors, giving scalar global moments;
the current implementation uses per-feature fit-only moments. This difference,
nominal-index calibration, reconstruction qualification, grouped splits and
training policy mean **architecture/core adaptation**, not exact paper execution.
It is not additional non-paper input smuggled through the ten zero columns.

The comparison remains valid for the declared frozen adaptation. Forcing the
simulator's fixed41 list would keep all-zero rccx in the parent and override
fit-support behavior in OOD cases; it would not restore the paper's full
preprocessing/training environment. No numerical result needs rerunning to
make a displayed dimension equal41. Exact public-code/paper-protocol
reproduction would be a different study, not a repair of these saved results.

## Reporting correction

Use: **“51 raw global positions; 40 retained/varying positions in each parent
gradient-fit transform; fit-dependent support in RQ4.”**
Do not use “51 actual global inputs” for parent Graph/MLP or “fixed41 schema”.
Random retains 22 global positions and masks18 test-only fields. QNN retains 40
and masks no test-only global field, yet fails catastrophically. Support loss
and extreme scaled counts are descriptive failure diagnostics, not causal
importance estimates or a complete explanation of all OOD failures.

Evidence: `feature_validation.json`, `original_corpus_support.json`, the 51-row
ledger,257-row width table and `generalization_field_support.csv`. Source pins
and frozen-authority preservation checks are bound by `manifest.json`.
