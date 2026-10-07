# Methodology

The study compares complete prediction pipelines on shared held-out observations.
Real-QPU and simulator targets remain separate. Within each simulator context,
the engine, precision, shots, timing boundary, quality policy and target revision
are part of the comparison identity. Each method's inputs are declared; a shared
dataset does not imply identical representations or paper-exact reproduction.

## Counting evaluation outcomes

Use the reader row's stated unit: QPU observation, exact circuit hash,
hash/configuration pair, measurement session or analog cell. A session count is
not a unique-circuit count. Aliases and neural seeds follow the frozen reduction
contract; three seeds are not three methods.

| Quantity | Meaning |
| --- | --- |
| Assigned | All identities selected for the method/context before inspecting prediction errors |
| Attempted | Assigned identities on which prediction was attempted; not training epochs or timing repetitions |
| Observed | Identities with an admissible measured target for the stated clock |
| Predicted | Identities with an admissible emitted prediction; a target can still be unavailable |
| Scored | Identities with both an admissible prediction and target on the assigned held-out rows |
| Unavailable | Missing required input, prediction or target support, with the producing ledger's reason retained |
| Failed | Explicit technical/computation failures; not high error or low fidelity |
| Quality assessed / pass / fail | Separate approximation-quality outcomes under the declared reference and threshold |

Predictor coverage is scored/assigned. These counts are not all disjoint: a
prediction may exist for an unavailable target, and a quality-failed approximation
may have a valid measured runtime. Quality coverage and technical success are
not runtime evaluation coverage. Legacy fields retain their source status-ledger
meaning rather than being silently reinterpreted.

The simulator CSV uses `expected_assigned_n`, `attempted_n`, `predicted_n` and
`metric_n` for the corresponding predictor counts. The QPU CSV uses `assigned`,
`observed` and `predicted`; for its 13 selected routes, every emitted prediction
has an observed target, so predicted also equals scored. This is not a general
benchmark property. QCSim controls emit 150 predictions but score 143 targets;
joint MPS emits 900 predictions but scores 840 targets. Native measurement rows
use session/stage counts and are not predictor scores.

## Comparisons and missing data

MAE, R² and tail errors describe the scored row set. Pair methods only on
identical successful held-out IDs under the same target and quality policy.
Retain the assigned denominator and unavailable/failure reasons alongside that
intersection. Lower error on a smaller supported subset is not a full-panel win.
R² is undefined for fewer than two targets or zero target variance; do not
replace an undefined metric with zero.

Do not infer missing labels from reconstructed circuits, nominal schedules or
predictions. Valid zero gate counts and explicitly supplied zero-duration virtual
instructions are not missing-data imputation. Low accuracy remains a result;
technical correctness must be assessed separately.

## Input evidence and method fidelity

The current QPU panel contains 7,350 observations from an 8,767-observation
archive: 340 Ma–Li source logical inputs, 230 Qonductor archive-supported recipes,
2,835 Qonductor unverified candidate recipes and 3,945 QPack representative-angle
structural reconstructions. Only inputs are reconstructed; labels remain archived
observations. Recipe availability does not prove exact historical logical-circuit
recovery. The 1,417 excluded Qonductor observations remain outside this panel.

Unified learned and calibrated analytical pipelines are adaptations. Native
cuTensorNet estimates apply to the selected contraction plan. Pasqal is an analog
companion. None replaces an unavailable original-paper route without disclosure.
See [data construction](data_preprocessing.md),
[feature definitions](data_and_feature_dictionary.md),
[training and measurement](training_and_measurement.md), and
[paper comparison](paper_comparison.md) for implemented rules.

The two [result tables](results.md) report this study scope.
[Reproduction scope](reproduction.md) distinguishes checking saved evidence from
recovering external data, fitting models and timing engines.
