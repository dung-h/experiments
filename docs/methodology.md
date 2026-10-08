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
`metric_n` for predictor counts. The QPU assigned denominator is 4,515 for
every method. Its CSV uses `assigned_rows`, `predicted_rows` and `coverage`;
all current QPU observations have valid targets, so the successful prediction
count is also the scored count. The 32 variants include six regressor families,
an inner-selected family selector and distinct raw/calibrated analytical routes;
they are not 32 paper methods.
QCSim controls emit 150 predictions but score 143 targets; joint MPS emits
900 predictions but scores 840 targets. Native measurement rows use session
or stage counts, not predictor scores.

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

The QPU dataset contains 4,515 observations: 340 Ma–Li source logical inputs,
230 Qonductor archive-supported recipes and 3,945 QPack representative-angle
structural reconstructions. Only inputs are reconstructed; all labels remain
archived observations. Recipe qualification does not establish exact historical
logical-circuit identity. Source-specific exclusions are documented directly
from the original source audits.

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
