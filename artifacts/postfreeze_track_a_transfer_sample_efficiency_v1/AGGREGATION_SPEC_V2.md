# P1 saved-prediction aggregation specification v2

Locked before executing the final aggregator. Training is complete; interim
validation inspected technical records and prediction extremes, but did not
calculate the aggregate transfer contrasts. This is a post-training analysis
specification, not a retrospective claim of preregistration.

Output namespace: `aggregation_v2/`. Historical partial reports, training code,
all cell artifacts and A1 results remain unchanged. No model inference/refit.

Require exactly360 completed non-100% cells with verified hashes,500 sequential
epochs, minimum-validation-MSE checkpoint selection, exact frozen test IDs and
labels, paired transforms, nested target-blind fit-group selection and disjoint
fit/validation/test groups. Reuse100% predictions and original A1 intervals.

Within each fraction/subset/treatment, take the median prediction over exactly
seeds42/1234/31415 per observation. Compute row absolute errors, average inside
each structural group, then average groups equally. Across the two fixed subset
draws, average errors, not predictions. Thus primary delta is the equal-group
mean of pretrained-minus-scratch absolute-error differences, averaged over the
two subset draws. Each draw must cover4515 rows/165 groups for each treatment.

Use the saved A1 seed42 whole-group bootstrap draw matrix:10000 draws of165
groups with replacement, shared across fractions, architectures and subset
draws. Save the matrix and all bootstrap effect distributions. Percentile95%
intervals are conditional on the frozen folds, training runs and two subset
draws; no claim of unconditional training-subset uncertainty.

For compatibility, report the inherited A1 two-sided bootstrap sign-tail score
with +1 correction, explicitly as a diagnostic rather than a calibrated
frequentist p-value. Holm adjustment of these scores across the six new
fraction/architecture contrasts is also diagnostic. Additionally report
Bonferroni percentile sensitivity intervals across six contrasts, with the
finite10000-draw tail-resolution limitation. Do not claim confirmatory
significance from either diagnostic. Original100% A1 p/CI records are reused
without reinterpretation or inclusion in the new family.

Report separate subset,fold,source,model-seed metrics; pooled MAE/RMSE/MedAE,
pooled R2,source-macro MAE,bias,P90/P99/max errors and coverage. Never average
fold R2 to obtain pooled R2. For pooled R2 over the two subset runs, duplicated
targets represent repeated evaluation outcomes and remain descriptive, not
additional independent observations. Tail observations remain scored.

Limitations: fraction applies to gradient-fit groups only; validation labels
remain fully available. Unstratified group sampling changes source support and
row counts. Inputs exclude shots. Two subset draws limit uncertainty claims.
Final scientific interpretation is permitted only after validation PASS.
