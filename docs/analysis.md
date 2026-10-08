# How the results are analysed

The input to analysis is saved out-of-fold predictions joined to observed
labels by observation ID, method, context and fold. An observation can be
one execution of a circuit in a particular backend, shot and calibration
context; repeated contexts are not treated as distinct circuit structures.
The analysis does not fit predictors or generate missing runtime labels.

## Populations and missing results

The real-QPU population contains 4,515 observations in 165 groups. Selection
uses source provenance and input qualification, not prediction accuracy. Its
source imbalance and reconstruction uncertainty remain part of the analysis.
See [preprocessing](data_preprocessing.md) for original-source inclusion and
exclusion counts. Fresh QPU scores, source slices and same-row comparisons
are in the [QPU table](../results/real_qpu/method_comparison.md).

Simulator member counts, exact-QASM counts, context counts, measured-label
counts and predictor counts are different denominators. A hash measured at
multiple precision or bond-dimension settings produces several context rows.
There is no single score that merges Aer, dense sampling, MPS extraction,
tensor contraction and analog emulation.

For every method/context, report assigned rows, available measured labels,
finite predictions and terminal failures. Timeout, adapter error, unsupported
input and invalid analytical output are separate statuses, not zero runtimes.
Compute accuracy only on rows with an admissible prediction and label. Compare
two methods on their shared successful held-out IDs, and state how many rows
the intersection excludes. A lower subset MAE is not evidence of full-envelope
superiority when the difficult observations are missing.

## Prediction reduction

For multi-seed neural cells, retain each seed's output and reduce per-row
predictions by the declared median. Do not choose the seed with lowest test
error. Source slices reuse the same unified fits and held-out predictions.
They do not train a separate model per source.

Measurement reductions follow the engine's contract. Do not substitute a warm
call, a first call or an end-to-end measurement for another clock. The
process-isolated QCSim engine-time reduction is not persistent warm execution.
MPS all-finite runtime and quality-pass runtime are separate evaluations.

## Error measures

For observed time `y` and prediction `p`, absolute error is `|p-y|` seconds.
MAE is its mean; MedAE is its median. Tail summaries include p90, p99 and
maximum absolute error. Inspect the largest errors with their source, family,
width, depth, shots, fold and reconstruction tier.

R² is `1 - sum((p-y)^2) / sum((y-mean(y))^2)` on the evaluated row set. Its
denominator changes with the row set. R² can be negative when tail errors are
large even if typical absolute errors are modest. Compare R² only when labels
and successful IDs match. Undefined metrics are not filled with zeros.

Log-MAE is the mean of `|log1p(y)-log1p(max(p,0))|`. The prediction floor
applies only to this metric; finite negative polynomial predictions remain
in ordinary seconds-based MAE and R². This is error on a shifted log scale,
not the logarithm of a raw-time ratio, especially for subsecond labels.
A model's training transform need not define its reported error metric.

For QPU observations, show pooled error and the mean of source-specific MAEs.
The latter gives Ma–Li, Qonductor and QPack equal source weight instead of
allowing the largest archive to dominate. Within-source correlation of runtime
with depth/width/shots is descriptive; it does not establish that one feature
causes runtime or that scheduled gate time equals observed service time.

## Paired uncertainty

Bootstrap comparisons resample the frozen circuit/workflow groups rather than
individual context rows. Both methods use the same resampled groups in each
replicate. The point estimate is the observed paired MAE difference, not the
mean of bootstrap replicates. Report the grouped 95% interval, bootstrap seed,
10,000-replicate count and the row population.

A difference interval crossing zero leaves the comparison unresolved at that
uncertainty level. Absence of a resolved difference does not prove equivalence.
Few independent QPack structural groups and highly unequal group sizes limit
how broadly the interval can be interpreted.

## Features and rare inputs

Graph versus matched global MLP compares adding the full node/graph branch,
including T1/T2 context. It is not a topology-only ablation. Source versus
transpiled Aer GNN addresses representation in the measured local context;
it is not proof of a universal graph advantage over regression.

Exact feature-signature overlap audits test reuse of recorded representations,
not unseen quantum semantics. Family conditioning is assessed by a paired
family-aware versus agnostic model, with both runtime error and MPS quality
violations. An isolated dynamic-circuit diagnostic cannot establish population
accuracy. Do not invent rare-class findings when support or labels are absent.

## What can be concluded

See [results](results.md) for the two current method tables and
[paper comparison](paper_comparison.md) for original claims, local evidence
and limits. Preserve unresolved comparisons and failed quality policies as
findings. There is no universal winning method across these engines and clocks.
The benchmark is retrospective and exploratory, not a fully analyst-blind
prospective preregistration. It does not claim exact reproduction of every
paper's numerical result or prove a general runtime law.
