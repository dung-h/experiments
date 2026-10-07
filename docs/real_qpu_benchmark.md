# Real-QPU runtime benchmark

## What this table compares

The current common-panel comparison is the **7,350-observation logical-input
panel** in [the reader table](../results/real_qpu/method_comparison.md) and its
[machine-readable CSV](../results/real_qpu/method_comparison.csv).
It predicts each archived one-circuit service-time label using frozen grouped
out-of-fold predictions. It is not the same population as the earlier
8,767-row compiled-input ledger or its 8,766-row compiled comparison. Those
older results remain in the historical reader; do not pool the tables or call
the 7,350 observations “all 8,767.”

The panel keeps every included source row and reconstruction tier visible:

| Source | Included rows | Input evidence in this panel | Qualification |
| --- | ---: | --- | --- |
| Ma–Li | 340 | Exact source logical QASM; target is the mean of three archived `Result.time_taken` values; 1,024 shots | Source logical input |
| Qonductor | 3,065 | 230 archive-supported logical-recipe adaptations and 2,835 candidate-recipe sensitivity rows | The remaining 1,417 of the original 4,482 one-circuit jobs are outside this logical-input panel; their submitted physical QASM remains in the separate compiled-input evidence |
| QPack MCP | 3,945 | Six pinned angle-insensitive QAOA structures rebuilt with representative angles | Reconstruction-qualified; optimized angles, submitted QASM and routing were not recovered |
| **Total** | **7,350** | **Common observation IDs, targets and frozen folds** | **Not 7,350 exact original logical circuits** |

The source labels are not newly generated from reconstructed circuits. The
three repositories' job/service boundaries are not guaranteed identical just
because each archive exposes an execution-time field. QPack rows are
one-circuit optimizer evaluations, not workflow-duration labels. Qonductor's
4,482 source jobs were verified as one-circuit jobs; only 3,065 are members of
this logical-input common panel.

## Split, features and fairness

The panel uses 206 transitive groups and one frozen five-fold group split.
No group is divided across train, validation and test. The graph and matched
MLP use the same 51-field global input contract; the graph model additionally
uses a 178-field directed DAG representation. Both use three registered seeds
(42, 1234, 31415); the reader prediction is the median across all three, never
the best seed. Qonductor-style polynomial and analytical rows use the same
outer folds and their own declared features. Every row reports assigned
coverage, target clock, method-output clock and fidelity class in the reader
table. An outer-train affine calibration is a separate adaptation, not a raw
scheduled-duration prediction.

There are four grouped inner folds per outer fold. The neural models fit on
inner folds 1–3, fit their masks/scales on those same inputs, and select an
epoch with inner fold 0 over 500 epochs; they do not then refit on validation.
All five masks retain 40 of the raw 51 globals. Polynomial selects degrees
2/3/4 by mean inner R² and refits on full outer-train; all folds choose degree
2. It keeps the upstream CX/CZ/ECR counter named `swap`, upstream operand-stack
depth on compiled/submitted inputs, touched width, shots and circuit count=1,
without log/scaling/pruning. Hyb Ridge selects alpha by source-balanced inner
MAE and refits on eligible
outer-train. These are declared method-specific procedures, not identical
input views: polynomial and timing methods use compiled/submitted inputs.

The execution record discloses inspection of aggregate fold-1 fit/validation/
test target summaries before all predictions were complete. No protocol change
in response is reported. The reviewed implementation excludes outer-test
labels from fit, transforms, selection and calibration, but the study is
retrospective and exploratory rather than fully analyst-blind/preregistered.

The [data and feature dictionary](data_and_feature_dictionary.md) defines the
source joins, input tiers, all 51 global positions, all 178 node positions,
snapshot use and train-only transforms. The
analysis manifest (not bundled)
pins the exact files, IDs, transformations and bootstrap settings.

## Results on the 7,350-row panel

The three learned models below predict all 7,350 assigned rows. Values are
from the same saved out-of-fold IDs:

| Method | Reader fidelity | Scored / assigned | MAE (s) | R² |
| --- | --- | ---: | ---: | ---: |
| Ma–Li-style logical graph + global features | Architecture adaptation | 7,350 / 7,350 | 1.219 | 0.429 |
| Matched Ma–Li-style global MLP | Architecture adaptation | 7,350 / 7,350 | 1.732 | 0.187 |
| Qonductor-style unified polynomial | Adaptation | 7,350 / 7,350 | 1.401 | −0.587 |

The graph-minus-MLP MAE difference is **−0.514 s** (10,000 group-bootstrap
replicates, 95% interval **[−0.907, −0.175] s**). This is evidence that the
graph adaptation has lower error than this matched global MLP on these frozen
out-of-fold predictions. The added branch includes node features and nominal
T1/T2 context; this comparison does not isolate topology alone. The interval
is conditional on the saved fits; it
does not include retraining uncertainty, and the comparisons are exploratory
and not multiplicity-adjusted. The graph-minus-polynomial difference is
**−0.182 s** with interval **[−0.488, 0.197] s**, which includes zero. These
results do not show that a graph model is universally better than regression
or generalizes to new backends, families or QPUs.

Other evaluated rows are adaptations or analytical references, not a single
flat ranking:

| Method / output variant | Scored / assigned | MAE (s) | R² | Interpretation |
| --- | ---: | ---: | ---: | --- |
| QCRE shot-scaled schedule, outer-train affine | 7,350 / 7,350 | 1.555 | 0.675 | Calibrated service-time adaptation; raw schedule remains a distinct clock |
| Hyb-HANAS-style nominal log-cost Ridge | 7,202 / 7,350 | 1.370 | 0.342 | 148 rows unavailable; do not impute them |
| Hyb-HANAS-style Kyoto composite log-cost Ridge | 7,350 / 7,350 | 1.369 | 0.355 | Cost-feature calibrated service-time adaptation |
| Hyb-HANAS-style Kyoto composite gate-time Ridge | 7,350 / 7,350 | 1.371 | 0.416 | Gate-time-feature calibrated service-time adaptation |
| Scholten-style nominal throughput, outer-train affine | 5,199 / 7,350 | 1.364 | 0.708 | Nominal sensitivity adaptation; 2,151 unavailable, not a full-panel win |

The reader also includes raw QCRE/Qiskit schedule and raw Scholten-style
throughput rows. Their output clocks differ from archived service execution
time. Their MAE/R² against the service-time labels are diagnostics, not
like-for-like measured-service-time estimates. The table preserves those
clock names instead of silently treating schedules or throughput proxies as
ground truth.

Source slices of the same unified fits give polynomial lower MAE on Ma–Li
and Qonductor, and graph lower MAE on QPack. Polynomial p99/max absolute error
is 4.003/187.054 s versus graph 7.082/76.094 s: MAE, squared-error R² and tail
criteria need not rank methods identically. Negative R² is retained as a
result, not taken alone as proof of faulty implementation.

## Reconstruction sensitivity and input support

The 4,515-row reconstruction-qualified sensitivity subset removes only the
2,835 Qonductor `candidate_recipe_sensitivity_only` rows. It **reuses the same
OOF predictions from models fitted on 7,350 rows**; it is not a separately
trained 4,515-row benchmark. On that subset, graph MAE is 0.716 s and
polynomial MAE is 1.164 s. The subset result is descriptive sensitivity to
input qualification, not evidence that training on 4,515 rows would produce
those scores.

The exact transformed global, graph and graph-plus-global tensor signatures
have zero repeated training-fold matches in this split. These are exact
floating-point model-input signatures (including the saved transform), not
circuit identities. Unique signatures do not prove family-, backend- or
topology-held-out generalization. Conversely, repeated signatures would not
prove quantum equivalence. The per-row evidence is in
`input_support.csv` (not bundled).

QPack contributes 3,945 rows but only six workflow groups; Qonductor contributes
52 groups and Ma–Li 148. Grouped intervals and source-balanced metrics are
reported for that reason, but six QPack groups still limit claims about new
workflows. Full source/tier/backend/shots/fold slices, tail errors and masks
are in the linked analysis CSVs. Poor R², unavailable predictions and large
tail errors remain results; no rows were removed to make a model look better.

## Reproduction boundary

This public checkout retains QPU aggregate tables and implementation sources,
but excludes the row-level labels, predictions and partition ledgers needed to
recompute these QPU scores. Earlier local replay receipts describe a larger
private evidence package. They do not attest fresh-clone QPU reproduction of
this reduced public selection. [Reproduction scope](reproduction.md) lists the
checks available here, including saved simulator scores. Complete upstream
archives and QPack optimized-angle submissions remain external; see
[data availability](distribution.md).
