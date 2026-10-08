# Real-QPU benchmark

## Dataset and target

All methods use the same **4,515 observations**: Ma–Li 340, Qonductor 230
and QPack 3,945. The sources supply observed circuit-execution/service time,
normalized to seconds. No live QPU is used and no reconstructed circuit
creates a runtime label.

The [dataset report](data_preprocessing.md) explains the source screens,
retained tiers, label boundaries, evidence and missing data. The input population
has 165 conservative groups and is reconstruction-qualified, not a collection
of 4,515 exact recovered historical logical circuits.

## Method inputs

| Method family | Declared input and implementation | Interpretation |
| --- | --- | --- |
| Ma–Li-style graph + globals | Source/qualified logical DAG, 51 ordered raw globals, 178 node positions; nominal-index T1/T2 | Graph architecture adaptation; original feature schema retained |
| Matched global MLP | Same raw global schema, train-only transforms and neural selection as graph | Test of adding the complete node/graph branch, not topology alone |
| Qonductor-style regressors | CX/CZ/ECR count under upstream name `swap`, operand-stack depth, touched width, shots, circuit count = 1 | Six regressor families, bounded nested search |
| QCRE | Pinned author `estimate_runtime` through BQSKit, nominal native durations and compiled/submitted unitary view | Scheduled estimate; terminal measurements and barriers omitted explicitly |
| Qiskit duration reference | Nominal Target and declared scheduled single-shot semantics | Scheduled duration, not provider execution time |
| Hyb-HANAS-style component | Audited nominal gate/cost components; separately calibrated service-time variants | One-circuit cost adaptation, not the full NAS objective |
| Scholten-style nominal throughput | Same-backend nominal throughput and compiled wire-depth proxy | Not original quantum-volume-normalized effective depth; original route unavailable when its inputs are absent |

Compiled inputs retain exact Qonductor submissions where admissible. Ma–Li
logical and QPack representative circuits use their recorded nominal target
compilation. Different methods can consume different views of the same
observation, but cannot replace its observed target or test assignment.
Numerical angles and shots are not silently appended to the upstream
Ma–Li feature schema.

The six regressors are Extra Trees, Random Forest, Gradient Boosting, AdaBoost,
Histogram Gradient Boosting and Polynomial Regression. The upstream `swap`
feature counts CX/CZ/ECR operations; it is **not** a literal SWAP-gate count.
The bounded search is disclosed as an adaptation of the upstream model/grid
rather than a repeat of its full search.

## Common train/test procedure

Five outer folds assign whole groups; each observation is tested once.
Four inner folds within outer train support tuning. All sources occur in every
outer test and inner validation set. Outer seed is 42; inner seeds are 43–47.
[Dataset metadata](../protocol/real_qpu_dataset.json) records exact counts
and excluded-input digests.

Regressors tune over the four inner folds and refit outer train. Graph and MLP
fit on inner folds 1–3 and select the epoch on inner fold 0; outer test never
selects an epoch, transform, model family or hyperparameter. Neural seeds are
42, 1234 and 31415, reduced by per-observation median after all three finish.

All learned models are fitted anew on this population. Calibration parameters
for analytical methods are also fitted anew on each outer train set. Reusing
an audited raw schedule is not reusing a fitted prediction.

## Clocks and calibration

Every accuracy table uses
`evaluation_target_clock = archived_observed_service_execution_time`.
Keep `method_output_clock` separate: raw schedule, shot-scaled schedule,
effective cost, nominal throughput, or calibrated observed-service seconds.

An outer-train affine, log-affine or Ridge bridge is another learned pipeline,
not a correction that turns the original raw formula into observed execution
time. Historical throughput is not inferred from T1/T2; CLOPS_h is not converted
into CLOPS_v. Missing duration, reliability or throughput support remains
unavailable for that method while the observation stays in the denominator.

## Results and acceptance

The [QPU result page](../results/real_qpu/method_comparison.md) reports the
completed fresh-fit run: 30 CUDA cells, five outer regression folds and
fold-local analytical calibration. Independent saved-evidence QA passed.
The 32 variants share the assigned population, not necessarily identical
successful subsets. Their clocks, coverage and claim boundaries remain explicit.

Final validation checked exact selected IDs and hashes, intact group
splits, recorded train-only transforms, complete required seeds, finite outputs
or explicit terminal reasons, and one held-out prediction per assigned
method/observation. Compare errors on identical successful test IDs, alongside
coverage against 4,515. Report MAE, MedAE, RMSE, R², log1p-MAE, p90/p99/max error,
negative predictions, source-balanced errors and group-resampled paired intervals.

Source slices reuse unified fits; they do not train three separate benchmarks.
These grouped folds do not establish temporal, unseen-backend or unseen-family
transfer. Reconstruction uncertainty and QPack's six structural groups limit
generalization even if prediction error is low.
