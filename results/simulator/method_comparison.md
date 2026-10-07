# Simulator runtime prediction

This table presents all 49 current prediction rows from the [simulator result CSV](method_comparison.csv), grouped by execution context in the saved row order. Scores from different engines, precision settings, measurement boundaries or quality targets are not one common leaderboard. A dash means the reader CSV does not record that metric or it does not apply; it is not zero.

The digital circuit manifest contains 204 members, 191 exact-QASM hashes and 22 families across widths 2–16. The predictor core uses 162 members / 150 distinct hashes at widths 2–9. The 42 members at widths 10–16 are a frontier panel, not added to these predictor scores. All core methods use the frozen exact-hash folds for their context; repeated aliases are not independent circuits. See the [member manifest](../../data/simulator/circuits/sim_common_q16_manifest.csv), [preprocessing guide](../../docs/data_preprocessing.md), and [measurement and training protocol](../../docs/training_and_measurement.md).

For the tables, “assigned / attempted / predicted / scored / unavailable” distinguishes what was scheduled for evaluation, what the method attempted and emitted, what had an observed target for scoring, and what could not be scored. Coverage is scored rows divided by assigned rows. Context target-label counts and both clocks are stated in each section. MAE, p99 and maximum absolute error are in seconds; R² is dimensionless.

## CUDA-Q dense statevector, FP32

Target clock: cudaq_dense_sample_32_warm_execution_seconds. Method output clock: predicted_cudaq_dense_sample_32_warm_execution_seconds. The target is 32-shot warm sample execution, not first-call, build, extraction or end-to-end time. There are 150 observed exact-hash targets and all nine methods predict and score all 150.

<!-- generated: sim_dense_fp32:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Outer-train median baseline (CUDA-Q dense, 32 shots) (outer_train_median) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0017 | −0.0302 | 0.0339 / 0.1035 | Not assessed |
| Linear regression on seven dense-circuit globals (linear_regression) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0022 | 0.3987 | 0.0106 / 0.0814 | Not assessed |
| Ridge regression (alpha=1) on seven dense-circuit globals (ridge_alpha1) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0021 | 0.3937 | 0.0104 / 0.0822 | Not assessed |
| RBF support-vector regression on seven dense-circuit globals (rbf_svr) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0097 | −0.5383 | 0.0268 / 0.0797 | Not assessed |
| Random-forest regression on seven dense-circuit globals (random_forest) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0008 | 0.5159 | 0.0077 / 0.0784 | Not assessed |
| XGBoost regression on seven dense-circuit globals (xgboost) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0010 | 0.5501 | 0.0209 / 0.0705 | Not assessed |
| Ma–Li-style source-DAG graph adaptation for CUDA-Q dense (mali_style_dense_graph) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0017 | 0.1345 | 0.0117 / 0.1025 | Not assessed |
| Matched seven-global-feature MLP baseline for CUDA-Q dense (matched_metadata_mlp) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0016 | 0.1421 | 0.0121 / 0.1025 | Not assessed |
| Work-scaled Ridge diagnostic baseline (not Maestro) (work_scaled_ridge) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0009 | 0.8728 | 0.0124 / 0.0335 | Not assessed |
<!-- generated: sim_dense_fp32:end -->

## CUDA-Q dense statevector, FP64

The target and output clocks, 32 shots and 150-hash population are the same as the FP32 block; precision is a separate execution context and its scores are not pooled with FP32.

<!-- generated: sim_dense_fp64:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Outer-train median baseline (CUDA-Q dense, 32 shots) (outer_train_median) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0018 | −0.0306 | 0.0339 / 0.1027 | Not assessed |
| Linear regression on seven dense-circuit globals (linear_regression) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0022 | 0.4038 | 0.0106 / 0.0805 | Not assessed |
| Ridge regression (alpha=1) on seven dense-circuit globals (ridge_alpha1) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0021 | 0.3988 | 0.0104 / 0.0813 | Not assessed |
| RBF support-vector regression on seven dense-circuit globals (rbf_svr) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0097 | −0.5443 | 0.0264 / 0.0790 | Not assessed |
| Random-forest regression on seven dense-circuit globals (random_forest) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0008 | 0.5191 | 0.0073 / 0.0777 | Not assessed |
| XGBoost regression on seven dense-circuit globals (xgboost) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0010 | 0.5551 | 0.0209 / 0.0696 | Not assessed |
| Ma–Li-style source-DAG graph adaptation for CUDA-Q dense (mali_style_dense_graph) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0017 | 0.1355 | 0.0116 / 0.1017 | Not assessed |
| Matched seven-global-feature MLP baseline for CUDA-Q dense (matched_metadata_mlp) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0016 | 0.1442 | 0.0117 / 0.1017 | Not assessed |
| Work-scaled Ridge diagnostic baseline (not Maestro) (work_scaled_ridge) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.0009 | 0.8649 | 0.0126 / 0.0345 | Not assessed |
<!-- generated: sim_dense_fp64:end -->

## CUDA-Q fixed-bond MPS, FP64, χ=16

Target clock: cudaq_mps_fp64_bond16_warm_state_execution_seconds. Method output clock: predicted_cudaq_mps_fp64_bond16_warm_state_execution_seconds. This warm state-producing get_state target has no shots. All 150 exact hashes have finite runtime labels; fidelity against a dense FP64 reference was assessed on all 150, with threshold 0.99. The quality-qualified column reports MAE only on the 147 pass rows; see the [fixed-target ledger](../../data/simulator/mps/recovered_targets/fixed_chi16_targets.csv).

<!-- generated: sim_fixed_mps:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Outer-training median, fixed CUDA-Q MPS chi=16 (train_fold_median) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 1.8706 | −0.0435 | 43.2778 / 79.8276 | 147 pass / 3 fail; pass-only MAE 1.8672 s |
| Ridge baseline (alpha=1), fixed CUDA-Q MPS chi=16 (ridge_alpha_1) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 1.2757 | 0.4151 | 26.2639 / 67.8137 | 147 pass / 3 fail; pass-only MAE 1.2792 s |
| Ma–Li-style source-DAG graph adaptation, fixed CUDA-Q MPS chi=16 (three-seed median) (graph_median_three_seeds) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.2961 | 0.9507 | 6.3080 / 21.5789 | 147 pass / 3 fail; pass-only MAE 0.2908 s |
| Family-Aware-inspired residual MLP, fixed MPS runtime adaptation (three-seed median) (family_residual_median_three_seeds) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 1.1481 | 0.5034 | 27.7241 / 57.8919 | 147 pass / 3 fail; pass-only MAE 1.1319 s |
| Family-agnostic residual MLP ablation, fixed MPS runtime adaptation (three-seed median) (family_agnostic_median_three_seeds) | baseline / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 1.1214 | 0.5199 | 24.3773 / 61.5300 | 147 pass / 3 fail; pass-only MAE 1.1404 s |
<!-- generated: sim_fixed_mps:end -->

## Noisy Qiskit Aer, FakeSherbrooke, optimization level 1

Target clock: warm_execution. Method output clock: predicted_warm_execution. The context uses the pinned FakeSherbrooke snapshot and noise model, optimization level 1, 1,024 shots and the historical CPU Aer environment; see the [Aer environment](../../data/simulator/aer/measurement/environment.json). Precision was not pinned in the recorded context, so it is not labeled FP32 or FP64. The core supplies 150 finite hash targets; all 18 predictors score all 150. GNN outputs are medians across seeds 42, 1234 and 31415; classical baselines use their frozen seed. The context’s source and feature hashes are in each row of the [result CSV](method_comparison.csv).

<!-- generated: sim_aer:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Linear regression baseline (source-view) (classical_linear_regression_source) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.7803 | −4.4495 | 12.5241 / 55.3281 |
| Linear regression baseline (hybrid-view) (classical_linear_regression_hybrid) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 3.0608 | −138.5770 | 73.3392 / 266.8196 |
| Linear regression baseline (transpiled-view) (classical_linear_regression_transpiled) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4584 | −0.0851 | 9.7827 / 19.8169 |
| Ridge baseline (source-view) (classical_ridge_source) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4717 | 0.1130 | 10.7579 / 16.8680 |
| Ridge baseline (hybrid-view) (classical_ridge_hybrid) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4103 | 0.2053 | 10.4041 / 15.8279 |
| Ridge baseline (transpiled-view) (classical_ridge_transpiled) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3622 | 0.5028 | 3.1616 / 16.5179 |
| RBF-SVR baseline (source-view) (classical_svr_rbf_source) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4205 | 0.0405 | 10.2429 / 17.0547 |
| RBF-SVR baseline (hybrid-view) (classical_svr_rbf_hybrid) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3922 | 0.0970 | 9.3934 / 16.9969 |
| RBF-SVR baseline (transpiled-view) (classical_svr_rbf_transpiled) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3222 | 0.1685 | 8.9913 / 16.6117 |
| Random forest baseline (source-view) (classical_random_forest_source) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3930 | 0.1498 | 9.4910 / 16.5135 |
| Random forest baseline (hybrid-view) (classical_random_forest_hybrid) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3416 | 0.1498 | 9.5049 / 16.6159 |
| Random forest baseline (transpiled-view) (classical_random_forest_transpiled) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3412 | 0.1381 | 9.5115 / 16.7636 |
| XGBoost baseline (source-view) (classical_xgboost_source) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3884 | 0.0989 | 10.9944 / 16.5393 |
| XGBoost baseline (hybrid-view) (classical_xgboost_hybrid) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3915 | 0.0843 | 13.0015 / 14.7473 |
| XGBoost baseline (transpiled-view) (classical_xgboost_transpiled) | common baseline / baseline | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.2970 | 0.2889 | 8.7089 / 15.0628 |
| Azizov-style three-seed GNN median (hybrid-view) (azizov_gnn_hybrid_median_three_seeds) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4064 | 0.0914 | 11.5204 / 16.6572 |
| Azizov-style three-seed GNN median (source-view) (azizov_gnn_source_median_three_seeds) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.4497 | 0.0927 | 11.3164 / 16.8113 |
| Azizov-style three-seed GNN median (transpiled-view) (azizov_gnn_transpiled_median_three_seeds) | adaptation / primary | 150 / 150 / 150 / 150 / 0 | 100.0% | 0.3008 | 0.5206 | 4.5663 / 15.4095 |
<!-- generated: sim_aer:end -->

## Joint CUDA-Q MPS runtime and quality, C44

Target clock: warm_get_state_seconds_per_hash_and_chi. Method output clock: predicted_warm_get_state_seconds_per_hash_and_chi. This primary split has 900 assigned hash-rung rows (150 hashes × χ values 2, 4, 8, 16, 32 and 64); 840 runtime and quality labels are observed, while 60 targets are unavailable. The table reports runtime metrics on the 840 observed rows. The quality threshold is fidelity ≥0.99; the Brier score is dimensionless and uses the observed quality labels. The saved measurements and metrics are in the [joint-MPS ledger](../../data/simulator/mps/joint/analysis/method_metrics.csv).

<!-- generated: sim_joint_c44:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Family-conditioned joint MPS runtime/quality adaptation (C44) (family_conditioned_joint_rung_c44) | adaptation / primary | 900 / 900 / 900 / 840 / 60 | 93.3% | 0.3116 | 0.5218 | 4.1776 / 8.4755 | 719 pass / 121 fail / 60 unassessed; Brier 0.0137 |
| Matched family-agnostic joint MPS ablation (C44) (family_agnostic_ablation_joint_rung_c44) | matched ablation / matched baseline | 900 / 900 / 900 / 840 / 60 | 93.3% | 0.3075 | 0.5237 | 4.3070 / 8.7964 | 719 pass / 121 fail / 60 unassessed; Brier 0.0146 |
<!-- generated: sim_joint_c44:end -->

## Joint CUDA-Q MPS family-component holdout diagnostic

The same 900 rung identities and 840 observed runtime/quality labels are evaluated under a separate family-component holdout split. These values are a diagnostic and do not replace C44.

<!-- generated: sim_joint_family_holdout:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Family-conditioned joint MPS runtime/quality adaptation (family-holdout diagnostic) (family_conditioned_joint_rung_family_holdout) | adaptation / primary | 900 / 900 / 900 / 840 / 60 | 93.3% | 0.3110 | 0.5753 | 3.8660 / 8.3074 | 719 pass / 121 fail / 60 unassessed; Brier 0.0135 |
| Matched family-agnostic joint MPS ablation (family-holdout diagnostic) (family_agnostic_ablation_joint_rung_family_holdout) | matched ablation / matched baseline | 900 / 900 / 900 / 840 / 60 | 93.3% | 0.3088 | 0.4880 | 4.3677 / 8.9614 | 719 pass / 121 fail / 60 unassessed; Brier 0.0136 |
<!-- generated: sim_joint_family_holdout:end -->

The measurement campaign attempted 2,700 rung sessions: 2,520 completed, 72 timed out and 108 failed on the unsupported RCCX adapter. This is a partial campaign; the 60 unavailable rung labels remain in the denominator. In the separate selected-rung diagnostic, 140 of 150 selected configurations have observed quality outcomes: 76 passed, 64 violated the threshold, and 10 are unknown. The 45.7% violation rate means the local quality selector is not dependable. Runtime and quality findings must be read separately.

## QCSim CPU statevector FP64, 1,000 shots

Target clock: maestro_qcsim_process_isolated_reported_execution. Method output clock: predicted_maestro_qcsim_reported_execution_seconds. Each call used a fresh spawned process; this is not persistent warm-process wall time. Each target was reduced from 15 finite engine-reported samples. There are 150 assigned hashes, 143 observed targets and seven unavailable targets; every method predicts and scores 143.

<!-- generated: sim_qcsim:start -->
| Method and ID | Fidelity / role | Assigned / attempted / predicted / scored / unavailable | Coverage | MAE (s) | R² | p99 / max error (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Maestro-style Statevector CPU component-cost adaptation (maestro_style_cpu_sv_component_optimizer_off_adaptation) | adaptation / primary | 150 / 143 / 143 / 143 / 7 | 95.3% | 0.0001 | 0.9490 | 0.0008 / — |
| Outer-training median control (outer_train_median) | common baseline / control | 150 / 143 / 143 / 143 / 7 | 95.3% | 0.0003 | −0.0426 | 0.0045 / — |
| Nested grouped Ridge control (nested_grouped_ridge) | common baseline / control | 150 / 143 / 143 / 143 / 7 | 95.3% | 0.0001 | 0.8824 | 0.0017 / — |
| Source-DAG graph control (source_dag_graph_adaptation_cuda) | common baseline / control | 150 / 143 / 143 / 143 / 7 | 95.3% | 0.0004 | −0.0092 | 0.0042 / — |
<!-- generated: sim_qcsim:end -->

The QCSim reader ledger does not provide maximum absolute error for these rows. In the saved paired comparison, the component-minus-Ridge observed MAE difference is −0.0000299 s, with a 95% interval from −0.0000625 to −0.0000016 s. This is a narrow CPU component-cost result; it does not evaluate Maestro's automatic simulator selector.

## Companion and unavailable routes

These rows retain method-specific evidence but do not have a common digital-panel MAE. They are not ranked with the 49 predictor rows.

<!-- generated: sim_companions:start -->
| Route | Evidence and status | Clock / target | Result |
| --- | --- | --- | --- |
| Maestro original automatic selector / closed Composer route | Original full selector route is not callable from the public repository; the local component adaptation above is a distinct method. | simulator_runtime_target_not_available / not_available | Unavailable (unavailable_original_route_closed_component) |
| cuTensorNet RUNTIME_EST, core q2–q9 | 477 successful paired sessions of 477 attempts | selected_plan_runtime_estimate versus matching warm scalar contraction | Median estimate/warm ratio 3.7340 |
| cuTensorNet RUNTIME_EST, frontier q10–q16 | 126 successful paired sessions of 126 attempts | selected_plan_runtime_estimate versus matching warm scalar contraction | Median estimate/warm ratio 4.1324 |
| Pasqal EMU-MPS analog companion (formula unpromoted) | 18 analog cells; 17 quality pass, 1 fail; formula unpromoted | analog_simulator_wall_clock; candidate fidelity threshold 0.99, reference threshold 0.9999 | No promoted predictor score; not comparable to digital QASM |
| Family-Aware original runtime + approximation-threshold route | Original approximation-threshold target lacks compatible quality labels; local joint runtime/quality adaptation is reported separately. | runtime_plus_approximation_threshold_not_available / not_available | Unavailable (unavailable_original_quality_target_incompatible) |
<!-- generated: sim_companions:end -->

The cuTensorNet ratio ledger contains 603 successful sessions out of 612 attempts in total. Those are repeated same-plan sessions, not 603 unique circuits, and the ratio is not a full simulator wall-clock prediction. The 18 historical engine-runtime distributions and three manifest-authored measurement records remain in the [source CSV](method_comparison.csv) as measurement evidence; they are not prediction scores.

Across these independent contexts, dense Random Forest has the lowest MAE in both recorded precisions, while work-scaled Ridge has the highest R² in both; those are different criteria, and the Ridge intercept is empirical rather than a measured launch overhead. In noisy Aer, the transpiled GNN reaches MAE 0.3008 s while transpiled XGBoost reaches 0.2970 s; the paired interval for their difference includes zero. In joint MPS, family conditioning does not show a resolved runtime advantage over the matched ablation, and selected-rung quality violations remain high. Full paper claims and local limits are discussed in [paper comparison](../../docs/paper_comparison.md).
