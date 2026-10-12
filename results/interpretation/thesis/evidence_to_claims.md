> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Evidence-to-claim registry

Các claim bên dưới tái sử dụng matrix120rows, không chạy thêm experiments. Negative result đã đánh giá có thể SUPPORTED. INCONCLUSIVE chỉ dùng khi stronger comparison/provenance chưa resolve, không phải vì model thất bại. Exact metrics và source selectors ở CSV; path/hash ở manifest.

| ID | Mục | Evidence strength | Câu thesis có thể bảo vệ |
| --- | --- | --- | --- |
| C01 | RQ1 | SUPPORTED WITH QUALIFICATION | Graph giảm pooled MAE 26,7% so với global train median trên cùng 4.515 rows; paired interval âm trong phép so sánh này. |
| C02 | RQ1 | INCONCLUSIVE | Graph có point MAE thấp hơn selector 5,9%; evidence chưa resolve superiority theo paired uncertainty đã lưu. |
| C03 | RQ1/RQ4 | SUPPORTED | Selector có equal-source MAE thấp hơn Graph; equal-group weighting đảo thứ tự Graph/MLP. Thứ hạng phụ thuộc distribution. |
| C04 | RQ1/RQ3 | INCONCLUSIVE | Point MAE nghiêng về Graph, nhưng chưa có resolved advantage so với global MLP; không xác định topology-only gain. |
| C05 | RQ3 | INCONCLUSIVE | Saved controlled topology study không resolve positive DAG increment trong coarse representation đó. Scoring dùng mean predictions của ba seeds, không mean của seed MAEs. |
| C06 | RQ3 | SUPPORTED WITH QUALIFICATION | Model đã fit phụ thuộc edges/wire identities dưới perturbation; không suy ra gain sau retrain. |
| C07 | RQ2/RQ3 | SUPPORTED | 51 raw positions support-refine thành 41 trong simulator corpus và 40 ở mọi parent fit; RQ4 width theo fit population. |
| C08 | RQ2 | INCONCLUSIVE | Không xác minh được tác giả thực thi real tensor 12/40/41; current width 40 được verified độc lập. |
| C09 | RQ2 | SUPPORTED WITH QUALIFICATION | Graph/MLP và nhiều methods là adaptations; pinned native components không làm toàn pipeline trở thành exact reproduction. |
| C10 | RQ2 | SUPPORTED WITH QUALIFICATION | Scheduled one-shot timing khác observed service. Calibration giảm error nhưng không identify physical overhead. |
| C11 | RQ2 | SUPPORTED WITH QUALIFICATION | Calibrated proxy tốt hơn eligible global median, không tốt hơn source-median point estimate. Không đánh giá accuracy original Scholten. |
| C12 | RQ2 | NOT EVALUATED | Original Scholten predictive accuracy không được evaluable trong cohort hiện tại. |
| C13 | RQ2/RQ5 | SUPPORTED WITH QUALIFICATION | Reliability-adjusted resource cost không phải literal service time; vài bridges còn kém matched simple references. |
| C14 | RQ4/RQ5 | SUPPORTED | Family qnn: cả Graph/MLP overpredict mọi row, MAE/MedAE lớn và R² âm nghiêm trọng. Relative Graph win không practical generalization. |
| C15 | RQ4/RQ5 | SUPPORTED | Family random: cả Graph/MLP overpredict mọi row, MAE/MedAE lớn và R² âm nghiêm trọng. Relative Graph win không practical generalization. |
| C16 | RQ4 | SUPPORTED WITH QUALIFICATION | Kyoto là known-circuit backend-context transfer với representation collision, không observation-ID leakage hoặc learned hardware adaptation. |
| C17 | RQ4 | SUPPORTED WITH QUALIFICATION | Full 21-case grid complete, không đồng nghĩa universal generalization hay causal hardware-generation effect. |
| C18 | RQ4 | SUPPORTED WITH QUALIFICATION | Size results là descriptive narrow probes, không general scaling law. |
| C19 | RQ4/RQ5 | NOT EVALUATED | Không có systematic real-QPU unseen-shot test; deterministic multiplication không thay thế evidence. |
| C20 | RQ2/RQ4 | SUPPORTED WITH QUALIFICATION | 32/32 prepared-state PROVED và output match hỗ trợ recipe qualification; 8/32 full-unitary PROVED, 24 counterexamples. Không exact historical recovery. |
| C21 | Track B | SUPPORTED WITH QUALIFICATION | Dense FP32: RF có positive skill và lower MAE trong small-width local context. |
| C22 | Track B | SUPPORTED WITH QUALIFICATION | Dense FP64: RF có positive skill; không cần graph để đạt lowest listed MAE. |
| C23 | Track B | SUPPORTED WITH QUALIFICATION | Fixed-chi MPS: Graph lower MAE hơn Ridge với saved CI âm. |
| C24 | Track B | SUPPORTED WITH QUALIFICATION | Maestro-style local component improves on matched Ridge. |
| C25 | RQ3/Track B | SUPPORTED WITH QUALIFICATION | Compiled representation improves source-GNN locally; transpiled GNN superiority over XGBoost chưa resolve. |
| C26 | RQ3/Track B | INCONCLUSIVE | Không resolved runtime increment từ family conditioning trên local ladder. |
| C27 | RQ5/Track B | SUPPORTED | Cả hai selectors fail fidelity ở 64/140 choices dù 138/140 hashes có passing rung. Runtime accuracy không bảo đảm quality safety. |
| C28 | RQ2/RQ5/Track B | SUPPORTED WITH QUALIFICATION | Native EST overpredicts median warm contraction trong panel này; không whole-API runtime estimate. |
| C29 | RQ5/Track B | SUPPORTED WITH QUALIFICATION | Analog companion ghi quality/time tradeoff, chưa establish resource-formula predictive skill. |
| C30 | RQ5/Track B | SUPPORTED WITH QUALIFICATION | Width holdout có thể đảo lựa chọn model; fitted intercept không physical launch latency. |
| C31 | RQ3 | SUPPORTED WITH QUALIFICATION | Compiled/timing/static blocks có predictive signal; critical-subgraph additions không stable MAE benefit. |
| C32 | RQ5 | NOT EVALUATED | Chưa có calibrated prospective prediction guarantee; report coverage/tails và uncertainty đúng mục đích. |

## C01

Unit: 4515observations/165groups. Target: archived observed service/execution seconds; source boundaries retained. Context: 5outer grouped OOF. Estimators: mali_logical_graph;outer_train_median.

Evidence type: saved OOF paired observational comparison. Strength: SUPPORTED WITH QUALIFICATION.

Graph giảm pooled MAE 26,7% so với global train median trên cùng 4.515 rows; paired interval âm trong phép so sánh này.

Giới hạn: Conditional group bootstrap; heterogeneous source boundaries.

Không được suy ra: Graph universally superior to all baselines.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); [qpu_shared_reference_comparisons_v1.csv](../../analysis/qpu_shared_reference_comparisons.csv).

Exact metrics: `{"ci_high": -0.04187688715915405, "ci_low": -0.7288686538022646, "graph_mae": 0.8341333281956075, "median_mae": 1.1382755517197642, "observed_delta": -0.30414222352415676, "skill": 0.2671956039683391}`.

## C02

Unit: 4515observations/165groups. Target: archived observed service/execution seconds; source boundaries retained. Context: same outer OOF. Estimators: mali_logical_graph;qonductor_budgeted_selector.

Evidence type: saved paired comparison. Strength: INCONCLUSIVE.

Graph có point MAE thấp hơn selector 5,9%; evidence chưa resolve superiority theo paired uncertainty đã lưu.

Giới hạn: CI includes0; pipeline inputs/budgets/refits differ.

Không được suy ra: Graph decisively beats strongest classical model.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); shared_row_comparisons.csv (historical source; not bundled).

Exact metrics: `{"ci_high": 0.45790751386189305, "ci_low": -0.3178575659475499, "observed_delta": -0.052522999942829995, "skill": 0.059237156805844404}`.

## C03

Unit: 3sources; same4515rows. Target: archived observed service/execution seconds; source boundaries retained. Context: equal-source versus pooled. Estimators: mali_logical_graph;qonductor_budgeted_selector;mali_global_mlp.

Evidence type: deterministic reweighting of frozen errors. Strength: SUPPORTED.

Selector có equal-source MAE thấp hơn Graph; equal-group weighting đảo thứ tự Graph/MLP. Thứ hạng phụ thuộc distribution.

Giới hạn: Weighting answers different estimands; source-aware reference uses provenance.

Không được suy ra: One universal winner independent of source/group weights.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); [qpu_source_balanced_skill.csv](../benchmark/qpu_source_balanced_skill.csv).

Exact metrics: `{"graph_equal_group": 0.6332420569952477, "graph_macro": 0.8980293916814103, "graph_macro_skill": -0.021669429587360067, "mlp_equal_group": 0.5806968860844279, "selector_macro": 0.8789823456341584}`.

## C04

Unit: 4515observations/165groups. Target: archived observed service/execution seconds; source boundaries retained. Context: same OOF rows. Estimators: mali_logical_graph;mali_global_mlp.

Evidence type: whole-pipeline comparison. Strength: INCONCLUSIVE.

Point MAE nghiêng về Graph, nhưng chưa có resolved advantage so với global MLP; không xác định topology-only gain.

Giới hạn: Graph gets nodes/T1T2; capacity differs; interval includes0.

Không được suy ra: DAG topology causally improves accuracy.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); shared_row_comparisons.csv (historical source; not bundled).

Exact metrics: `{"ci_high": 0.4845667107853439, "ci_low": -0.329073813615333, "graph_mae": 0.8341333281956075, "graph_minus_mlp_delta": -0.05928965190522192, "mlp_mae": 0.8934229801008293}`.

## C05

Unit: 340observations/170QASM historical coarse256bin. Target: historical log/runtime labels. Context: retrained; per-row arithmetic-mean predictions, then score. Estimators: node-only;true/reverse/shuffledDAG.

Evidence type: retrained topology control. Strength: INCONCLUSIVE.

Saved controlled topology study không resolve positive DAG increment trong coarse representation đó. Scoring dùng mean predictions của ba seeds, không mean của seed MAEs.

Giới hạn: Coarsened representation; not full4515model.

Không được suy ra: Topology irrelevant for every graph model.

Nguồn: REPORT.md (historical source; not bundled); summary.json (historical source; not bundled); evaluate_mali_topology_controls.py (historical source; not bundled).

Exact metrics: `{"bootstrap_replicates": 5000, "bootstrap_seed": 20260921, "delta_ci_high": 0.07615244463573934, "delta_ci_low": -0.016481910107335865, "node_only_mae": 0.814544241537902, "seeds": [1234, 2025, 31415], "true_dag_mae": 0.8448321019144901}`.

## C06

Unit: 340historical observations. Target: historical service/runtime. Context: fixed-model test-time perturbation. Estimators: scratchGraph.

Evidence type: intervention without retraining. Strength: SUPPORTED WITH QUALIFICATION.

Model đã fit phụ thuộc edges/wire identities dưới perturbation; không suy ra gain sau retrain.

Giới hạn: Unnatural inputs; historical preprocessing/single seed.

Không được suy ra: Edge removal proves causal practical topology advantage.

Nguồn: REPORT.md (historical source; not bundled).

Exact metrics: `{"full_mae": 0.4783, "no_edge_mae": 5.9652, "qubit_zero_mae": 4.6862, "reverse_mae": 0.5984}`.

## C07

Unit: 1510namedQASM×2sim contexts; parent4515;257transforms. Target: feature schema, not target metric. Context: static support and fit-only replay. Estimators: Ma–Li Graph/MLP.

Evidence type: source-feature reconstruction and exact transform replay. Strength: SUPPORTED.

51 raw positions support-refine thành 41 trong simulator corpus và 40 ở mọi parent fit; RQ4 width theo fit population.

Giới hạn: Tensor positions may include zeroed constants; support varies by train population.

Không được suy ra: Hard-coded41 or51actual current inputs.

Nguồn: [mali_feature_schema_audit.md](../benchmark/mali_feature_schema_audit.md); [effective_input_widths.csv](../benchmark/effective_input_widths.csv); [global_feature_ledger.csv](../benchmark/global_feature_ledger.csv).

Exact metrics: `{"current_parent": 40, "original_sim_support": 41, "raw": 51, "rq4_final_widths": [11, 22, 40], "transforms": 257}`.

## C08

Unit: published340real rows/upstream notebook. Target: author input provenance. Context: source/notebook comparison. Estimators: author Ma–Li real model.

Evidence type: source provenance audit. Strength: INCONCLUSIVE.

Không xác minh được tác giả thực thi real tensor 12/40/41; current width 40 được verified độc lập.

Giới hạn: Author processed real tensor/executed mask absent.

Không được suy ra: Authors definitely used40, or current model is paper-exact.

Nguồn: [mali_feature_schema_audit.md](../benchmark/mali_feature_schema_audit.md); upstream_pipeline_evidence.json (historical source; not bundled).

Exact metrics: `{"constructor_default": 41, "notebook_example": 12, "recomputed_released_support": 40}`.

## C09

Unit: qualified4515 and per-context simulator. Target: native versus learned output clocks. Context: method-source audit. Estimators: all pipeline families.

Evidence type: implementation/source comparison. Strength: SUPPORTED WITH QUALIFICATION.

Graph/MLP và nhiều methods là adaptations; pinned native components không làm toàn pipeline trở thành exact reproduction.

Giới hạn: Dataset change alone not fidelity failure; implementation/procedure differences matter.

Không được suy ra: Every evaluated row is an original-paper reproduction.

Nguồn: [track_a_fidelity.csv](../benchmark/track_a_fidelity.csv); method_fidelity_v1.csv (historical source; not bundled).

Exact metrics: `{}`.

## C10

Unit: 4515timing rows. Target: archived observed service/execution seconds; source boundaries retained. Context: raw/shot-scaled versus outer-train bridge. Estimators: QCRE;Qiskit.

Evidence type: analytical definition and train-only calibrated comparison. Strength: SUPPORTED WITH QUALIFICATION.

Scheduled one-shot timing khác observed service. Calibration giảm error nhưng không identify physical overhead.

Giới hạn: Nominal context; omitted terms; affine is new model.

Không được suy ra: Raw duration and service time are interchangeable.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); [analytical_controls_v1.csv](../../analysis/analytical_controls.csv).

Exact metrics: `{"qcre_one_shot_mae": 6.115987676979388, "qcre_shot_affine_mae": 1.0512661768661808, "qiskit_one_shot_mae": 6.115983267291188, "qiskit_shot_affine_mae": 1.0496963665009937}`.

## C11

Unit: 3399successful/4515assigned. Target: archived observed service/execution seconds; source boundaries retained. Context: eligible grouped OOF. Estimators: Scholten-inspired proxy/bridges.

Evidence type: generalized proxy plus saved secondary controls. Strength: SUPPORTED WITH QUALIFICATION.

Calibrated proxy tốt hơn eligible global median, không tốt hơn source-median point estimate. Không đánh giá accuracy original Scholten.

Giới hạn: Original eligibility0; nominal throughput/generic depth;1116unavailable.

Không được suy ra: Original D_eff verified or calibrated proxy full-panel winner.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); [analytical_controls_v1.csv](../../analysis/analytical_controls.csv); [runtime_estimator_interpretation.md](../benchmark/runtime_estimator_interpretation.md).

Exact metrics: `{"affine": 0.7483515231538816, "eligible_intercept": 0.8309828265525118, "eligible_median": 0.8200668937428495, "eligible_source_median": 0.692070418476177, "raw": 610.2881368551784}`.

## C12

Unit: 0verified eligible/4515. Target: original kernel/QV runtime. Context: no qualified original evaluation. Estimators: Scholten original.

Evidence type: eligibility determination. Strength: NOT EVALUATED.

Original Scholten predictive accuracy không được evaluable trong cohort hiện tại.

Giới hạn: Missing original kernel/QV ensemble semantics.

Không được suy ra: Proxy failure refutes original paper.

Nguồn: [runtime_estimator_interpretation.md](../benchmark/runtime_estimator_interpretation.md).

Exact metrics: `{}`.

## C13

Unit: 4175finite raw;4367log-cost/4515. Target: archived observed service/execution seconds; source boundaries retained. Context: route-specific grouped OOF. Estimators: Hyb identifiable component/bridges.

Evidence type: cost semantics and matched reference comparison. Strength: SUPPORTED WITH QUALIFICATION.

Reliability-adjusted resource cost không phải literal service time; vài bridges còn kém matched simple references.

Giới hạn: Zero survival/overflow explicit; log route different subset.

Không được suy ra: Reliability-adjusted seconds directly predict observed execution.

Nguồn: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); [analytical_controls_v1.csv](../../analysis/analytical_controls.csv).

Exact metrics: `{"affine_mae": 1.1977015227998407, "log_n": 4367, "raw_mae": 5.361342000783178, "raw_n": 4175}`.

## C14

Unit: 26rows/14groups. Target: archived observed service/execution seconds; source boundaries retained. Context: family holdout with structural embargo. Estimators: mali_logical_graph;mali_global_mlp;CPUselector.

Evidence type: frozen family-OOD prediction/residual comparison. Strength: SUPPORTED.

Family qnn: cả Graph/MLP overpredict mọi row, MAE/MedAE lớn và R² âm nghiêm trọng. Relative Graph win không practical generalization.

Giới hạn: Support diagnostics descriptive, no mechanism-isolating intervention.

Không được suy ra: Discard rare family/outliers or claim useful model because Graph wins.

Nguồn: [generalization_decomposition.csv](../benchmark/generalization_decomposition.csv); [generalization_predictive_skill.csv](../benchmark/generalization_predictive_skill.csv).

Exact metrics: `{"mali_global_mlp": {"mae_seconds": "50.763748119708865", "max_absolute_error_seconds": "63.457831269841144", "medae_seconds": "50.357145674406254", "overprediction_rows": "26", "p99_absolute_error_seconds": "62.84790073363639", "r2_seconds": "-4318.143165236568", "rmse_seconds": "51.28393399336556"}, "mali_logical_graph": {"mae_seconds": "36.27217082455224", "max_absolute_error_seconds": "44.79664886627669", "medae_seconds": "35.84080732601758", "overprediction_rows": "26", "p99_absolute_error_seconds": "44.39711430335579", "r2_seconds": "-2197.2979063132043", "rmse_seconds": "36.586916213605335"}}`.

## C15

Unit: 30rows/16groups. Target: archived observed service/execution seconds; source boundaries retained. Context: family holdout with structural embargo. Estimators: mali_logical_graph;mali_global_mlp;CPUselector.

Evidence type: frozen family-OOD prediction/residual comparison. Strength: SUPPORTED.

Family random: cả Graph/MLP overpredict mọi row, MAE/MedAE lớn và R² âm nghiêm trọng. Relative Graph win không practical generalization.

Giới hạn: Support diagnostics descriptive, no mechanism-isolating intervention.

Không được suy ra: Discard rare family/outliers or claim useful model because Graph wins.

Nguồn: [generalization_decomposition.csv](../benchmark/generalization_decomposition.csv); [generalization_predictive_skill.csv](../benchmark/generalization_predictive_skill.csv).

Exact metrics: `{"mali_global_mlp": {"mae_seconds": "188.8147135376325", "max_absolute_error_seconds": "230.09746326197916", "medae_seconds": "198.6023017551068", "overprediction_rows": "30", "p99_absolute_error_seconds": "229.92003855869584", "r2_seconds": "-22124.32004323568", "rmse_seconds": "190.55134270730107"}, "mali_logical_graph": {"mae_seconds": "145.70825983279852", "max_absolute_error_seconds": "180.80525745143228", "medae_seconds": "148.5215096523125", "overprediction_rows": "30", "p99_absolute_error_seconds": "180.61979685947708", "r2_seconds": "-13158.48988697331", "rmse_seconds": "146.95580967753213"}}`.

## C16

Unit: 148Kyoto rows/117groups. Target: archived observed service/execution seconds; source boundaries retained. Context: backend-context holdout. Estimators: mali_logical_graph;mali_global_mlp.

Evidence type: raw/post-transform equality and saved split audit. Strength: SUPPORTED WITH QUALIFICATION.

Kyoto là known-circuit backend-context transfer với representation collision, không observation-ID leakage hoặc learned hardware adaptation.

Giới hạn: Repeated known structures; no MLP backend/calibration inputs.

Không được suy ra: All148unseen inputs or isolated generation transfer.

Nguồn: [generalization_decomposition.csv](../benchmark/generalization_decomposition.csv); [rq4_report.md](../../generalization/report.md).

Exact metrics: `{"graph_complete_collisions": 0, "mlp_raw_and_transformed_collisions": 148, "observation_overlap": 0}`.

## C17

Unit: 21cases:9family/10backend/2size;126cells. Target: archived observed service/execution seconds; source boundaries retained. Context: frozen case-specific partitions. Estimators: Graph/MLP/CPU.

Evidence type: completed-grid saved evaluation. Strength: SUPPORTED WITH QUALIFICATION.

Full 21-case grid complete, không đồng nghĩa universal generalization hay causal hardware-generation effect.

Giới hạn: Cases overlap; topology/source/shots/clock confounding; historical Twolocal incident preserved.

Không được suy ra: Pool6684asindependent; universal hardware transfer.

Nguồn: [generalization.md](../benchmark/generalization.md); [rq4_report.md](../../generalization/report.md).

Exact metrics: `{"assignments": 6684, "cells": 126, "cpu_cases": 21, "mlp_lower_mae_backend_cases": 9, "neural_cases": 21, "unique_ids": 4515}`.

## C18

Unit: 1321rows/3groups and498rows/2groups. Target: archived observed service/execution seconds; source boundaries retained. Context: QPack size probes. Estimators: mali_logical_graph;mali_global_mlp.

Evidence type: frozen low-support size holdout. Strength: SUPPORTED WITH QUALIFICATION.

Size results là descriptive narrow probes, không general scaling law.

Giới hạn: No CI below10groups; second-case negativeR²/all underprediction.

Không được suy ra: Broad unseen-size extrapolation established.

Nguồn: [generalization_decomposition.csv](../benchmark/generalization_decomposition.csv).

Exact metrics: `{}`.

## C19

Unit: observational shot diversity in4515. Target: observed service. Context: no systematic shot holdout. Estimators: all estimators.

Evidence type: inventory of completed protocols. Strength: NOT EVALUATED.

Không có systematic real-QPU unseen-shot test; deterministic multiplication không thay thế evidence.

Giới hạn: Neural lacks shots; source/workload confounding; simulator shot probe separate.

Không được suy ra: Verified real-QPU shot scaling.

Nguồn: [shot_diversity.csv](../benchmark/shot_diversity.csv); [generalization.md](../benchmark/generalization.md).

Exact metrics: `{}`.

## C20

Unit: 32pairs/11recipes/230linked observations. Target: functional zero-input equivalence. Context: frozen declared layout mappings. Estimators: DD/ZX;measured-output checker.

Evidence type: formal DD prepared-state plus numerical output checks. Strength: SUPPORTED WITH QUALIFICATION.

32/32 prepared-state PROVED và output match hỗ trợ recipe qualification; 8/32 full-unitary PROVED, 24 counterexamples. Không exact historical recovery.

Giới hạn: Assumed mapping, zero input;ZX inconclusive not non-equivalence.

Không được suy ra: 230independentproofs; full algorithmic/historical identity.

Nguồn: results.json (historical source; not bundled).

Exact metrics: `{"measured_match": 32, "prepared_proved": 32, "unitary_counterexample": 24, "unitary_proved": 8}`.

## C21

Unit: 150shared hashes. Target: FP32 warm sample wall-clock. Context: frozen grouped OOF. Estimators: random_forest;outer_train_median.

Evidence type: saved matched-context comparison. Strength: SUPPORTED WITH QUALIFICATION.

Dense FP32: RF có positive skill và lower MAE trong small-width local context.

Giới hạn: FP32 sample only; no width26–30 extrapolation.

Không được suy ra: Universal simulator superiority.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"method_mae": 0.0008213726118033391, "paired_evidence": {"ci_high_seconds": "-0.00032379689341326373", "ci_low_seconds": "-0.0016903512737958524", "context_id": "cudaq_dense_statevector_fp32", "groups": "150", "model": "random_forest", "model_mae_seconds": "0.00082137261180333912", "n": "150", "observed_mae_delta_seconds": "-0.00092764785510748599", "reference": "outer_train_median", "reference_mae_seconds": "0.0017490204669108251", "skill": "0.53038136068580533", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, "r2": 0.5158552845811555, "reference_mae": 0.0017490204669108251, "skill": 0.5303813606858053}`.

## C22

Unit: 150shared hashes. Target: FP64 warm sample wall-clock. Context: frozen grouped OOF. Estimators: random_forest;outer_train_median.

Evidence type: saved matched-context comparison. Strength: SUPPORTED WITH QUALIFICATION.

Dense FP64: RF có positive skill; không cần graph để đạt lowest listed MAE.

Giới hạn: MAE winner không R² winner; local panel only.

Không được suy ra: Universal simulator superiority.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"method_mae": 0.0008129140768325767, "paired_evidence": {"ci_high_seconds": "-0.00033167138966903393", "ci_low_seconds": "-0.0017098262056504387", "context_id": "cudaq_dense_statevector_fp64", "groups": "150", "model": "random_forest", "model_mae_seconds": "0.00081291407683257674", "n": "150", "observed_mae_delta_seconds": "-0.00093998770979528883", "reference": "outer_train_median", "reference_mae_seconds": "0.0017529017866278656", "skill": "0.53624664939362332", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, "r2": 0.519112446326888, "reference_mae": 0.0017529017866278656, "skill": 0.5362466493936233}`.

## C23

Unit: 150shared hashes. Target: FP64 chi16 warm get_state wall-clock. Context: frozen grouped OOF. Estimators: graph_median_three_seeds;ridge_alpha_1.

Evidence type: saved matched-context comparison. Strength: SUPPORTED WITH QUALIFICATION.

Fixed-chi MPS: Graph lower MAE hơn Ridge với saved CI âm.

Giới hạn: 150runtime rows include3quality failures; get_state not sample.

Không được suy ra: Universal simulator superiority.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"method_mae": 0.296091793167725, "paired_evidence": {"ci_high_seconds": "-0.33725969214746648", "ci_low_seconds": "-1.8858226370719373", "context_id": "cudaq_mps_fp64_bond16_warm_state_execution_seconds", "groups": "150", "model": "graph_median_three_seeds", "model_mae_seconds": "0.29609179316772499", "n": "150", "observed_mae_delta_seconds": "-0.9795755771284651", "reference": "ridge_alpha_1", "reference_mae_seconds": "1.27566737029619", "skill": "0.76789263403439012", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, "r2": 0.9507199894516872, "reference_mae": 1.27566737029619, "skill": 0.7678926340343901}`.

## C24

Unit: 143shared hashes. Target: fresh-process engine-reported execution; optimizer off/1000shots. Context: frozen grouped OOF. Estimators: maestro_style_cpu_sv_component_optimizer_off_adaptation;nested_grouped_ridge.

Evidence type: saved matched-context comparison. Strength: SUPPORTED WITH QUALIFICATION.

Maestro-style local component improves on matched Ridge.

Giới hạn: Engine-reported fresh-process time; no closed selection reproduction.

Không được suy ra: Universal simulator superiority.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"method_mae": 0.00010370132150117413, "paired_evidence": {"ci_high_seconds": "-1.5582064153360885e-06", "ci_low_seconds": "-6.3439293246966369e-05", "context_id": "qcsim_statevector_cpu_fp64_1000_shots", "groups": "143", "model": "maestro_style_cpu_sv_component_optimizer_off_adaptation", "model_mae_seconds": "0.00010370132150117413", "n": "143", "observed_mae_delta_seconds": "-2.9905523412076707e-05", "reference": "nested_grouped_ridge", "reference_mae_seconds": "0.00013360684491325084", "skill": "0.22383227020661978", "test_ids_sha256": "bdbe19bc9c1e9d2e8392cafcd3251aae72617d2658859c0b31c1136aaf028124", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, "r2": 0.9490322403411977, "reference_mae": 0.00013360684491325084, "skill": 0.22383227020661978}`.

## C25

Unit: 150shared hashes. Target: warm noisyAer after transpile. Context: FakeSherbrooke Opt1 shots1024. Estimators: source/hybrid/transpiledGNN;XGBoost.

Evidence type: locally retrained representation variants. Strength: SUPPORTED WITH QUALIFICATION.

Compiled representation improves source-GNN locally; transpiled GNN superiority over XGBoost chưa resolve.

Giới hạn: Source→transpiled CI resolves; hybrid→transpiled/GNN→XGB do not.

Không được suy ra: Full paper encoding reproduced or all GNNs beat trees.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"metric_rows": {"azizov_gnn_hybrid_median_three_seeds": 0.40635904268503137, "azizov_gnn_source_median_three_seeds": 0.44965848202068714, "azizov_gnn_transpiled_median_three_seeds": 0.30076412231978206, "classical_xgboost_transpiled": 0.29703532248021886}, "paired_evidence": [{"ci_high_seconds": "0.3061256112875923", "ci_low_seconds": "0.024855211252561404", "context_id": "qiskit_aer_fake_sherbrooke_noisy_opt1", "groups": "150", "model": "azizov_gnn_source_median_three_seeds", "model_mae_seconds": "0.44965848202068714", "n": "150", "observed_mae_delta_seconds": "0.14889435970090509", "reference": "azizov_gnn_transpiled_median_three_seeds", "reference_mae_seconds": "0.30076412231978206", "skill": "-0.49505359400080251", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, {"ci_high_seconds": "0.26642849102399363", "ci_low_seconds": "-0.0058483732520786006", "context_id": "qiskit_aer_fake_sherbrooke_noisy_opt1", "groups": "150", "model": "azizov_gnn_hybrid_median_three_seeds", "model_mae_seconds": "0.40635904268503137", "n": "150", "observed_mae_delta_seconds": "0.10559492036524931", "reference": "azizov_gnn_transpiled_median_three_seeds", "reference_mae_seconds": "0.30076412231978206", "skill": "-0.35108881854258334", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, {"ci_high_seconds": "0.129733479819472", "ci_low_seconds": "-0.17470461891888733", "context_id": "qiskit_aer_fake_sherbrooke_noisy_opt1", "groups": "150", "model": "azizov_gnn_transpiled_median_three_seeds", "model_mae_seconds": "0.30076412231978206", "n": "150", "observed_mae_delta_seconds": "0.0037287998395632016", "reference": "classical_xgboost_transpiled", "reference_mae_seconds": "0.29703532248021886", "skill": "-0.012553388628759876", "test_ids_sha256": "25a3a43db70b434eb1e1e5389bbb4741bb88862549bf363d8a479cf658fab6ae", "uncertainty_status": "pointwise_conditional_group_bootstrap"}]}`.

## C26

Unit: 840rungs/140hashes. Target: warm get_state seconds. Context: grouped hash and family holdout. Estimators: family_conditioned;family_agnostic_ablation.

Evidence type: matched retrained family-conditioning control. Strength: INCONCLUSIVE.

Không resolved runtime increment từ family conditioning trên local ladder.

Giới hạn: Saved runtime delta intervals include0; this does not prove equality.

Không được suy ra: Family context never helps or methods proven equivalent.

Nguồn: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

Exact metrics: `{"paired_evidence": [{"ci_high_seconds": "0.039040952934673745", "ci_low_seconds": "-0.02739220790814265", "context_id": "cudaq_mps_ladder_c44", "groups": "140", "model": "family_conditioned", "model_mae_seconds": "0.31162769053420247", "n": "840", "observed_mae_delta_seconds": "0.0041083008544109956", "reference": "family_agnostic_ablation", "reference_mae_seconds": "0.30751938967979148", "skill": "-0.013359485587848052", "test_ids_sha256": "e7e99a027c053c4ca7b6ff5c1aa868b207e0d21216494fc0aa581e70a64d6714", "uncertainty_status": "pointwise_conditional_group_bootstrap"}, {"ci_high_seconds": "0.037754202160806334", "ci_low_seconds": "-0.034337647032543009", "context_id": "cudaq_mps_ladder_family_holdout", "groups": "140", "model": "family_conditioned", "model_mae_seconds": "0.31101004473227151", "n": "840", "observed_mae_delta_seconds": "0.0022183186184365766", "reference": "family_agnostic_ablation", "reference_mae_seconds": "0.30879172611383493", "skill": "-0.0071838667646775267", "test_ids_sha256": "e7e99a027c053c4ca7b6ff5c1aa868b207e0d21216494fc0aa581e70a64d6714", "uncertainty_status": "pointwise_conditional_group_bootstrap"}]}`.

## C27

Unit: 140assessed/150assignedhashes. Target: selected-state fidelity>=.99. Context: frozen threshold and selection policy. Estimators: both jointMPSselectors.

Evidence type: frozen quality-safety evaluation. Strength: SUPPORTED.

Cả hai selectors fail fidelity ở 64/140 choices dù 138/140 hashes có passing rung. Runtime accuracy không bảo đảm quality safety.

Giới hạn: 10unassessed separate; runtime/Brier/operational endpoints distinct.

Không được suy ra: Hide failed choices or infer safe selection from Brier.

Nguồn: [mps_selection_safety_v1.csv](../../analysis/mps_selection_safety.csv); mps_hash_tradeoff_v1.csv (historical source; not bundled).

Exact metrics: `{"family_agnostic_ablation": {"quality_violation_fraction_observed": "0.45000000000000001", "quality_violations": "63", "selected_observed": "140", "unassessed_selections": "10"}, "family_conditioned": {"quality_violation_fraction_observed": "0.4642857142857143", "quality_violations": "65", "selected_observed": "140", "unassessed_selections": "10"}}`.

## C28

Unit: 603/612planrows;188hashes. Target: same-plan scalar contraction. Context: frozen path/slices/workspace/GPU. Estimators: nativeRUNTIME_EST.

Evidence type: native plan-level calibration diagnostic. Strength: SUPPORTED WITH QUALIFICATION.

Native EST overpredicts median warm contraction trong panel này; không whole-API runtime estimate.

Giới hạn: No matched simple learned reference; opaqueinternals; excludesbuild/search.

Không được suy ra: Internal cost model known or general CUDA-Q timing predictor.

Nguồn: [tn_same_plan_slices_v1.csv](../../analysis/tn_same_plan_slices.csv).

Exact metrics: `{"assigned": 612, "core_ratio": 3.7340122316333324, "frontier_ratio": 4.132415604942254, "success": 603}`.

## C29

Unit: 18analogcells. Target: MPSBackend.run +separatequality. Context: analogTDVPgrid. Estimators: Pasqal companion.

Evidence type: measured timing/quality companion. Strength: SUPPORTED WITH QUALIFICATION.

Analog companion ghi quality/time tradeoff, chưa establish resource-formula predictive skill.

Giới hạn: No identified promoted cost coefficients/predictive evaluation.

Không được suy ra: Identified asymptotic predictor or digital comparison.

Nguồn: [pasqal_stage_quality_v1.csv](../../analysis/pasqal_stage_quality.csv).

Exact metrics: `{"cells": 18, "quality_pass": 17}`.

## C30

Unit: historical208kernelrows +separateRXprobe. Target: kernel/API clocks kept separate. Context: saved grouped/width holdout. Estimators: PyTorchworkRidge/HGB;CUDA-Qmicrobench.

Evidence type: saved historical scaling diagnostics. Strength: SUPPORTED WITH QUALIFICATION.

Width holdout có thể đảo lựa chọn model; fitted intercept không physical launch latency.

Giới hạn: Different kernels; zero-work includes state/API/sampling costs.

Không được suy ra: General T=launch+k2^ngates law verified across simulators.

Nguồn: [runtime_estimator_interpretation.md](../benchmark/runtime_estimator_interpretation.md); README.md (historical source; not bundled); REPORT.md (historical source; not bundled).

Exact metrics: `{"hgb_log_r2": 0.7295, "width_holdout_ridge_log_r2": 0.8754}`.

## C31

Unit: historicalphysical descriptor population. Target: historicalservice/logruntime. Context: retrained grouped/strict/family controls. Estimators: compiled/static/timingblocks.

Evidence type: retrained feature-block controls. Strength: SUPPORTED WITH QUALIFICATION.

Compiled/timing/static blocks có predictive signal; critical-subgraph additions không stable MAE benefit.

Giới hạn: Different population and representation than full4515.

Không được suy ra: Complete native4515feature ladder or per-gate causal values.

Nguồn: REPORT.md (historical source; not bundled); REPORT.md (historical source; not bundled).

Exact metrics: `{}`.

## C32

Unit: currentbenchmark. Target: prospective deployment accuracy. Context: no prospective interval test. Estimators: allestimators.

Evidence type: scope/evidence inventory. Strength: NOT EVALUATED.

Chưa có calibrated prospective prediction guarantee; report coverage/tails và uncertainty đúng mục đích.

Giới hạn: Conditional bootstrap estimates comparison error, not deployment coverage.

Không được suy ra: 95%pairedCI means95%futurejobs covered.

Nguồn: [benchmark_question_matrix.csv](../benchmark/benchmark_question_matrix.csv).

Exact metrics: `{}`.
