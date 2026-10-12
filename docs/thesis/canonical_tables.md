> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# Canonical thesis tables

CSV giữ precision từ saved evidence và source path/hash theo từng row. Markdown làm tròn để đọc, không tạo metric mới. Không đọc dọc Table D để tìm universal simulator winner.

## Table A — Method taxonomy

[CSV](../../results/thesis/tables/a_method_taxonomy.csv).

| method | track | original_target | benchmark_target | inputs | method_category | fidelity_classification | major_caveat |
|---|---|---|---|---|---|---|---|
| Ma–Li-style Graph + globals | A | observed runtime | archived observed service/execution seconds; source boundaries retained | DAG, raw178 node/calibration; raw51 globals, parent40; no shots/backend ID | TransformerConv DAG + global MLP | Architecture/core adaptation | Fit-only per-feature moments, nominal wire-index T1/T2, qualified recipes, grouped splits; no active acquisition; parent retains inner1–3 fit. Exact reproduction; topology-only gain; recovered job-day placement |
| Matched global-feature MLP | A | observed runtime | archived observed service/execution seconds; source boundaries retained | raw51 / parent40; no graph/calibration/shots/backend | TransformerConv DAG + global MLP | Architecture/core adaptation | No author processed real tensor; input support varies by fit. Identical-information topology ablation; hardware adaptation from collisions |
| Qonductor-style six regressors + inner selector | A | observed job time | archived observed service/execution seconds; source boundaries retained | swap counts CX/CZ/ECR; operand-stack depth; active width; shots; circuit_count=1 | Six regressor families and polynomial features | Architecture/core adaptation | Pinned extraction/constructors; bounded rather than full upstream search; qualified nominal compiled QPack view; mitigation not reproduced. Literal SWAP; full search/mitigation reproduction; source-local fits masquerading as unified |
| QCRE/Tremba pinned timing component | A | unitary critical-path seconds | archived observed service/execution seconds; source boundaries retained | BQSKit unitary view and nominal native duration map | Per-wire max arrival time + location-dependent native gate duration | Original/native estimator | Native component only; terminal measurement/barriers omitted; historical routing unknown. Full QCRE compilation/paper reproduction; raw service-time predictor |
| Qiskit scheduled-duration reference | A | scheduled single-shot seconds | archived observed service/execution seconds; source boundaries retained | Frozen circuit and nominal Target | Instruction durations and circuit scheduling | Original/native estimator | Backend-matched snapshot not verified job-day; possible measurement/reset semantics differ from QCRE. Scheduled duration equals observed service or QCRE unitary clock |
| Deterministic shot multiplication | A | S × scheduled one-shot seconds | archived observed service/execution seconds; source boundaries retained | Native timing and shots | Multiply duration by S | Generalized proxy | No measured inter-shot/readout/host/batch cost decomposition. Verified shot extrapolation or provider runtime identity |
| Scholten-inspired generalized proxy | A | M*S*D_eff/C original; generic-depth nominal seconds locally | archived observed service/execution seconds; source boundaries retained | Compiled wire depth + admitted nominal throughput | Shots × circuit count × QV-normalized effective layers / CLOPS | Generalized proxy | Generic wire depth is not ensemble/QV-normalized D_eff; 1116 unavailable; no CLOPS_h/v conversion. Scholten original accuracy; manufactured CLOPS; full-panel ranking |
| Scholten original (not evaluable) | A | M*S*D_eff/C | archived observed service/execution seconds; source boundaries retained | Missing kernel/template and QV ensemble expectations with compatible throughput | Shots × circuit count × QV-normalized effective layers / CLOPS | UNVERIFIED / not evaluated | 0/4515 verified eligibility; not executed as original. Use proxy scores as original-paper reproduction |
| Hyb-HANAS identifiable cost component | A | useful-work/resource effective seconds | archived observed service/execution seconds; source boundaries retained | Native timing/errors/T2/routing and survival/log-cost | Gate/routing cost divided by survival; full paper scales gradient/steps | Architecture/core adaptation | No full NAS/gradient/step model; finite4175; log-cost4367; undefined values explicit. Literal execution-time identity; full hybrid NAS reproduction |
| Outer-train analytical-to-service bridge | A/B | native timing/cost/throughput | archived observed service/execution seconds; source boundaries retained | Raw or shot-scaled predictor plus eligible outer-train labels | Affine/log-affine; separately named log-cost Ridge | Calibrated adaptation | New fitted predictor; coverage depends on route. Physical launch/host overhead identified; native estimator fidelity restored |
| Train-only constant references | A/B | observed service/runtime reference | archived observed service/execution seconds; source boundaries retained | Outer-train target labels; source identity only for source median | Mean/global median/source median; fallback rules explicit | Benchmark-defined diagnostic | Retrospective secondary controls; source-provenance information advantage. Preregistered primary models or equal-information source reference |
| Context-specific Graph/global controls | B | simulator observed execution time | context-specific warm sample/get_state or engine-reported time | Local DAG/node/globals per frozen context | DAG/global learned predictors | Architecture/core adaptation | Different engines, schemas and fit policies; not exact upstream model run. Universal simulator winner; isolated topology benefit |
| Azizov-described source/hybrid/transpiled GNN | B | noisy Aer after-transpile execution | warm Aer execution seconds after transpilation | Source/compiled graph and globals; FakeSherbrooke Opt1 shots1024 | Source/hybrid/transpiled two-branch GNN + five regressors | Architecture/core adaptation | Paper full encoding/code unavailable; one backend/level and local hardware/software. Exact authors encoding or full two-backend/four-level reproduction |
| Context-specific classical/complexity controls | B | context-specific measured runtime | same context as each neural comparator | Local source/compiled counters or 2^n×gate work features | Median, six tabular/linear models, work-scaled Ridge | Benchmark-defined diagnostic | Different feature views, budgets and clocks; not a single pooled method. Physical launch coefficient or universal scaling law |
| Family-aware joint MPS adaptation | B | joint runtime/approximation target | warm get_state seconds plus quality probability | Family context + local six-chi ladder, fidelity>=0.99 | Family classifier + FiLM/residual + runtime/threshold branches | Architecture/core adaptation | Quantum Rings threshold replaced by local bond/fidelity policy; 840rungs/140hashes. Original engine reproduction; Brier improvement proves safe selection |
| Fixed-MPS family residual control | B | family-conditioned runtime | warm get_state at FP64 chi16 | Fixed-state context family/global inputs | Family classifier + FiLM/residual + runtime/threshold branches | Architecture/core adaptation | Runtime-only fixed-bond control, not full joint approximation model. Joint model reproduction or quality guarantee |
| Maestro-style local component | B | calibrated runtime supporting engine choice | QCSim CPU FP64 engine-reported seconds; fresh processes | Width/gate/sampling calibration, shots1000 | Hardware-calibrated operation complexity sum + sampling term | Architecture/core adaptation | Optimizer off, one engine; closed Composer selection not reproduced; 143/150 hashes. Persistent warm wall-clock or full automatic selection reproduction |
| cuTensorNet native RUNTIME_EST | B | all-slice selected-plan contraction seconds | same-plan warm scalar contraction | Network/path/slices/workspace/GPU/complex64 | Native architecture-aware RUNTIME_EST | Original/native estimator | Scalar not full state/sample; internals opaque; build/search/first excluded from comparison. Internal formula recovered; predicts whole simulator API |
| Pasqal analog MPS/TDVP companion | B | analog MPS/TDVP runtime resource model | host MPSBackend.run, excluding quality capture | Atoms, pulse program, chi and TDVP settings | Work terms N²χ³ and N³χ² with fitted coefficients | Benchmark-defined diagnostic | 18cells; coefficients not identifiable/promoted; no validated predictor. Digital leaderboard or identified extrapolation coefficients |

Category là component/implementation, không mặc định toàn paper. Faithful reproduction là taxonomy chuẩn nhưng không gán nhãn đó cho Graph/MLP hiện tại.

## Table B — Track A main accuracy

[CSV](../../results/thesis/tables/b_qpu_main_accuracy.csv).

| method | n | mae_s | r2 | source_balanced_mae_s | equal_group_mae_s | p99_absolute_error_s |
|---|---|---|---|---|---|---|
| mali_logical_graph | 4515 | 0.834133 | 0.468091 | 0.898029 | 0.633242 | 2.2242 |
| mali_global_mlp | 4515 | 0.893423 | 0.363158 | 0.976019 | 0.580697 | 2.5975 |
| qonductor_budgeted_selector | 4515 | 0.886656 | 0.411307 | 0.878982 | 0.6693 | 2.68987 |
| qonductor_extra_trees | 4515 | 0.966441 | 0.361942 | 1.09499 | 0.768643 | 2.72374 |
| qonductor_random_forest | 4515 | 1.02738 | 0.102276 | 1.01229 | 0.711199 | 3.56177 |
| qonductor_gradient_boosting | 4515 | 1.01727 | 0.0704359 | 1.00828 | 0.681543 | 3.71879 |
| qonductor_adaboost | 4515 | 1.03363 | 0.269806 | 1.08096 | 0.908914 | 2.82636 |
| qonductor_histogram_gradient_boosting | 4515 | 0.97239 | 0.32402 | 0.906698 | 0.669682 | 2.74615 |
| qonductor_polynomial_regression | 4515 | 0.917725 | 0.402409 | 1.00791 | 0.634507 | 2.47737 |
| outer_train_median | 4515 | 1.13828 | -0.143558 | 2.17064 | 2.30387 | 4.65431 |
| outer_train_source_median | 4515 | 0.974362 | 0.181923 | 1.22965 | 1.47863 | 3.49694 |

Source median nhận source identity nên không equal-information reference. Native analytical variants có eligible denominators khác; không trộn vào bảng full4515 này. Raw/calibrated details: [controls](../../results/analysis/analytical_controls.csv).

## Table C — Track A generalization

[CSV](../../results/thesis/tables/c_qpu_generalization.csv).

| case_id | method_id | n | structural_groups | mae_seconds | r2_seconds | exact_consumed_input_collision_rows |
|---|---|---|---|---|---|---|
| family:qft | mali_logical_graph | 46 | 26 | 0.562994 | 0.531692 | 0 |
| family:qft | mali_global_mlp | 46 | 26 | 0.512961 | 0.572636 | 0 |
| family:qnn | mali_logical_graph | 26 | 14 | 36.2722 | -2197.3 | 0 |
| family:qnn | mali_global_mlp | 26 | 14 | 50.7637 | -4318.14 | 0 |
| family:random | mali_logical_graph | 30 | 16 | 145.708 | -13158.5 | 0 |
| family:random | mali_global_mlp | 30 | 16 | 188.815 | -22124.3 | 0 |
| backend_context:ibm_kyoto | mali_logical_graph | 148 | 117 | 0.664048 | 0.795814 | 0 |
| backend_context:ibm_kyoto | mali_global_mlp | 148 | 117 | 0.541317 | 0.853584 | 148 |
| backend_context:ibm_osaka | mali_logical_graph | 192 | 148 | 0.796944 | 0.733889 | 0 |
| backend_context:ibm_osaka | mali_global_mlp | 192 | 148 | 0.430253 | 0.914454 | 149 |
| backend_context:ibm_quito | mali_logical_graph | 1051 | 7 | 1.95833 | -6.12485 | 0 |
| backend_context:ibm_quito | mali_global_mlp | 1051 | 7 | 1.51674 | -3.55937 | 1051 |
| qpack_size:train_through_4 | mali_logical_graph | 1321 | 3 | 0.727175 | -0.0724722 | 0 |
| qpack_size:train_through_4 | mali_global_mlp | 1321 | 3 | 0.641164 | 0.0627129 | 0 |
| qpack_size:train_through_5 | mali_logical_graph | 498 | 2 | 0.474422 | -4.68543 | 0 |
| qpack_size:train_through_5 | mali_global_mlp | 498 | 2 | 0.788557 | -13.0604 | 0 |

Markdown chọn các case minh họa theo mục tiêu synthesis; CSV giữ đủ 42 neural method rows/21 cases. Đây không phải chọn lại seed/model. CPU selectors và train-only references trong CSV; source-weighted parent sensitivity ở Table B.

## Table D — Track B by context

[CSV](../../results/thesis/tables/d_simulator_contexts.csv).

| context | method | evaluated_n | mae_s | r2 | runtime_boundary | reported_quantity | diagnostic_value |
|---|---|---|---|---|---|---|---|
| cudaq_dense_statevector_fp32 | random_forest | 150 | 0.000821373 | 0.515855 | FP32 warm sample wall-clock | OOF accuracy |  |
| cudaq_dense_statevector_fp64 | random_forest | 150 | 0.000812914 | 0.519112 | FP64 warm sample wall-clock | OOF accuracy |  |
| cudaq_dense_statevector_fp32 | mali_style_dense_graph | 150 | 0.00171594 | 0.134538 | FP32 warm sample wall-clock | OOF accuracy |  |
| cudaq_dense_statevector_fp64 | mali_style_dense_graph | 150 | 0.00173022 | 0.135541 | FP64 warm sample wall-clock | OOF accuracy |  |
| cudaq_dense_statevector_fp32 | matched_metadata_mlp | 150 | 0.00157772 | 0.142063 | FP32 warm sample wall-clock | OOF accuracy |  |
| cudaq_dense_statevector_fp64 | matched_metadata_mlp | 150 | 0.00159008 | 0.144241 | FP64 warm sample wall-clock | OOF accuracy |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | ridge_alpha_1 | 150 | 1.27567 | 0.415102 | FP64 chi16 warm get_state wall-clock | OOF accuracy |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | graph_median_three_seeds | 150 | 0.296092 | 0.95072 | FP64 chi16 warm get_state wall-clock | OOF accuracy |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | azizov_gnn_source_median_three_seeds | 150 | 0.449658 | 0.0926793 | warm Aer execution after transpilation; Opt1/1024 shots | OOF accuracy |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | azizov_gnn_transpiled_median_three_seeds | 150 | 0.300764 | 0.520606 | warm Aer execution after transpilation; Opt1/1024 shots | OOF accuracy |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | classical_xgboost_transpiled | 150 | 0.297035 | 0.288858 | warm Aer execution after transpilation; Opt1/1024 shots | OOF accuracy |  |
| qcsim_statevector_cpu_fp64_1000_shots | maestro_style_cpu_sv_component_optimizer_off_adaptation | 143 | 0.000103701 | 0.949032 | fresh-process engine-reported execution; optimizer off/1000shots | OOF accuracy |  |
| qcsim_statevector_cpu_fp64_1000_shots | nested_grouped_ridge | 143 | 0.000133607 | 0.882386 | fresh-process engine-reported execution; optimizer off/1000shots | OOF accuracy |  |
| cudaq_mps_ladder_c44 | family_agnostic_ablation | 840 | 0.307519 | 0.523697 | warm get_state; six chi rungs; grouped hash folds | OOF accuracy |  |
| cudaq_mps_ladder_c44 | family_conditioned | 840 | 0.311628 | 0.521823 | warm get_state; six chi rungs; grouped hash folds | OOF accuracy |  |
| cudaq_mps_ladder_family_holdout | family_agnostic_ablation | 840 | 0.308792 | 0.487994 | warm get_state; six chi rungs; family holdout | OOF accuracy |  |
| cudaq_mps_ladder_family_holdout | family_conditioned | 840 | 0.31101 | 0.575311 | warm get_state; six chi rungs; family holdout | OOF accuracy |  |
| cuTensorNet core_q2_q9 | native RUNTIME_EST | 477 | NOT_COMPARABLE | NOT_REPORTED | same-plan warm scalar contraction; all-slice estimate | median EST/actual ratio | 3.73401 |
| cuTensorNet frontier_q10_q16 | native RUNTIME_EST | 126 | NOT_COMPARABLE | NOT_REPORTED | same-plan warm scalar contraction; all-slice estimate | median EST/actual ratio | 4.13242 |
| Pasqal analog MPS/TDVP | local companion grid | 18 | NOT_A_PREDICTIVE_SCORE | NOT_A_PREDICTIVE_SCORE | network/first/warm separately | quality-pass count | 17 |

## Table E — Accuracy and warm full-B2 cost

[CSV](../../results/thesis/tables/e_accuracy_cost.csv).

| context | method | n | accuracy_unit | mae | rmse | r2 | median_B2_ms |
|---|---|---|---|---|---|---|---|
| QPU exact45 | graph | 45 | s | 0.724177 | 0.94754 | 0.897687 | 332.557 |
| QPU exact45 | mlp | 45 | s | 0.812521 | 1.25757 | 0.81978 | 149.514 |
| Dense FP32 exact30 | graph | 30 | ms | 1.75343 | 3.59679 | 0.678366 | 7.25009 |
| Dense FP32 exact30 | mlp | 30 | ms | 1.52333 | 3.3867 | 0.714842 | 3.68971 |
| Dense FP64 exact30 | graph | 30 | ms | 0.807979 | 1.62501 | 0.925081 | 7.0761 |
| Dense FP64 exact30 | mlp | 30 | ms | 0.618299 | 1.14641 | 0.962713 | 3.61045 |

MAE/RMSE và B2 là hai đại lượng khác nhau, không lấy hiệu. Paired CI và preparation fraction giữ trong CSV. Ratio là ratio of marginal medians, không median paired ratio.

## Table F — Failure modes

[CSV](../../results/thesis/tables/f_failure_modes.csv).

| failure_mode | evidence | affected_context | scientific_implication |
|---|---|---|---|
| Family-OOD overprediction | QNN26/Random30: cả Graph/MLP overpredict mọi test row; negative R² | QPU held-out family | Relative winner không đồng nghĩa practically useful prediction |
| Representation collision | Kyoto148/148; Osaka149/192; Quito1051/1051 MLP fit-vector matches, no observation-ID leakage | QPU backend-context | Global-only context blind; không claim learned hardware adaptation |
| Ranking sensitivity | Pooled Graph<MLP; equal-group MLP<Graph; source-balanced selector<Graph | QPU shared4515 | Estimand/source composition là phần của kết luận, không lựa chọn metric đẹp |
| Tail sensitivity | p99/max error trong Table B; large negative R² ở QNN/Random | QPU OOF/holdouts | MAE đơn lẻ không mô tả tail risk |
| Subset-selection sensitivity | Dense exact30 R² cao hơn full150 neural R² khoảng0.13–0.14 | Dense FP32/FP64 | Không biến timed-subset accuracy thành full-context performance |
| Metric dependence | Dense train-median MedAE tốt hơn neural, MAE/RMSE/R² kém hơn | Dense exact30 | Không một metric/winner hợp với mọi mục đích |
| Unsafe quality selection | 64/140 choices vi phạm gate ở mỗi frozen c44 selector;10/150 chưa assessed | MPS ladder c44 | Runtime/classifier fit không đủ cho safe approximation choice |
| Native cost-model bias | Same-plan EST/actual median core3.734/frontier4.132 | cuTensorNet | Cost estimate cần đánh giá đúng plan/clock; không suy ra opaque formula |
| Preprocessing incompatibility | 38/45 QPU graph-order mismatch trước recovery;210/210 model-ready exact sau recovery | AM1 frozen input replay | Graph-semantic agreement không thay được exact consumed-input identity |
| Analog quality failure | 17/18 cells pass; chi16 quench cell fails | Pasqal companion | Chạy hoàn tất không bảo đảm đủ quality; chưa validated fitted predictor |

## Table G — RQ closure

[CSV](../../results/thesis/tables/g_rq_closure.csv).

| rq | track | status | blocking | explanation |
|---|---|---|---|---|
| RQ1 | A | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Shared 4.515-row accuracy và competitive/analytical controls đủ bảo vệ pipeline comparison; point wins còn conditional uncertainty. |
| RQ1 | B | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Matched context-specific accuracy đủ; seven predictive contexts/native companions giữ clock và method applicability riêng. |
| RQ2 | A | PARTIALLY ANSWERED | NON-BLOCKING | 21/21 cases và severe negative family findings đã evaluated. Source/shot và isolated generation effects chưa tested; low-support size/backend cases descriptive. |
| RQ2 | B | PARTIALLY ANSWERED | NON-BLOCKING | Family MPS và historical width findings có evidence; systematic current-engine/hardware/precision-transfer chưa measured. |
| RQ3 | A | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Tails, source-weighted rank changes, QNN/Random và collisions đã established. Causal feature attribution/prospective safety chưa established. |
| RQ3 | B | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Quality violations, subset-selection sensitivity, metric dependence/TN bias và resource/support failures đã established trong từng context. |
| RQ4 | A | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | ANSWERED WITHIN MEASURED SCOPE: matched warm B2 + accuracy cho Graph/MLP exact45; stronger classical/cold/scaling utility không thuộc bounded conclusion. |
| RQ4 | B | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | ANSWERED WITHIN MEASURED SCOPE: exact30 per Dense precision cho paired accuracy/cost; MLP ưu thế có conditional CIs. Các engine/cold/complete Pareto frontier khác còn ngoài đo. |
| RQ5 | A | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Fidelity/input/clock applicability được chốt, provenance ambiguities đã resolved; historical author tensor/mappings và original Scholten giữ limits. |
| RQ5 | B | ANSWERED WITH QUALIFICATIONS | NON-BLOCKING | Context/API/quality/native-versus-adaptation rules và practical conditional selection đủ; không production-safe automatic selection hoặc universal engine portability. |

## Table H — Limitations/future work

CSV (historical source; not bundled).

| id | rq | track | gap | blocking | impact |
|---|---|---|---|---|---|
| G1 | RQ2 | A | Unseen source/shot và isolated generation/topology transfer | NON-BLOCKING | Không claim portability qua nguồn/throughput/hardware generation độc lập. |
| G2 | RQ2/RQ4 | B | Systematic current-engine width/hardware/precision portability | NON-BLOCKING | Core small-width/context scores không quy thành universal scaling. |
| G3 | RQ4 | A/B | Strong classical fitted-state overhead/complete frontier | NON-BLOCKING | Không chọn deployment-optimal estimator trên toàn method set. |
| G4 | RQ4 | A/B | Cold start, deployment, operational break-even/scheduler utility | NON-BLOCKING | Warm B2 không include full lifecycle/system gains. |
| G5 | RQ4 | B | MPS/Maestro/cuTensorNet full-B2 under equivalent selection task | NON-BLOCKING | Không full multi-engine cost frontier; plan/build latency không isolated internal EST inference. |
| G6 | RQ5 | A | Author Ma–Li real tensor, original logical/layout/history and Scholten eligibility | NON-BLOCKING | Chỉ adaptations/native components/proxy claims; không faithful original paper inference. |
| G7 | RQ3/RQ5 | B | Prospective quality-safe MPS selection and Pasqal predictor identification | NON-BLOCKING | Observed safety failure là finding; analog companion không claimed predictor. |
| G8 | RQ3 | A/B | Causal feature/topology attribution and prospective intervals | NON-BLOCKING | Descriptive support shifts/tails không causal mechanism/future safety guarantee. |
| G9 | Delivery | A/B | Public rights/metadata, exact export and clean-release reproduction | NON-BLOCKING FOR SCIENCE; RELEASE GATE | Scientific sufficiency không public-release approval; reduced QPU materials không self-contained source-refit. |
