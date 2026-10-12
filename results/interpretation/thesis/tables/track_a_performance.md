> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../../docs/thesis/README.md).

# Bảng 2. Track A trên cohort 4.515

Giữ toàn bộ 34 dòng đã lưu, kể cả cost/shot-scaled variants có sai số lớn. Đây không phải leaderboard ngang hàng giữa native outputs và calibrated service estimators. Coverage=n_success/4515. MAE=(Σ|prediction−target|)/n_success; RMSE=sqrt(mean residual²); R²=1−SSE/SST trên chính rows đó. Skill=1−MAE_method/MAE_reference chỉ trên cùng rows. Graph 26.7195604% so với global median và 5.9237157% so với inner-selected regressor đã đối chiếu độc lập. Source-balanced là mean của ba source MAE, không mean R². Các controls eligible-specific ở bảng analytical riêng.

| Variant | Output clock | n thành công | Coverage /4515 | MAE (s) | MedAE (s) | RMSE (s) | R² | Equal-source MAE (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hyb_nominal_r2_effective_cost_v1__affine | calibrated_observed_service_seconds | 4175 | 0.924695 | 1.1977 | 0.971359 | 1.42838 | -0.289082 | 1.86326 |
| hyb_nominal_r2_effective_cost_v1__log_affine | calibrated_observed_service_seconds | 4175 | 0.924695 | 1.16836 | 1.12202 | 1.33018 | -0.117931 | 1.77801 |
| hyb_nominal_r2_effective_cost_v1__raw | one_circuit_effective_cost_seconds | 4175 | 0.924695 | 5.36134 | 5.72155 | 5.61882 | -18.9471 | 4.49147 |
| hyb_nominal_r2_log_cost_ridge_v1 | calibrated_observed_service_seconds | 4367 | 0.96722 | 0.986425 | 0.989315 | 1.17151 | 0.286652 | 1.34288 |
| hyb_nominal_r2_shot_effective_cost_v1__affine | calibrated_observed_service_seconds | 4175 | 0.924695 | 1.1976 | 0.971361 | 1.42819 | -0.288741 | 1.86319 |
| hyb_nominal_r2_shot_effective_cost_v1__log_affine | calibrated_observed_service_seconds | 4175 | 0.924695 | 1.0491 | 0.952672 | 1.21142 | 0.0727893 | 1.65004 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | shot_linear_effective_cost_seconds | 4175 | 0.924695 | 2600.51 | 6.00854 | 7888.06 | -3.93125e+07 | 1377.25 |
| mali_global_mlp | predicted_observed_service_seconds | 4515 | 1 | 0.893423 | 0.649846 | 1.17911 | 0.363158 | 0.976019 |
| mali_logical_graph | predicted_observed_service_seconds | 4515 | 1 | 0.834133 | 0.636842 | 1.0776 | 0.468091 | 0.898029 |
| outer_train_mean | observed_service_reference_seconds | 4515 | 1 | 1.16156 | 0.941528 | 1.50348 | -0.035409 | 2.16433 |
| outer_train_median | observed_service_reference_seconds | 4515 | 1 | 1.13828 | 0.757269 | 1.58004 | -0.143558 | 2.17064 |
| outer_train_source_median | observed_service_reference_seconds | 4515 | 1 | 0.974362 | 0.703012 | 1.3364 | 0.181923 | 1.22965 |
| qcre_source_estimator__affine | calibrated_observed_service_seconds | 4515 | 1 | 1.06129 | 0.942336 | 1.28286 | 0.246164 | 1.54015 |
| qcre_source_estimator__log_affine | calibrated_observed_service_seconds | 4515 | 1 | 1.08748 | 0.962133 | 1.28637 | 0.242027 | 1.50273 |
| qcre_source_estimator__raw | scheduled_unitary_critical_path_seconds | 4515 | 1 | 6.11599 | 6.60081 | 6.29177 | -17.1328 | 6.01031 |
| qcre_source_estimator_shot_scaled__affine | calibrated_observed_service_seconds | 4515 | 1 | 1.05127 | 0.937578 | 1.2658 | 0.266082 | 1.52417 |
| qcre_source_estimator_shot_scaled__log_affine | calibrated_observed_service_seconds | 4515 | 1 | 1.07582 | 1.01204 | 1.2656 | 0.266316 | 1.52395 |
| qcre_source_estimator_shot_scaled__raw | shot_scaled_unitary_seconds | 4515 | 1 | 5.67186 | 6.20247 | 5.83092 | -14.5738 | 4.3601 |
| qiskit_estimate_duration_snapshot__affine | calibrated_observed_service_seconds | 4515 | 1 | 1.06099 | 0.942585 | 1.28253 | 0.246555 | 1.53993 |
| qiskit_estimate_duration_snapshot__log_affine | calibrated_observed_service_seconds | 4515 | 1 | 1.08726 | 0.96215 | 1.2861 | 0.242349 | 1.5025 |
| qiskit_estimate_duration_snapshot__raw | scheduled_single_shot_seconds | 4515 | 1 | 6.11598 | 6.6008 | 6.29177 | -17.1328 | 6.0103 |
| qiskit_shot_scaled_schedule_seconds__affine | calibrated_observed_service_seconds | 4515 | 1 | 1.0497 | 0.93687 | 1.26365 | 0.268569 | 1.52156 |
| qiskit_shot_scaled_schedule_seconds__log_affine | calibrated_observed_service_seconds | 4515 | 1 | 1.07277 | 1.00781 | 1.26117 | 0.271434 | 1.51828 |
| qiskit_shot_scaled_schedule_seconds__raw | shot_scaled_scheduled_seconds | 4515 | 1 | 5.65423 | 6.18055 | 5.81265 | -14.4763 | 4.34743 |
| qonductor_adaboost | predicted_observed_service_seconds | 4515 | 1 | 1.03363 | 0.915434 | 1.26258 | 0.269806 | 1.08096 |
| qonductor_budgeted_selector | predicted_observed_service_seconds | 4515 | 1 | 0.886656 | 0.766674 | 1.13366 | 0.411307 | 0.878982 |
| qonductor_extra_trees | predicted_observed_service_seconds | 4515 | 1 | 0.966441 | 0.81675 | 1.18024 | 0.361942 | 1.09499 |
| qonductor_gradient_boosting | predicted_observed_service_seconds | 4515 | 1 | 1.01727 | 0.608255 | 1.42456 | 0.0704359 | 1.00828 |
| qonductor_histogram_gradient_boosting | predicted_observed_service_seconds | 4515 | 1 | 0.97239 | 0.826586 | 1.21481 | 0.32402 | 0.906698 |
| qonductor_polynomial_regression | predicted_observed_service_seconds | 4515 | 1 | 0.917725 | 0.840773 | 1.1422 | 0.402409 | 1.00791 |
| qonductor_random_forest | predicted_observed_service_seconds | 4515 | 1 | 1.02738 | 0.737862 | 1.39995 | 0.102276 | 1.01229 |
| scholten_unified_nominal_throughput__affine | calibrated_observed_service_seconds | 3399 | 0.752824 | 0.748352 | 0.536737 | 1.03219 | 0.351767 | 1.4061 |
| scholten_unified_nominal_throughput__log_affine | calibrated_observed_service_seconds | 3399 | 0.752824 | 0.7266 | 0.475924 | 1.01106 | 0.378038 | 1.21308 |
| scholten_unified_nominal_throughput__raw | nominal_throughput_proxy_seconds | 3399 | 0.752824 | 610.288 | 124.85 | 1573.87 | -1.50712e+06 | 1624.85 |
