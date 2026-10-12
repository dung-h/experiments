> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../../docs/thesis/README.md).

# Bảng 5. Track B theo execution context

49 predictive metric rows trong7contexts +2native/analog diagnostics. Mỗi context có frozen grouped evaluation riêng. Ladder n=840rungs từ140hashes, không840circuits độc lập;150hashes được assign. FixedMPS150 có147qualitypass/3fail. Maestro143/150 supported. Không dùng runtime của different APIs để tạo universal rank. CSV giữ đầy đủ tails/coverage; missing metric là unavailable, không0.

| Context | Runtime boundary | Estimator | n | MAE (s) | R² | Diagnostic nếu không có MAE |
| --- | --- | --- | --- | --- | --- | --- |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | outer_train_median | 150 | 0.00174902 | -0.030203 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | outer_train_median | 150 | 0.0017529 | -0.0305704 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | linear_regression | 150 | 0.00215633 | 0.398666 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | linear_regression | 150 | 0.00215008 | 0.403832 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | ridge_alpha1 | 150 | 0.00214127 | 0.393728 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | ridge_alpha1 | 150 | 0.00213581 | 0.398844 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | rbf_svr | 150 | 0.0097169 | -0.538344 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | rbf_svr | 150 | 0.00970684 | -0.544307 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | random_forest | 150 | 0.000821373 | 0.515855 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | random_forest | 150 | 0.000812914 | 0.519112 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | xgboost | 150 | 0.000969311 | 0.550085 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | xgboost | 150 | 0.000960089 | 0.555111 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | mali_style_dense_graph | 150 | 0.00171594 | 0.134538 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | mali_style_dense_graph | 150 | 0.00173022 | 0.135541 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | matched_metadata_mlp | 150 | 0.00157772 | 0.142063 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | matched_metadata_mlp | 150 | 0.00159008 | 0.144241 |  |
| cudaq_dense_statevector_fp32 | FP32 warm sample wall-clock | work_scaled_ridge | 150 | 0.000920541 | 0.872809 |  |
| cudaq_dense_statevector_fp64 | FP64 warm sample wall-clock | work_scaled_ridge | 150 | 0.000936564 | 0.864923 |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | FP64 chi16 warm get_state wall-clock | train_fold_median | 150 | 1.8706 | -0.0435379 |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | FP64 chi16 warm get_state wall-clock | ridge_alpha_1 | 150 | 1.27567 | 0.415102 |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | FP64 chi16 warm get_state wall-clock | graph_median_three_seeds | 150 | 0.296092 | 0.95072 |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | FP64 chi16 warm get_state wall-clock | family_residual_median_three_seeds | 150 | 1.14813 | 0.503425 |  |
| cudaq_mps_fp64_bond16_warm_state_execution_seconds | FP64 chi16 warm get_state wall-clock | family_agnostic_median_three_seeds | 150 | 1.1214 | 0.519938 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | azizov_gnn_hybrid_median_three_seeds | 150 | 0.406359 | 0.0914342 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | azizov_gnn_source_median_three_seeds | 150 | 0.449658 | 0.0926793 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | azizov_gnn_transpiled_median_three_seeds | 150 | 0.300764 | 0.520606 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_linear_regression_hybrid | 150 | 3.06084 | -138.577 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_linear_regression_source | 150 | 0.780337 | -4.44952 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_linear_regression_transpiled | 150 | 0.458436 | -0.0850585 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_random_forest_hybrid | 150 | 0.341629 | 0.149808 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_random_forest_source | 150 | 0.392973 | 0.149822 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_random_forest_transpiled | 150 | 0.341241 | 0.138065 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_ridge_hybrid | 150 | 0.41029 | 0.205311 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_ridge_source | 150 | 0.471672 | 0.113044 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_ridge_transpiled | 150 | 0.362203 | 0.502817 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_svr_rbf_hybrid | 150 | 0.392227 | 0.096995 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_svr_rbf_source | 150 | 0.420457 | 0.0404889 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_svr_rbf_transpiled | 150 | 0.322177 | 0.168483 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_xgboost_hybrid | 150 | 0.39151 | 0.0842793 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_xgboost_source | 150 | 0.38844 | 0.0988675 |  |
| qiskit_aer_fake_sherbrooke_noisy_opt1 | warm Aer execution after transpilation; Opt1/1024 shots | classical_xgboost_transpiled | 150 | 0.297035 | 0.288858 |  |
| qcsim_statevector_cpu_fp64_1000_shots | fresh-process engine-reported execution; optimizer off/1000shots | maestro_style_cpu_sv_component_optimizer_off_adaptation | 143 | 0.000103701 | 0.949032 |  |
| qcsim_statevector_cpu_fp64_1000_shots | fresh-process engine-reported execution; optimizer off/1000shots | outer_train_median | 143 | 0.000305782 | -0.0426355 |  |
| qcsim_statevector_cpu_fp64_1000_shots | fresh-process engine-reported execution; optimizer off/1000shots | nested_grouped_ridge | 143 | 0.000133607 | 0.882386 |  |
| qcsim_statevector_cpu_fp64_1000_shots | fresh-process engine-reported execution; optimizer off/1000shots | source_dag_graph_adaptation_cuda | 143 | 0.000423773 | -0.00917585 |  |
| cudaq_mps_ladder_c44 | warm get_state; six chi rungs; grouped hash folds | family_agnostic_ablation | 840 | 0.307519 | 0.523697 |  |
| cudaq_mps_ladder_c44 | warm get_state; six chi rungs; grouped hash folds | family_conditioned | 840 | 0.311628 | 0.521823 |  |
| cudaq_mps_ladder_family_holdout | warm get_state; six chi rungs; family holdout | family_agnostic_ablation | 840 | 0.308792 | 0.487994 |  |
| cudaq_mps_ladder_family_holdout | warm get_state; six chi rungs; family holdout | family_conditioned | 840 | 0.31101 | 0.575311 |  |
| cuTensorNet same-plan | all-slice complex64 scalar contraction vs same-plan warm | native_RUNTIME_EST | 603 |  |  | EST/warm median: core3.734012; frontier4.132416;201members/188hashes |
| Pasqal analog companion | host MPSBackend.run excluding quality capture | unpromoted_resource_formula | 18 |  |  | 17qualitypass/18; no identified predictive coefficients |
