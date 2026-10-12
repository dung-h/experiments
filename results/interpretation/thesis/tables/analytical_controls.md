> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../../docs/thesis/README.md).

# Analytical raw, affine và matched controls

35 dòng giữ nguyên. Eligible controls dùng chính training eligibility của route; không thay bằng global reference của full cohort. Calibration là predictor bổ sung. Scholten affine .74835 so eligible median .82007 và source median .69207, không phải hai median đều .69207.

| Route | Variant/control | n | MAE (s) | R² |
| --- | --- | --- | --- | --- |
| hyb_nominal_r2_effective_cost_v1__raw | affine | 4175 | 1.1977 | -0.289082 |
| hyb_nominal_r2_effective_cost_v1__raw | eligible_intercept | 4175 | 1.06955 | -0.0486858 |
| hyb_nominal_r2_effective_cost_v1__raw | eligible_median | 4175 | 1.0453 | -0.301058 |
| hyb_nominal_r2_effective_cost_v1__raw | eligible_source_median | 4175 | 0.935549 | -0.0290117 |
| hyb_nominal_r2_effective_cost_v1__raw | raw | 4175 | 5.36134 | -18.9471 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | affine | 4175 | 1.1976 | -0.288741 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | eligible_intercept | 4175 | 1.06955 | -0.0486858 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | eligible_median | 4175 | 1.0453 | -0.301058 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | eligible_source_median | 4175 | 0.935549 | -0.0290117 |
| hyb_nominal_r2_shot_effective_cost_v1__raw | raw | 4175 | 2600.51 | -3.93125e+07 |
| qcre_source_estimator__raw | affine | 4515 | 1.06129 | 0.246164 |
| qcre_source_estimator__raw | eligible_intercept | 4515 | 1.16156 | -0.035409 |
| qcre_source_estimator__raw | eligible_median | 4515 | 1.13828 | -0.143558 |
| qcre_source_estimator__raw | eligible_source_median | 4515 | 0.974362 | 0.181923 |
| qcre_source_estimator__raw | raw | 4515 | 6.11599 | -17.1328 |
| qcre_source_estimator_shot_scaled__raw | affine | 4515 | 1.05127 | 0.266082 |
| qcre_source_estimator_shot_scaled__raw | eligible_intercept | 4515 | 1.16156 | -0.035409 |
| qcre_source_estimator_shot_scaled__raw | eligible_median | 4515 | 1.13828 | -0.143558 |
| qcre_source_estimator_shot_scaled__raw | eligible_source_median | 4515 | 0.974362 | 0.181923 |
| qcre_source_estimator_shot_scaled__raw | raw | 4515 | 5.67186 | -14.5738 |
| qiskit_estimate_duration_snapshot__raw | affine | 4515 | 1.06099 | 0.246555 |
| qiskit_estimate_duration_snapshot__raw | eligible_intercept | 4515 | 1.16156 | -0.035409 |
| qiskit_estimate_duration_snapshot__raw | eligible_median | 4515 | 1.13828 | -0.143558 |
| qiskit_estimate_duration_snapshot__raw | eligible_source_median | 4515 | 0.974362 | 0.181923 |
| qiskit_estimate_duration_snapshot__raw | raw | 4515 | 6.11598 | -17.1328 |
| qiskit_shot_scaled_schedule_seconds__raw | affine | 4515 | 1.0497 | 0.268569 |
| qiskit_shot_scaled_schedule_seconds__raw | eligible_intercept | 4515 | 1.16156 | -0.035409 |
| qiskit_shot_scaled_schedule_seconds__raw | eligible_median | 4515 | 1.13828 | -0.143558 |
| qiskit_shot_scaled_schedule_seconds__raw | eligible_source_median | 4515 | 0.974362 | 0.181923 |
| qiskit_shot_scaled_schedule_seconds__raw | raw | 4515 | 5.65423 | -14.4763 |
| scholten_unified_nominal_throughput__raw | affine | 3399 | 0.748352 | 0.351767 |
| scholten_unified_nominal_throughput__raw | eligible_intercept | 3399 | 0.830983 | -0.0279214 |
| scholten_unified_nominal_throughput__raw | eligible_median | 3399 | 0.820067 | -0.0387434 |
| scholten_unified_nominal_throughput__raw | eligible_source_median | 3399 | 0.69207 | 0.270899 |
| scholten_unified_nominal_throughput__raw | raw | 3399 | 610.288 | -1.50712e+06 |
