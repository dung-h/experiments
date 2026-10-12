> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Replay validation

| context | method | comparisons | failed | unavailable | max_abs_seconds | mean_abs_seconds | max_relative |
|---|---|---|---|---|---|---|---|
| dense_fp32 | mali_style_dense_graph | 120 | 0 | 0 | 7.450580596923828e-09 | 8.769954244295756e-10 | 4.783129453997365e-05 |
| dense_fp32 | matched_metadata_mlp | 120 | 0 | 0 | 7.450580596923828e-09 | 8.692344029744466e-10 | 6.55762060834739e-05 |
| dense_fp32 | outer_train_median | 30 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| dense_fp64 | mali_style_dense_graph | 120 | 0 | 0 | 3.7834979593753815e-09 | 7.300210806230704e-10 | 2.38155995339957e-05 |
| dense_fp64 | matched_metadata_mlp | 120 | 0 | 0 | 7.450580596923828e-09 | 9.158005317052205e-10 | 4.328818032300254e-05 |
| dense_fp64 | outer_train_median | 30 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | 30 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| maestro | outer_train_median | 30 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| qpu | mali_global_mlp | 200 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| qpu | mali_logical_graph | 200 | 0 | 0 | 1.9073486328125e-06 | 3.600120544433594e-07 | 2.2459118508795385e-07 |
| qpu | outer_train_median | 50 | 0 | 0 | 0.0 | 0.0 | 0.0 |

Frozen FP32 neural gate atol=1e-6 seconds, rtol=1e-5; accommodates singleton versus frozen dense batch32 rounding. Absolute tolerance can be relatively large for tiny near-zero predictions; all absolute/relative differences remain exposed in CSV. Scalars/native formula: atol=1e-12, rtol=1e-10. No tolerance widening. Aggregated replay compares median of three exactly as the frozen estimator, not a chosen seed. Missing saved outputs are in the denominator and not model replay failures.

No model accuracy rerun or original prediction rewrite.
