> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Prediction overhead

Units below are milliseconds of estimator work, not predicted quantum-runtime seconds. Each model bundle receives 10 warmups using its first admissible request; each selected request then has 30 retained timings. Across-request percentiles use request medians, not an SLA. No outlier removed.

| context | method | boundary | requests | raw_n | median_request_ms | p90_request_ms | p95_request_ms | max_request_ms |
|---|---|---|---|---|---|---|---|---|
| dense_fp32 | mali_style_dense_graph | B1 | 30 | 900 | 3.230237 | 3.367323300000001 | 4.148218675 | 4.218668999999999 |
| dense_fp32 | mali_style_dense_graph | P1 | 30 | 900 | 4.5828767500000005 | 17.186703550000033 | 39.160224424999996 | 86.9772375 |
| dense_fp32 | matched_metadata_mlp | B1 | 30 | 900 | 0.4924454999999999 | 0.5265952000000002 | 0.634510425 | 0.6620005 |
| dense_fp32 | matched_metadata_mlp | P1 | 30 | 900 | 0.778178 | 0.8442849500000001 | 0.97841165 | 0.9893864999999998 |
| dense_fp32 | noop_dispatch | CONTROL | 5 | 150 | 0.0057335 | 0.005856900000000001 | 0.0058877 | 0.0059185 |
| dense_fp32 | outer_train_median | B1 | 30 | 900 | 0.007936 | 0.008420800000000003 | 0.0098962 | 0.010082500000000001 |
| dense_fp64 | mali_style_dense_graph | B1 | 30 | 900 | 3.2085874999999997 | 3.28060675 | 3.313354125 | 3.8912035 |
| dense_fp64 | mali_style_dense_graph | P1 | 30 | 900 | 4.39050875 | 8.421954000000001 | 12.13295949999998 | 86.78129600000001 |
| dense_fp64 | matched_metadata_mlp | B1 | 30 | 900 | 0.48615575 | 0.4988801999999999 | 0.5069924999999998 | 0.6552034999999999 |
| dense_fp64 | matched_metadata_mlp | P1 | 30 | 900 | 0.7811055 | 0.80203585 | 0.8141128000000001 | 0.9920484999999999 |
| dense_fp64 | noop_dispatch | CONTROL | 5 | 150 | 0.0059575 | 0.0060147 | 0.0060181 | 0.0060215 |
| dense_fp64 | outer_train_median | B1 | 30 | 900 | 0.00804025 | 0.00819555 | 0.0083102 | 0.010306 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | B1 | 29 | 870 | 0.0017585 | 0.0018475000000000002 | 0.0018532000000000002 | 0.0019605 |
| maestro | noop_dispatch | CONTROL | 1 | 30 | 0.00028000000000000003 | 0.00028000000000000003 | 0.00028000000000000003 | 0.00028000000000000003 |
| maestro | outer_train_median | B1 | 30 | 900 | 0.0007355 | 0.00076765 | 0.000787 | 0.000822 |
| qpu | mali_global_mlp | B1 | 50 | 1500 | 0.60276325 | 0.6251979 | 0.628588225 | 0.6311135 |
| qpu | mali_global_mlp | P1 | 50 | 1500 | 0.7810335 | 0.7936567999999999 | 0.7972738 | 0.8069905 |
| qpu | mali_logical_graph | B1 | 50 | 1500 | 5.888531250000001 | 6.888106850000001 | 7.048309724999999 | 7.3862895 |
| qpu | mali_logical_graph | P1 | 50 | 1500 | 15.647386 | 18.363862850000004 | 19.016317424999997 | 19.957349999999998 |
| qpu | noop_dispatch | CONTROL | 5 | 150 | 0.007389 | 0.007453400000000001 | 0.0074632 | 0.0074730000000000005 |
| qpu | outer_train_median | B1 | 50 | 1500 | 0.010141 | 0.0103316 | 0.010394124999999999 | 0.0105445 |

## Cold setup

| context | fresh_processes | median_setup_seconds | min_seconds | max_seconds |
|---|---|---|---|---|
| dense_fp32 | 15 | 1.999815438 | 1.96576365 | 2.070297266 |
| dense_fp64 | 15 | 1.975082987 | 1.965221276 | 2.057099243 |
| maestro | 3 | 0.01147431 | 0.011220897 | 0.013697795 |
| qpu | 15 | 2.510281128 | 2.497695254 | 2.676069169 |

B3 is a context/fold bundle, not six independent cold deployments: shared imports, CUDA initialization, all three seeds for both neural estimators, feature loading and 10 warmups. Stage times, fresh parent launch-to-child-exit and first-after-ready intervals are separate in receipts. Parent process launch-to-ready is not directly instrumented; child-start-to-ready excludes launcher/interpreter startup and must not be called complete cold end-to-end latency. Three fresh processes/bundle; OS cache uncontrolled.

B1: loaded models, resident GPU payloads, inverse target transform and median3 host result. P1 QPU: saved NPZ/global input loading/hash, feature transform, batching/H2D and inference. P1 dense: saved normalized IR/global metadata to tensor encode/H2D and inference. P1 Qiskit (if qualified): in-memory frozen compiled QASM parsing plus native duration and optional frozen affine bridge. P1 is a partial input pipeline, not B2 from raw logical QASM. Full B2 NOT EXECUTED. Maestro B1 includes knot lookup/count conversion/formula and saved shots1000. No quantum workloads.

Memory absolute completed-bundle/process high-water: RSS 1.3469 GiB, CUDA reserved 0.2012 GiB. Memory is not isolated per-B1 allocation. Shared loaded assets, initialization/cache and profiling lifetimes remain included. Sampled observer peaks are lower bounds. Failed manual Qiskit parse peak is NOT MEASURED, so the whole pilot cannot claim independently verified RAM-cap compliance for that branch.
