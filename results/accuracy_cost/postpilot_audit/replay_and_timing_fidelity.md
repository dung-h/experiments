> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Replay and measurement fidelity

| context | method | selected_ids | comparisons | per_seed_comparisons | ensemble_comparisons | native_comparisons | passed | unavailable |
|---|---|---|---|---|---|---|---|---|
| dense_fp32 | mali_style_dense_graph | 30 | 120 | 90 | 30 | 0 | 120 | 0 |
| dense_fp32 | matched_metadata_mlp | 30 | 120 | 90 | 30 | 0 | 120 | 0 |
| dense_fp32 | outer_train_median | 30 | 30 | 0 | 0 | 30 | 30 | 0 |
| dense_fp64 | mali_style_dense_graph | 30 | 120 | 90 | 30 | 0 | 120 | 0 |
| dense_fp64 | matched_metadata_mlp | 30 | 120 | 90 | 30 | 0 | 120 | 0 |
| dense_fp64 | outer_train_median | 30 | 30 | 0 | 0 | 30 | 30 | 0 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | 30 | 30 | 0 | 0 | 30 | 29 | 1 |
| maestro | outer_train_median | 30 | 30 | 0 | 0 | 30 | 30 | 0 |
| qpu | mali_global_mlp | 50 | 200 | 150 | 50 | 0 | 200 | 0 |
| qpu | mali_logical_graph | 50 | 200 | 150 | 50 | 0 | 200 | 0 |
| qpu | outer_train_median | 50 | 50 | 0 | 0 | 50 | 50 | 0 |

Counts: 660 per-seed + 220 ensemble + 170 scalar/formula =1,050 comparisons, 1,049 PASS and 1 unavailable. Unique context requests: 140: QPU 50, FP32/FP64 each 30, Maestro 30. Dense circuits can repeat across precision/Maestro; do not sum these as unique physical circuits. No analytical 50 replay completed; they are excluded from 1,050, not silently successful. Tolerance: 1e-6 s+1e-5*abs(saved) for FP32 neural, 1e-12+1e-10 for scalar. Maximum small-prediction relative discrepancy can exceed 1e-5 because absolute tolerance applies. Values remain within gate. All 220 ensemble saved medians independently match three saved seed outputs and full OOF medians.

Maestro unavailable ID `a53b2a3a1ba8397a96b487c44f825f3cab2e0c714b69298f2ab0a839975b173c`: target_status=unavailable_unsupported, runtime empty, normalized metadata missing, component_prediction_status=normalized_execution_metadata_unavailable. Coverage is 29/30 component timing, not a model output lost during AM1; reference timing 30/30 but accuracy 29/30.

B1 host timer starts after initial CUDA sync, includes three actual sequential model calls, each output transfer/inverse transform, host median and final sync. All batch 1. No measurement is seed-latency multiplied by 3. Graph/MLP input contracts differ as frozen, so comparisons concern implemented estimators, not a controlled pure-topology causal effect. Scalar references in GPU workers pay final sync overhead; controls ~6–7 microseconds versus reference ~8–10 microseconds dominate intrinsic scalar compute. Maestro CPU no-op ~0.28 microseconds is smaller but nonzero relative to component ~1.76 microseconds. No controls are subtracted. Control is no-op with the timing wrapper, not separate device-kernel time.

Ten warmups per method/bundle use its first eligible request, not ten for every workload. 30 primary repetitions come from cold process 0 only; cold1/2 contribute startup and first-after-ready, not three independent warm campaigns. Matched Graph/MLP methods rotate order across repetitions. Bootstrap of request medians by structural group addresses workload selection, not repeated-process or operating-system temporal uncertainty.

B1 and partial P1 are timed sequentially; P1 also changes caches/allocator state, so subtraction is not isolated stage profiling. P1 QPU includes disk read/hash; dense starts at saved IR/global metadata. Neither is complete B2. B3 has48 fresh process bundles, shared six models for neural contexts, all inputs, expected prediction CSV/hash validation and warmups. It starts after Python interpreter startup; child-start-to-ready and parent-launch-to-exit are different intervals. There is no parent-launch-to-ready timer or standalone model cold deployment measurement. Memory is bundle/process absolute peak. Native manual failure lacks per-input identity/progress and peak RAM trace.

## Independently reconciled paired latency

| context | shared_requests | groups | graph_minus_mlp_ms | low_ms | high_ms |
|---|---|---|---|---|---|
| qpu | 50 | 49 | 5.324746719387755 | 5.143610678826531 | 5.5017339218112244 |
| dense_fp32 | 30 | 30 | 2.7610037333333333 | 2.6580674779166666 | 2.8813788025 |
| dense_fp64 | 30 | 30 | 2.68691905 | 2.6346053775 | 2.7504598404166667 |

The fixed seed 42, 2,000 structural-group bootstrap reproduces the historical intervals. It resamples selected groups, not timing repetitions. These positive intervals support extra Graph cost within the selected contexts. They do not include independent day/process variability, changed background load, unobserved full B2 or hardware portability.
