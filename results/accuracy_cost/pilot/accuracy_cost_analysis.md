> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Accuracy and estimator cost

| context | method | accuracy_n | accuracy_assigned | frozen_mae_seconds | frozen_r2 | timing_n | B1_median_request_ms | accuracy_population_vs_timing | target_clock |
|---|---|---|---|---|---|---|---|---|---|
| cudaq_dense_statevector_fp32 | mali_style_dense_graph | 150 | 150 | 0.0017159400695527279 | 0.13453781167675194 | 30 | 3.230237 | different denominators; not per-observation Pareto frontier | FP32 warm sample wall-clock |
| cudaq_dense_statevector_fp32 | matched_metadata_mlp | 150 | 150 | 0.0015777218337946887 | 0.14206277869990747 | 30 | 0.4924454999999999 | different denominators; not per-observation Pareto frontier | FP32 warm sample wall-clock |
| cudaq_dense_statevector_fp32 | outer_train_median | 150 | 150 | 0.0017490204669108251 | -0.030202976603751708 | 30 | 0.007936 | different denominators; not per-observation Pareto frontier | FP32 warm sample wall-clock |
| cudaq_dense_statevector_fp64 | mali_style_dense_graph | 150 | 150 | 0.0017302181986451515 | 0.13554069824258741 | 30 | 3.2085874999999997 | different denominators; not per-observation Pareto frontier | FP64 warm sample wall-clock |
| cudaq_dense_statevector_fp64 | matched_metadata_mlp | 150 | 150 | 0.0015900815471347168 | 0.14424058611073276 | 30 | 0.48615575 | different denominators; not per-observation Pareto frontier | FP64 warm sample wall-clock |
| cudaq_dense_statevector_fp64 | outer_train_median | 150 | 150 | 0.0017529017866278656 | -0.030570386954233308 | 30 | 0.00804025 | different denominators; not per-observation Pareto frontier | FP64 warm sample wall-clock |
| qcsim_statevector_cpu_fp64_1000_shots | maestro_style_cpu_sv_component_optimizer_off_adaptation | 143 | 150 | 0.00010370132150117413 | 0.94903224034119771 | 29 | 0.0017585 | different denominators; not per-observation Pareto frontier | fresh-process engine-reported execution; optimizer off/1000shots |
| qcsim_statevector_cpu_fp64_1000_shots | outer_train_median | 143 | 150 | 0.0003057820174825125 | -0.042635500563666273 | 30 | 0.0007355 | different denominators; not per-observation Pareto frontier | fresh-process engine-reported execution; optimizer off/1000shots |
| qualified_qpu_4515 | mali_global_mlp | 4515 | 4515 | 0.89342298010082932 | 0.36315832410711923 | 50 | 0.60276325 | different denominators; not per-observation Pareto frontier | archived_observed_service_execution_time |
| qualified_qpu_4515 | mali_logical_graph | 4515 | 4515 | 0.83413332819560748 | 0.46809087553927997 | 50 | 5.888531250000001 | different denominators; not per-observation Pareto frontier | archived_observed_service_execution_time |
| qualified_qpu_4515 | outer_train_median | 4515 | 4515 | 1.1382755517197642 | -0.14355758757093651 | 50 | 0.010141 | different denominators; not per-observation Pareto frontier | archived_observed_service_execution_time |

## Paired prediction latency

| context | shared_requests | structural_groups | equal_group_mean_graph_minus_mlp_ms | bootstrap_low_ms | bootstrap_high_ms | bootstrap_unit | bootstrap_seed | repetitions_not_independent |
|---|---|---|---|---|---|---|---|---|
| qpu | 50 | 49 | 5.324746719387755 | 5.143610678826531 | 5.5017339218112244 | structural group | 42 | True |
| dense_fp32 | 30 | 30 | 2.7610037333333333 | 2.6580674779166666 | 2.8813788025 | structural group | 42 | True |
| dense_fp64 | 30 | 30 | 2.68691905 | 2.6346053775 | 2.7504598404166667 | structural group | 42 | True |

FACT: all measured fitted variants pass replay on their selected frozen held-out identities. Latency evidence is narrow and environment-specific. Full accuracy and timing subset have different denominators; no per-observation accuracy-cost pairing is fabricated. Dense precisions are different simulator contexts, not different estimator numeric precisions.

INTERPRETATION: Graph adds graph-convolution and input handling over MLP. QPU input contracts are asymmetric (node calibration information versus globals), so the difference is not pure topology benefit. The paired latency comparison holds selected workload and fold identities fixed. P1-B1 differences describe the measured partial path, not parsing/transpilation costs missing from full B2. Frozen QPU equal-group MAE differs from pooled interpretation; QPack dominates observation-weighted accuracy.

Unavailable Qonductor winning selectors/trees, dense RF/XGB, Azizov and other fitted comparators have no complete serialized original state. They receive no guessed cost, no new refit and no Pareto claim. No accuracy-cost advantage over the classical selector or scheduler-throughput gain is established. No scaling law or operational deployment readiness follows from this small workstation pilot.

## Exact workload cost versus saved execution

| context | method | matched_requests | median_estimation_over_saved_execution | estimation_slower_than_saved_execution_n |
|---|---|---|---|---|
| dense_fp32 | mali_style_dense_graph | 30 | 13.498039752917748 | 27 |
| dense_fp32 | matched_metadata_mlp | 30 | 2.0307128552730322 | 20 |
| dense_fp32 | outer_train_median | 30 | 0.033079033833087405 | 0 |
| dense_fp64 | mali_style_dense_graph | 30 | 16.0967698307204 | 28 |
| dense_fp64 | matched_metadata_mlp | 30 | 2.446228184639695 | 22 |
| dense_fp64 | outer_train_median | 30 | 0.040221534539040676 | 0 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | 29 | 0.005098333237210912 | 0 |
| maestro | outer_train_median | 29 | 0.0021887075379202954 | 0 |
| qpu | mali_global_mlp | 50 | 7.029301384943375e-05 | 0 |
| qpu | mali_logical_graph | 50 | 0.000700873780905921 | 0 |
| qpu | outer_train_median | 50 | 1.1935488088367232e-06 | 0 |

These ratios join workload ID/hash to its historical measured execution target and AM1 median B1 latency. They are not a new simulator execution or production scheduling test. B1 Graph exceeds saved dense execution in 27/30 FP32 and 28/30 FP64 requests; MLP does so in 20/30 and 22/30. This is a negative overhead finding for this small-width subset, not universal scaling. Full accuracy remains based on150/context, timing on30/context. QPU50 query costs are small relative to their archived service labels, without establishing deployment usefulness. Exact joins: workload_accuracy_cost.csv.
