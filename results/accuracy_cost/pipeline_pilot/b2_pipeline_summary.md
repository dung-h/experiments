> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Complete B2 results

| context | method | population | n | mae_seconds | medae_seconds | rmse_seconds | r2 | groups | median_full_pipeline_ms | boundary | source_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_fp32 | outer_train_median | exact_measured_IDs | 30 | 0.0021353613003157077 | 0.00017965123697649688 | 0.006648354355353433 | -0.0989077828557583 | 30 | 7.175000000000001e-05 | complete_native_scalar | {"local_dense": 30} |
| dense_fp64 | outer_train_median | exact_measured_IDs | 30 | 0.001437863934552297 | 0.0001715139987936709 | 0.006073692482948284 | -0.046615067173850555 | 30 | 7.025e-05 | complete_native_scalar | {"local_dense": 30} |
| qualified_qpu_4515 | mali_global_mlp | exact_measured_IDs | 45 | 0.8125209634576434 | 0.535864568500978 | 1.2575724544385458 | 0.819780020637797 | 44 | 218.77019 | complete_B2 | {"mali_real_qpu": 38, "qonductor_single_circuit_ibm": 7} |
| qualified_qpu_4515 | outer_train_median | exact_measured_IDs | 45 | 2.8641263459175113 | 2.6305666661449787 | 3.215408613586476 | -0.17817348614127115 | 44 | 0.00031800000000000003 | complete_native_scalar | {"mali_real_qpu": 38, "qonductor_single_circuit_ibm": 7} |

## Stage costs

| context | method | stage | queries | repetitions_per_query | median_query_median_ms | median_query_p95_ms | min_query_median_ms | max_query_median_ms |
|---|---|---|---|---|---|---|---|---|
| dense_fp32 | outer_train_median | complete_native_scalar | 30 | 30 | 7.175000000000001e-05 | 9.204999999999999e-05 | 6.9e-05 | 8e-05 |
| dense_fp64 | outer_train_median | complete_native_scalar | 30 | 30 | 7.025e-05 | 8.679999999999998e-05 | 6.7e-05 | 7.65e-05 |
| qualified_qpu_4515 | mali_global_mlp | batch_and_H2D | 45 | 30 | 0.183725 | 0.19575035 | 0.06765 | 0.188948 |
| qualified_qpu_4515 | mali_global_mlp | complete_B2 | 45 | 30 | 218.77019 | 224.46102255 | 1.9173244999999999 | 271.508448 |
| qualified_qpu_4515 | mali_global_mlp | extract_51_globals | 45 | 30 | 74.87784149999999 | 77.61310505 | 0.1524855 | 104.5867915 |
| qualified_qpu_4515 | mali_global_mlp | frozen_transform_and_tensor_encode | 45 | 30 | 0.129527 | 0.14274569999999998 | 0.059393 | 0.13544899999999999 |
| qualified_qpu_4515 | mali_global_mlp | host_median_aggregation | 45 | 30 | 0.0038665 | 0.004457549999999999 | 0.0021955 | 0.004324 |
| qualified_qpu_4515 | mali_global_mlp | parse_QASM | 45 | 30 | 138.4675225 | 146.09513215 | 0.2171825 | 164.2743405 |
| qualified_qpu_4515 | mali_global_mlp | three_seed_inference_inverse_and_D2H | 45 | 30 | 0.5426605 | 0.6143558499999999 | 0.42788349999999997 | 0.5676035 |
| qualified_qpu_4515 | outer_train_median | complete_native_scalar | 45 | 30 | 0.00031800000000000003 | 0.00045224999999999974 | 0.00024249999999999999 | 0.000353 |


Complete B2 = instrumented, independently timed raw in-memory QASM-to-host
prediction. No disk retrieval, route selection, checkpoint load or training.
Global extraction itself uses upstream structural/Supermarq calculations,
including its required DAG traversal; MLP does not use the Graph node encoder.
Latency summaries are median over per-query medians, NOT pooled repeat-N.
p95 is linearly interpolated within 30 repeats, then summarized across queries;
not a reliable worst-case service bound. Native scalar timer is close to
perf_counter/float-access overhead: no claim of nanosecond deployment endpoint.

No fresh Graph complete B2 or dense neural complete B2 data exist here.
Their gates failed before timing. Preprocessing/inference breakdown comes
only from the qualified QPU MLP, not a substitute Graph path.
