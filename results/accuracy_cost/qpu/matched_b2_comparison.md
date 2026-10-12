> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Standalone matched full B2

| source | method | n | median_request_median_B2_ms | median_request_p95_B2_ms |
|---|---|---|---|---|
| all_selected | graph | 45 | 332.557293 | 343.18168099999997 |
| all_selected | mlp | 45 | 149.5139715 | 161.0664207 |
| all_selected | median_reference | 45 | NOT_MEASURED | NOT_MEASURED |
| mali_real_qpu | graph | 38 | 357.89728275 | 368.468615725 |
| mali_real_qpu | mlp | 38 | 153.866288 | 164.963270725 |
| mali_real_qpu | median_reference | 38 | NOT_MEASURED | NOT_MEASURED |
| qonductor_single_circuit_ibm | graph | 7 | 9.6221195 | 10.087154 |
| qonductor_single_circuit_ibm | mlp | 7 | 3.8299209999999997 | 4.04390695 |
| qonductor_single_circuit_ibm | median_reference | 7 | NOT_MEASURED | NOT_MEASURED |

Observed paired statistics: {'observed_mae_graph_minus_mlp_seconds': -0.08834426643654947, 'median_paired_latency_difference_ms': 183.04332150000002, 'ratio_of_marginal_latency_medians': 2.2242556308525323, 'median_of_paired_latency_ratios': 2.3709988865833393}

The total is independently host-timed from resident raw QASM to returned scalar prediction. There are 30 repeats per ID/method, 3 warmups and 15/15 first-order positions. No outliers were removed. p95 within 30 repeats is interpolated, not a production tail bound. Repetitions are not independent circuits.

{'all_selected': {'groups': 44, 'observations': 45, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.3548650735233648, 0.1378862161955058], 'paired_latency_difference_ms': [150.23852350000004, 219.50463274999998], 'marginal_median_latency_ratio': [2.067538851043498, 2.4085255418188094], 'replicates': 2000, 'seed': 20261011}, 'mali_real_qpu': {'groups': 37, 'observations': 38, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.40985907664732296, 0.12686176500357105], 'paired_latency_difference_ms': [162.080745, 227.08486325], 'marginal_median_latency_ratio': [2.1090199453714207, 2.438538972082125], 'replicates': 2000, 'seed': 20261011}, 'qonductor_single_circuit_ibm': {'groups': 7, 'observations': 7, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.8043819274221149, 0.7319414956229073], 'paired_latency_difference_ms': [5.573688499999999, 6.033076999999999], 'marginal_median_latency_ratio': [2.4771378020795045, 2.517590671300797], 'replicates': 2000, 'seed': 20261011}}

Nonoverlapping shares use sums of stage durations divided by the sum of independently measured totals for the same 1,350 requests per model. CPU spans are nested inside host wait. The bridge residual includes pipe I/O, scheduling, outer-envelope encoding and uninstrumented CPU work, not pure transport latency.

[{'method': 'mali_logical_graph', 'denominator_total_ns': 404973388736, 'shares': {'preparation': 0.761086175620114, 'serialization': 0.06666892867274446, 'IPC_and_unattributed_bridge': 0.051787842152937094, 'host_deserialization': 0.0919014028086225, 'tensor_and_H2D': 0.005256225093317535, 'inference_inverse_D2H': 0.023029963467747532, 'aggregation': 1.397951854977462e-05, 'coordinator_residual': 0.0002554826659670901}, 'stage_sums_ns': {'preparation': 308219647661, 'serialization': 26999141968, 'IPC_and_unattributed_bridge': 20972697932, 'host_deserialization': 37217622525, 'tensor_and_H2D': 2128631288, 'inference_inverse_D2H': 9326522348, 'aggregation': 5661333, 'coordinator_residual': 103463681}}, {'method': 'mali_global_mlp', 'denominator_total_ns': 178686912580, 'shares': {'preparation': 0.9724880083324567, 'serialization': 0.012445842932141618, 'IPC_and_unattributed_bridge': 0.007865674876276941, 'host_deserialization': 0.0004192529207557927, 'tensor_and_H2D': 0.0018829184977347823, 'inference_inverse_D2H': 0.0043708738694017675, 'aggregation': 3.195291091866489e-05, 'coordinator_residual': 0.0004954756603137454}, 'stage_sums_ns': {'preparation': 173770879730, 'serialization': 2223909248, 'IPC_and_unattributed_bridge': 1405493159, 'host_deserialization': 74915010, 'tensor_and_H2D': 336452893, 'inference_inverse_D2H': 781017957, 'aggregation': 5709567, 'coordinator_residual': 88535016}}]

Shared-preparation opportunity: QASM parsing and the 51 globals could legitimately be reused when deploying both models together. Exclusive measured CPU durations show potential savings. This is an analytical opportunity, not measured shared-pipeline latency. Primary standalone costs do not subtract these stages.
