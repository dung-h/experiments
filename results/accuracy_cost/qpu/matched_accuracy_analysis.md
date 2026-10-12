> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Frozen accuracy on exactly 45 observations

| source | method | n | mae_seconds | medae_seconds | rmse_seconds | r2 |
|---|---|---|---|---|---|---|
| all_selected | graph | 45 | 0.724176697021094 | 0.479326275266601 | 0.9475399431446588 | 0.8976867320714272 |
| all_selected | mlp | 45 | 0.8125209634576435 | 0.535864568500978 | 1.2575724544385458 | 0.819780020637797 |
| all_selected | median_reference | 45 | 2.8641263459175117 | 2.6305666661449787 | 3.215408613586476 | -0.17817348614127115 |
| mali_real_qpu | graph | 38 | 0.6300089369814714 | 0.47438465917903727 | 0.8113649777939788 | 0.8735935853565961 |
| mali_real_qpu | mlp | 38 | 0.7296105819124038 | 0.5038559746695963 | 1.1366031621697956 | 0.7519413830607345 |
| mali_real_qpu | median_reference | 38 | 2.763026128036109 | 2.497226088769491 | 3.1417517330568656 | -0.8953103081552296 |
| qonductor_single_circuit_ibm | graph | 7 | 1.23537310866476 | 1.5864573795025636 | 1.4825912951996563 | -0.4422836895635769 |
| qonductor_single_circuit_ibm | mlp | 7 | 1.2626058918460867 | 0.8893238292800292 | 1.7758718214543883 | -1.0693360929797042 |
| qonductor_single_circuit_ibm | median_reference | 7 | 3.412956100130842 | 3.3423584851956787 | 3.588979454472369 | -7.451810376950654 |

Paired cluster bootstrap: 2,000 replicates, seed 20261011, 44 structural groups. Intervals are conditional on this selected subset, not population-random sampling.

{'all_selected': {'groups': 44, 'observations': 45, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.3548650735233648, 0.1378862161955058], 'paired_latency_difference_ms': [150.23852350000004, 219.50463274999998], 'marginal_median_latency_ratio': [2.067538851043498, 2.4085255418188094], 'replicates': 2000, 'seed': 20261011}, 'mali_real_qpu': {'groups': 37, 'observations': 38, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.40985907664732296, 0.12686176500357105], 'paired_latency_difference_ms': [162.080745, 227.08486325], 'marginal_median_latency_ratio': [2.1090199453714207, 2.438538972082125], 'replicates': 2000, 'seed': 20261011}, 'qonductor_single_circuit_ibm': {'groups': 7, 'observations': 7, 'unit': 'paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians', 'mae_difference_graph_minus_mlp_seconds': [-0.8043819274221149, 0.7319414956229073], 'paired_latency_difference_ms': [5.573688499999999, 6.033076999999999], 'marginal_median_latency_ratio': [2.4771378020795045, 2.517590671300797], 'replicates': 2000, 'seed': 20261011}}

Source-balanced macro MAE: {'graph': 0.9326910228231157, 'mlp': 0.9961082368792453, 'median_reference': 3.0879911140834757}. No full 4,515-cohort result changed. The median reference comes from saved outer-train-only parameters and was not refitted. Labels measure QPU observed service time in seconds, not estimator computation time.
