> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# FP64 matched full B2

| method | n | mae_seconds | rmse_seconds | r2 | median_request_median_B2_ms | median_request_p95_B2_ms |
|---|---|---|---|---|---|---|
| graph | 30 | 0.0008079790864333821 | 0.0016250050065766545 | 0.9250812037880595 | 7.07609625 | 7.512216824999999 |
| mlp | 30 | 0.0006182988499252436 | 0.001146405994657583 | 0.962712928068043 | 3.61045225 | 3.8929926249999998 |
| median_reference | 30 | 0.001437863934552297 | 0.006073692482948284 | -0.046615067173850555 | NOT_MEASURED | NOT_MEASURED |

Direct total from resident raw QASM to returned scalar at CUDA coordinator. Each circuit/model has3 warmups and30 repetitions, batch1. p95 is median of30 per-circuit interpolated p95 values, not pooled production tail. No outlier deletion.

Paired statistics:
{
  "median_of_paired_latency_ratios": 1.917203633229077,
  "median_paired_latency_difference_ms": 3.465644,
  "observed_mae_graph_minus_mlp_seconds": 0.0001896802365081385,
  "ratio_of_marginal_latency_medians": 1.9598919359756108
}

Conditional paired cluster bootstrap:
{
  "groups": 30,
  "mae_difference_graph_minus_mlp_seconds": [
    2.7736609020697263e-05,
    0.0004135997958898464
  ],
  "marginal_median_latency_ratio": [
    1.8970877960339763,
    2.0104388025326574
  ],
  "observations": 30,
  "paired_latency_difference_ms": [
    3.1007815,
    3.8293017500000004
  ],
  "replicates": 2000,
  "seed": 20261011,
  "unit": "paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians"
}
