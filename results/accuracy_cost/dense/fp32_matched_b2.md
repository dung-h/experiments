> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# FP32 matched full B2

| method | n | mae_seconds | rmse_seconds | r2 | median_request_median_B2_ms | median_request_p95_B2_ms |
|---|---|---|---|---|---|---|
| graph | 30 | 0.001753429142991081 | 0.0035967864653569915 | 0.6783657854820451 | 7.25009225 | 7.620789125 |
| mlp | 30 | 0.0015233294436863313 | 0.003386698694894943 | 0.7148416771916386 | 3.68970775 | 3.9718615 |
| median_reference | 30 | 0.0021353613003157077 | 0.006648354355353433 | -0.0989077828557583 | NOT_MEASURED | NOT_MEASURED |

Direct total from resident raw QASM to returned scalar at CUDA coordinator. Each circuit/model has3 warmups and30 repetitions, batch1. p95 is median of30 per-circuit interpolated p95 values, not pooled production tail. No outlier deletion.

Paired statistics:
{
  "median_of_paired_latency_ratios": 1.9145351169347458,
  "median_paired_latency_difference_ms": 3.555169,
  "observed_mae_graph_minus_mlp_seconds": 0.00023009969930474963,
  "ratio_of_marginal_latency_medians": 1.964950272823098
}

Conditional paired cluster bootstrap:
{
  "groups": 30,
  "mae_difference_graph_minus_mlp_seconds": [
    5.6416867543399955e-05,
    0.00045900444635966163
  ],
  "marginal_median_latency_ratio": [
    1.9162801583980307,
    2.0139734716332045
  ],
  "observations": 30,
  "paired_latency_difference_ms": [
    3.2751924999999993,
    3.92449975
  ],
  "replicates": 2000,
  "seed": 20261011,
  "unit": "paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians"
}
