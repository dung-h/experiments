> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Accuracy and full online estimator cost

## Paired conclusion

MLP has lower point MAE and lower warm B2 in both measured contexts. Conditional
95% paired intervals support these directions on the selected circuits:

| Context | Graph minus MLP MAE (ms) | 95% interval (ms) | Median paired B2 difference (ms) | 95% interval (ms) | Ratio of marginal B2 medians | 95% interval |
| --- | ---: | --- | ---: | --- | ---: | --- |
| FP32 | 0.230100 | [0.056417, 0.459004] | 3.555169 | [3.275192, 3.924500] | 1.964950 | [1.916280, 2.013973] |
| FP64 | 0.189680 | [0.027737, 0.413600] | 3.465644 | [3.100782, 3.829302] | 1.959892 | [1.897088, 2.010439] |

Thus MLP dominates this pair on the two reported criteria within these exact
30-circuit contexts. This is not an all-method or universal dominance claim.
Intervals are conditional on structurally selected subsets, not random samples
of future circuits. The median paired latency ratios are 1.914535 and 1.917204.
Those differ from the ratios of marginal medians in the table and do not inherit
that table's intervals.

## Accuracy outside the timed subset

Frozen full-core results use 150 OOF circuits per precision. Graph/MLP R² is
0.134538/0.142063 for FP32 and 0.135541/0.144241 for FP64. The timed-subset R²
values are 0.678366/0.714842 and 0.925081/0.962713. The high subset values
must not be presented as full-context estimator quality or generalization.
Full-core MLP MAE is also lower in point estimate, but no new full-core
uncertainty experiment or model selection was performed.

Train-only median-reference MedAE is lower than either neural model on the
timed subsets: 0.179651 ms for FP32 and 0.171514 ms for FP64. Its MAE/RMSE/R²
are worse. The neural advantage is therefore metric-dependent, rather than
uniformly useful on every small circuit. The reference cost was not remeasured.

## Why full B2 matters

CPU preparation accounts for 67.63%/86.16% of total accumulated B2 for
Graph/MLP in FP32 and 62.80%/83.09% in FP64. Graph inference plus inverse
transform and D2H accounts for 23.44%/28.05%, versus 6.62%/8.27% for MLP.
The Graph-specific DAG/encoding and larger representation add online work,
but raw parsing and globals are shared kinds of work paid by both estimators.
No ablation was run, so these attributions do not isolate a causal graph-only
increment by subtracting marginal stage medians.

Historical B1 was about 3.23/0.49 ms and 3.21/0.49 ms for Graph/MLP. The new
B2 includes several milliseconds of parsing and preparation omitted by B1.
The cost ordering does not reverse. The Graph/MLP ratio is about 1.96 in B2,
versus about 6.6 in historical B1, but different warmup/instrumentation and
execution occasions prevent treating the change as a matched B1/B2 speedup.

FP32: {
  "median_of_paired_latency_ratios": 1.9145351169347458,
  "median_paired_latency_difference_ms": 3.555169,
  "observed_mae_graph_minus_mlp_seconds": 0.00023009969930474963,
  "ratio_of_marginal_latency_medians": 1.964950272823098
}
Conditional accuracy interval excludes zero on this subset only.

FP64: {
  "median_of_paired_latency_ratios": 1.917203633229077,
  "median_paired_latency_difference_ms": 3.465644,
  "observed_mae_graph_minus_mlp_seconds": 0.0001896802365081385,
  "ratio_of_marginal_latency_medians": 1.9598919359756108
}
Conditional accuracy interval excludes zero on this subset only.

Cost is standalone genuine per-method work at the same external boundary. Graph pays DAG construction/encoding/large-payload bridge; MLP omits DAG only. Both pay raw parsing,RCCX normalization,seven globals and frozen transforms. Stage evidence distinguishes preparation and inference; no stage medians are summed. Source/family mixture and small target variance limit accuracy interpretation. No utility score,Pareto-frontier,universal winner or scheduler claim. Cached Pilot B1(approximately3.23/0.49ms FP32 and3.21/0.49ms FP64) omits preparation and bridge. Warmup/instrumentation/environment differ, so no exact matched B2/B1 speedup is asserted.
