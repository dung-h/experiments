> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Accuracy population alignment

| context | method | population | assigned | n | MAE_s | R2 | B1_median_ms |
|---|---|---|---|---|---|---|---|
| dense_fp32 | mali_style_dense_graph | full_frozen | 150 | 150 | 0.0017159400695527438 | 0.1345378116767515 | 3.230237 |
| dense_fp32 | mali_style_dense_graph | exact_timing_subset | 30 | 30 | 0.001753429142991081 | 0.6783657854820451 | 3.230237 |
| dense_fp32 | matched_metadata_mlp | full_frozen | 150 | 150 | 0.0015777218337946883 | 0.14206277869990702 | 0.4924454999999999 |
| dense_fp32 | matched_metadata_mlp | exact_timing_subset | 30 | 30 | 0.0015233294436863313 | 0.7148416771916386 | 0.4924454999999999 |
| dense_fp32 | outer_train_median | full_frozen | 150 | 150 | 0.0017490204669108303 | -0.03020297660375082 | 0.007936 |
| dense_fp32 | outer_train_median | exact_timing_subset | 30 | 30 | 0.0021353613003157077 | -0.0989077828557583 | 0.007936 |
| dense_fp64 | mali_style_dense_graph | full_frozen | 150 | 150 | 0.0017302181986451615 | 0.13554069824258796 | 3.2085874999999997 |
| dense_fp64 | mali_style_dense_graph | exact_timing_subset | 30 | 30 | 0.0008079790864333821 | 0.9250812037880595 | 3.2085874999999997 |
| dense_fp64 | matched_metadata_mlp | full_frozen | 150 | 150 | 0.0015900815471347112 | 0.14424058611073187 | 0.48615575 |
| dense_fp64 | matched_metadata_mlp | exact_timing_subset | 30 | 30 | 0.0006182988499252436 | 0.962712928068043 | 0.48615575 |
| dense_fp64 | outer_train_median | full_frozen | 150 | 150 | 0.0017529017866278688 | -0.030570386954231976 | 0.00804025 |
| dense_fp64 | outer_train_median | exact_timing_subset | 30 | 30 | 0.001437863934552297 | -0.046615067173850555 | 0.00804025 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | full_frozen | 150 | 143 | 0.00010370132150113144 | 0.9490322403412017 | 0.0017585 |
| maestro | maestro_style_cpu_sv_component_optimizer_off_adaptation | exact_timing_subset | 29 | 29 | 0.00018836622938674417 | 0.9468294901782965 | 0.0017585 |
| maestro | outer_train_median | full_frozen | 150 | 143 | 0.0003057820174825175 | -0.04263550056365051 | 0.0007344999999999999 |
| maestro | outer_train_median | exact_timing_subset | 30 | 29 | 0.0006815366379310344 | -0.0734583038219685 | 0.0007344999999999999 |
| qpu | mali_global_mlp | full_frozen | 4515 | 4515 | 0.8934229801008294 | 0.36315832410711923 | 0.60276325 |
| qpu | mali_global_mlp | exact_timing_subset | 50 | 50 | 0.7977622924811417 | 0.818089143876056 | 0.60276325 |
| qpu | mali_logical_graph | full_frozen | 4515 | 4515 | 0.8341333281956075 | 0.46809087553927997 | 5.888531250000001 |
| qpu | mali_logical_graph | exact_timing_subset | 50 | 50 | 0.7062226713466947 | 0.8926120161382467 | 5.888531250000001 |
| qpu | outer_train_median | full_frozen | 4515 | 4515 | 1.1382755517197642 | -0.14355758757093673 | 0.010141 |
| qpu | outer_train_median | exact_timing_subset | 50 | 50 | 2.6571518541022496 | -0.1434216967315145 | 0.010141 |

Full frozen scores are independently recomputed from authoritative saved predictions and match the historical AM1 accuracy table within 1e-10. No leaderboard value or prediction is replaced. Full assigned populations: QPU 4,515, each dense precision 150, Maestro150 with 143 observed targets. Exact timing subset: QPU 50, dense 30/context, Maestro component 29/30 and median30/30 with 29 evaluable labels. Detailed MedAE/RMSE, group support, source counts and denominators: accuracy_population_metrics.csv.

Query routing uses its frozen held-out outer fold, with the original three-seed median for neural outputs. The timing-subset metrics are descriptive post-pilot calculations of already-saved predictions, not new inference, reselection, calibration or tuning. Selection used metadata but is deliberately stratified and does not estimate the full-population utility without a sampling model. QPU timing50 differs from full source proportions. Pooled and source/group-balanced accuracy must not be conflated.

A descriptive accuracy-latency comparison is now possible for exactly aligned IDs and context. This does not supply a complete Pareto frontier, an exchange rate between MAE and compute time, error-tolerance guarantees or a scheduler benefit. Missing original Qonductor selector/tree and dense RF/XGB fitted states prohibit corresponding cost claims. Maestro median's extra 30th timing has no execution label and is excluded from paired accuracy.

## Sampling and interpretation

The QPU timing subset contains 38 Ma–Li, 7 Qonductor and 5 QPack observations. The full cohort contains 340, 230 and 3,945 respectively. The subset gives metadata/size coverage and heavily enriches Ma–Li, rather than approximating the pooled deployment mix. Its MAE and R² must not replace or explain full-cohort performance as a representative sample. Dense FP64 Graph R² is about 0.925 on the selected 30 but about 0.136 on the full 150. This is a population effect, not improved model fitting. Maestro's baseline latency on 29 label-matched requests is 0.7345 microseconds, slightly different from 0.7355 over all 30 timed requests.
