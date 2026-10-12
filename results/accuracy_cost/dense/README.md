> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Dense Graph–MLP estimator accuracy–cost

SUCCESS. Both precisions pass raw-input replay and matched B2:60 context requests,480 replay comparisons,3,600 timed predictions. No training or quantum simulation.

| context | method | mae_seconds | rmse_seconds | r2 | median_request_median_B2_ms | median_request_p95_B2_ms |
|---|---|---|---|---|---|---|
| dense_fp32 | graph | 0.001753429142991081 | 0.0035967864653569915 | 0.6783657854820451 | 7.25009225 | 7.620789125 |
| dense_fp32 | mlp | 0.0015233294436863313 | 0.003386698694894943 | 0.7148416771916386 | 3.68970775 | 3.9718615 |
| dense_fp64 | graph | 0.0008079790864333821 | 0.0016250050065766545 | 0.9250812037880595 | 7.07609625 | 7.512216824999999 |
| dense_fp64 | mlp | 0.0006182988499252436 | 0.001146405994657583 | 0.962712928068043 | 3.61045225 | 3.8929926249999998 |

Read accuracy_cost_tradeoff.md, matched_accuracy_analysis.md and rq4_trackb_decision.md. Exact inputs/checkpoints and source hashes:predeclaration.json. Each precision is a separate target. Original models are float32 for both simulator precision contexts. Full context is150 frozen core OOF circuits; timing subset is30.
