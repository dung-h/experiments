> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Frozen predictive accuracy

Timed subset:

| context | method | n | mae_seconds | medae_seconds | rmse_seconds | r2 | mean_signed_residual_seconds |
|---|---|---|---|---|---|---|---|
| dense_fp32 | graph | 30 | 0.001753429142991081 | 0.0006072786782169715 | 0.0035967864653569915 | 0.6783657854820451 | 0.0005221433675615117 |
| dense_fp32 | mlp | 30 | 0.0015233294436863313 | 0.0003785365042858757 | 0.003386698694894943 | 0.7148416771916386 | 0.00049059610949674 |
| dense_fp32 | median_reference | 30 | 0.0021353613003157077 | 0.00017965123697649688 | 0.006648354355353433 | -0.0989077828557583 | -0.0019889673373351493 |
| dense_fp64 | graph | 30 | 0.0008079790864333821 | 0.0003573168796719983 | 0.0016250050065766545 | 0.9250812037880595 | 9.38784772491393e-05 |
| dense_fp64 | mlp | 30 | 0.0006182988499252436 | 0.0003180680505465716 | 0.001146405994657583 | 0.962712928068043 | 8.767376566538587e-05 |
| dense_fp64 | median_reference | 30 | 0.001437863934552297 | 0.0001715139987936709 | 0.006073692482948284 | -0.046615067173850555 | -0.0012683474696435345 |

Full frozen core context (150 OOF/context, not204 envelope or30 subset):

| context | population | method | n | mae_seconds | medae_seconds | rmse_seconds | r2 | mean_signed_residual_seconds |
|---|---|---|---|---|---|---|---|---|
| dense_fp32 | full_frozen_core_OOF | graph | 150 | 0.0017159400695527438 | 0.0004239435220370069 | 0.008692055444874855 | 0.1345378116767516 | -0.00016446050018809427 |
| dense_fp32 | full_frozen_core_OOF | mlp | 150 | 0.0015777218337946883 | 0.0003251862435718067 | 0.008654185374953513 | 0.1420627786999069 | -0.00020681364772220454 |
| dense_fp64 | full_frozen_core_OOF | graph | 150 | 0.0017302181986451615 | 0.0004327795759309083 | 0.008636725151971551 | 0.13554069824258785 | -0.000167005696130218 |
| dense_fp64 | full_frozen_core_OOF | mlp | 150 | 0.0015900815471347112 | 0.0003329506653244607 | 0.008593155405291413 | 0.14424058611073176 | -0.00020758544337392475 |


Accuracy uses frozen predictions and R5 warm sample(32) targets. Timing estimates are replay checks, not newly selected predictions. Reference uses already-frozen outer-train medians, no test-label fitting. Paired errors, signed residuals and target seconds:paired_circuit_results.csv. Cluster CIs use30 existing QASM-hash groups/context and2,000 replicates seed20261011. Repetitions are not independent circuits. Subset chosen for structural coverage, not population-random inference. Full-context scores unchanged.
