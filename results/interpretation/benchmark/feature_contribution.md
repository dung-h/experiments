> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Feature contribution: usable evidence and its limits

This synthesis preserves historical populations, transforms, seeds and clocks.
Older340-row diagnostics do not become full 4,515-row ablations. A smaller error
from the complete Graph pipeline is not by itself a topology effect: Graph
also sees node type, wires, node position and nominal T1/T2 that the global MLP
does not receive. Neither neural input contains shots or explicit backend ID.

## Evidence classes

| Saved comparison | What it establishes | What it cannot establish |
| --- | --- | --- |
| Test-time zeroing of node type/qubit/T1T2/index; edge removal/reversal/rewiring | Reliance of a fixed trained model under that intervention | Benefit after retraining, causality, robustness to a realistic deployment shift |
| Retrained true/reverse/shuffled/no-edge topology controls | Incremental effect in that controlled coarse representation | Universal value of native DAG topology in the full model |
| Retrained physical/static feature blocks | Predictive signal of descriptor sets under those historical splits | Individual-gate causal importance or full-model neural ablation |
| Graph/global-only models in the paper | Authors' component result in their setup | A locally replicated matched comparison on current 4515 |
| Current Graph versus global MLP | Complete-pipeline contrast on matched test observations | Topology-only effect or fairness of identical information availability |
| QWalk calibration intervention / support distance | A specific calibration-transfer sensitivity | That calibration is generally useless or every OOD error has the same cause |

## Quantitative evidence

1. **Paper-reported, not local replication.** Ma–Li's
   [Tables6–7](https://arxiv.org/html/2411.15631v1) report simulator R² .943 for
   the full model, .928 graph-only and .573 global-only. Retrained node-feature
   variants report .961 without node type, .878 without qubit index, .678
   without T1/T2 and .932 without node index. These assess different questions
   from our fixed-model zeroing. No matching local full 3020 graph-only/global-only
   result was found. The saved local full 3020 rebuild has MAE .226694s and R²
   .426203, not a reproduction of .943.

2. **Historical test-time intervention.** In the single-seed 340-row audit,
   scratch MAE is .4783s; no edges raises it to5.9652s, reverse edges to.5984s
   and rewiring to.6258s. Zeroing qubit index raises it to4.6862s; zeroing node
   type to.8964s; zeroing node index gives.5043s. This proves reliance under
   abnormal inputs, not stable incremental/topological benefit. The artifact
   used historical preprocessing and is not today's fit-only 4515 experiment.

3. **Retrained topology controls, three-seed mean.** On170 grouped QASM
   structures/340 observations with a256-bin coarse graph, node-only gives
   MAE .81454s/log-R² .76664; true DAG .84483/.73825; reverse .89670/.69912;
   shuffled .85296/.72851. True-DAG minus node-only MAE CI is approximately
   [−.0165,.0762]s; log-R² CI [−.065,.0008]. There is **no resolved topology
   advantage in this coarse representation**. These are mean seed metrics,
   not the current median-of-three prediction estimator.

4. **Retrained static blocks.** Grouped physical-descriptor MAE improves from
   QCRE-only .8274s to compiled .6889, graph-size .5766, layer .5635, timing
   .5109 and rich .4937. Edge-only summaries .5766 do not improve on graph size.
   Strict-backend+QASM rich is .5230 vs compiled .6924; family-holdout rich
   .5753 vs compiled .7228. These support useful compiled/static/timing
   information, not superiority of a GNN. Adding critical-subgraph descriptors
   to rich worsens MAE .4937→.4987 grouped, .5230→.5306 strict and .5753→.5934
   family, despite some log-R² gains. Do not select one favorable metric as proof.

5. **QWalk forensic case.** Two backend observations correspond to one QASM.
   A historical pretrained prediction36.0312s versus actual13.6962s improves
   toward19.4024s when T1/T2 are neutralized; scratch is12.8724s (neutralized
   13.2135s). Backend graphs otherwise agree. This is a specific negative
   calibration-transfer mechanism, not independent family-level evidence.
   Aggregate scratch T1/T2 zeroing barely changes MAE (.4783→.4756s).

6. **Strict transfer.** After excluding all 170 real QASM hashes from simulator
   pretraining, the compact logical screen reports scratch MAE .6831s/log-R²
   .3360; fine-tune .6956/.0816; frozen1.9152/−.2533. Fine-tune minus scratch
   MAE CI [−.0354,.0572] is unresolved; frozen minus scratch [1.0097,1.4335]
   is harmful. This is not a full-model proof about all simulator pretraining.

7. **Current feature support.** Eleven raw positions are zero throughout4515.
   Random has 18 other global fields active only at test and therefore masked;
   QNN has none, but both show extreme count-support extrapolation. There is
   no saved SHAP or retrained isolated ablation establishing predictive value
   for the eleven zero positions. Zero support implies no learnable signal for
   that training population, not irrelevance in every possible circuit corpus.

## Conclusions

Compiled structure, gate mix, width/depth and timing summaries clearly carry
runtime signal in saved matched experiments. Fixed Graph models rely on
wire/node identities and edges. Controlled saved topology evidence does not
resolve a positive DAG-topology increment; a full 4515 topology-only retrained
ablation remains unanswered. Nominal calibration can differentiate backend
Graph inputs, but may introduce harmful OOD sensitivity. Global-MLP collisions
and QWalk prevent describing calibration as demonstrated hardware adaptation.

The current parent Graph's modest pooled advantage over MLP/classical models
does not settle these component questions. Source-balanced/equal-group rankings
and paired uncertainty must accompany the pooled comparison.

## Source map

- `artifacts/mali/graph_intervention_audit_20260921/REPORT.md`
- `artifacts/validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md` and `summary.json`
- `artifacts/validation/mali_physical_dag_v1/feature_ablation/REPORT.md` and `metrics.csv`
- `artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/REPORT.md`
- `artifacts/validation/mali_physical_dag_v1/qwalk_forensics/REPORT.md`
- `artifacts/validation/mali_strict_transfer_logical_v1/evaluation/REPORT.md`
- `artifacts/mali/MA_LI_TRANSFER_BASELINE_COMPARISON.md`
- `artifacts/mali/MA_LI_REPLICATION_REPORT.md` and `full_result.json`
- Current raw/transform support: this package's ledger, width and support tables.

Exact source file hashes are in the package manifest. Paper-reported results
are explicitly separated from locally saved evidence.
