> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Experiment-completion decision

## SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS

Central benchmark comparison and measured Graph–MLP accuracy–cost contributions are scientifically defensible. Material limits concern input reconstruction, source/group support, procedure asymmetry, nominal hardware context, limited simulator resource/engine support and cost scope. They must be prominent in thesis claims. They do not make the measured comparison unanswerable.

RQ4 **Track A: ANSWERED WITHIN MEASURED SCOPE**. Source analysis.json (historical source; not bundled).

| context | model | n | mae_s | rmse_s | r2 | median_B2_ms |
|---|---|---|---|---|---|---|
| QPU exact45 | graph | 45 | 0.724176697021094 | 0.9475399431446588 | 0.8976867320714272 | 332.557293 |
| QPU exact45 | mlp | 45 | 0.8125209634576435 | 1.2575724544385458 | 0.819780020637797 | 149.5139715 |


Graph−MLP observed MAE = -0.088344266 s; conditional 95% interval [-0.3548650735233648, 0.1378862161955058] s includes zero. Ratio of marginal B2 medians = 2.224256; interval [2.067538851043498, 2.4085255418188094]. Median paired ratio = 2.370999, a different statistic. Exact45 includes 38 Ma–Li/7 Qonductor and no QPack. No full4.515 cost generalization follows.

RQ4 **Track B: ANSWERED WITHIN MEASURED SCOPE**. Sources analysis.json (historical source; not bundled); [full_context_accuracy.csv](../../accuracy_cost/dense/full_context_accuracy.csv).

| context | model | n | mae_ms | rmse_ms | r2 | median_B2_ms |
|---|---|---|---|---|---|---|
| Dense FP32 exact30 | graph | 30 | 1.7534291429910809 | 3.5967864653569914 | 0.6783657854820451 | 7.25009225 |
| Dense FP32 exact30 | mlp | 30 | 1.5233294436863314 | 3.386698694894943 | 0.7148416771916386 | 3.68970775 |
| Dense FP64 exact30 | graph | 30 | 0.807979086433382 | 1.6250050065766546 | 0.9250812037880595 | 7.07609625 |
| Dense FP64 exact30 | mlp | 30 | 0.6182988499252435 | 1.1464059946575829 | 0.962712928068043 | 3.61045225 |


MLP has lower MAE and lower B2 in both exact30 contexts with conditional paired intervals excluding equality. The conclusion is bounded to the two saved estimators, warm B2, selected circuits and recorded hardware/environment. Full150 neural R² remains about 0.13–0.14. No all-method Pareto frontier, operational break-even, scheduler-throughput gain, cross-hardware scaling or cold-start cost is established. Historical simulator execution timings are cross-environment and cannot provide matched break-even.

The previous closure v1 predates completed AM1 and described a material matched-overhead gap. This v2 supersedes that gap/status only for executed Phase4/Phase5 scope; original decisions and historical authority remain untouched. Missing strong classical fitted-state overhead limits competitive deployment selection but does not erase the bounded Graph–MLP comparison. RQ2 portability questions remain partial/non-blocking.

Mandatory new experiments: **NONE**. Existing frozen scope can stop. Future extensions require a separate research amendment.

**STOP EXPERIMENTS AND MOVE TO THESIS SYNTHESIS**
