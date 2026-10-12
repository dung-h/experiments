> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Central claims and their limits

Central thesis objective: systematic comparison of runtime-estimation pipelines for archived real-QPU execution and recorded simulator configurations, using explicit input, clock, split, coverage and quality semantics.

The essential claims are: (1) qualified shared-row/context evaluation is reproducible and attributable; (2) conditional accuracy and failure patterns are established against simple/classical/analytical controls; (3) measured warm Graph–MLP cost includes preparation, not only cached neural inference; (4) applicability/fidelity boundaries prevent schedule, service, sample, state, contraction and analog costs being silently interchanged. A universal estimator, graph causal benefit or reproduction of every original paper is not required to establish these claims.

## Track A

| method | rows | mae_s | r2 | source_balanced_mae_s | equal_group_mae_s |
|---|---|---|---|---|---|
| mali_logical_graph | 4515 | 0.83413332819560748 | 0.46809087553927997 | 0.89802939168141027 | 0.63324205699524772 |
| mali_global_mlp | 4515 | 0.89342298010082932 | 0.36315832410711923 | 0.97601920576331747 | 0.58069688608442793 |
| qonductor_budgeted_selector | 4515 | 0.88665632813843842 | 0.41130738284003998 | 0.87898234563415845 | 0.66929968981550914 |
| qonductor_polynomial_regression | 4515 | 0.91772468272167707 | 0.40240929864114194 | 1.0079100256276019 | 0.63450748550353031 |
| outer_train_median | 4515 | 1.1382755517197642 | -0.14355758757093651 | 2.1706400039416889 | 2.3038735713343641 |
| outer_train_source_median | 4515 | 0.97436199662306222 | 0.18192289901723924 | 1.2296540459716638 | 1.4786321928279298 |

Source: [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv); shared_row_comparisons.csv (historical source; not bundled); [qpu_source_metrics_v1.csv](../../analysis/qpu_source_metrics.csv). Graph–MLP point difference favors Graph but grouped CI crosses zero. Graph–selector paired delta is −0.052523 s with interval [−0.317858, +0.457908] s. Per-source/group views change rankings. QPack 3.945/4.515 rows from six structures dominates pooled frequency; 165 groups and qualified reconstructions limit unseen-design claims.

Original QCRE critical-path component and native Qiskit schedule remain different clocks from service. Hyb is an effective-cost adaptation. Scholten current scored variant is a generalized proxy (3.399 eligible); original effective-depth model has no verified eligible rows. Outer-train affine/log calibration is a learned bridge. It does not identify physical launch/inter-shot overhead. Partial eligible coverage must remain beside errors.

Ma–Li's 41 là supported width tái dựng cho original simulator corpus; parent real-QPU width là 40 từ 51 raw positions. Current Dense adaptation dùng seven globals, vì vậy không gán width41 cho mọi simulator branch. Historical author real tensor vẫn unverified; no faithful-paper claim follows merely from matching a displayed feature count.

## Track B

Within-context OOF compares 150 core hashes where supported: noisy Aer, Dense FP32/FP64, fixed MPS, joint runtime–quality MPS and CPU QCSim component. Native same-plan cuTensorNet and Pasqal analog companion answer separate timing/quality questions. Compiled Aer GNN improves versus source GNN, but superiority over XGBoost is unresolved. Dense RF has lower local MAE than neural branches; fixed-MPS Graph has useful conditional signal. Family conditioning lacks resolved runtime gain and frozen selectors fail the quality gate on 64/140 assessed hashes. Pasqal coefficients remain unpromoted, so its 18 cells support descriptive quality/time findings rather than validated predictive accuracy.

These are declared implementations and adaptations, with source-independent runtime labels retained. Quantum simulator endpoint/API/noise/precision/shots/quality/workspace/plan form the target context. Same QASM or same seconds unit alone is insufficient for cross-engine ranking.

## Secondary findings and limits

QNN/Random catastrophic overprediction, backend input collisions, subset-selected R², TN estimate bias and preprocessing compatibility are secondary research findings supporting robustness/applicability. They need no successful universal transfer result to be useful. QCEC zero-input checks qualify recipes; they do not recover historical logical instances, mapping or compilation history.

## Blocking gaps

None for the central claims stated above. There are gaps for stronger claims (all-paper-exact reproduction, full competitive deployment frontier, hardware/source portability, scheduler utility), which remain explicitly unclaimed. Publication rights/metadata/clean-release verification are independent delivery gates, not a reason to reopen training.
