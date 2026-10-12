> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Accuracy–cost interpretation

| context | comparison | shared_queries | groups | status | paired_mean_latency_delta_ms | latency_ci95 | observed_MAE_delta_seconds | MAE_delta_ci95 |
|---|---|---|---|---|---|---|---|---|
| qualified_qpu_4515 | mali_logical_graph vs mali_global_mlp | 0 | 0 | NOT_EXECUTED: raw_input_gate_failed | NA | NA | NA | NA |
| qualified_qpu_4515 | mali_global_mlp vs outer_train_median | 45 | 44 | measured native contracts; unequal information allowed | 174.7719710625 | [147.8687944372159, 198.2847762494318] | -2.051605382459868 | [-2.5071033099075164, -1.573467918481134] |
| dense_fp32 | mali_style_dense_graph vs matched_metadata_mlp | 0 | 0 | NOT_EXECUTED: raw_input_gate_failed | NA | NA | NA | NA |
| dense_fp32 | matched_metadata_mlp vs outer_train_median | 0 | 0 | NOT_EXECUTED: raw_input_gate_failed | NA | NA | NA | NA |
| dense_fp64 | mali_style_dense_graph vs matched_metadata_mlp | 0 | 0 | NOT_EXECUTED: raw_input_gate_failed | NA | NA | NA | NA |
| dense_fp64 | matched_metadata_mlp vs outer_train_median | 0 | 0 | NOT_EXECUTED: raw_input_gate_failed | NA | NA | NA | NA |


QPU MLP measured population =45 IDs (38 Ma–Li,7 Qonductor), not the full
4515 cohort and not the previous 50-ID AM1 subset. QPack5 remain unavailable
for this declared raw-input boundary; no accuracy/timing-based selection.
Mean pooled preparation fraction over the SAME 1350 B2 requests:
98.134025%. Stage median parsing=138.467523ms,
global extraction=74.877841ms,
three-seed inference/inverse/D2H=0.542660ms.
Parsing and global features materially change MLP deployment cost relative to
prediction-only; original B1 is historical corroboration, not same-run subtraction.

Comparison to native train-fold median is useful but weak: it needs no circuit
features. It is not the missing strong selector/RF/XGB. Accuracy uses original
frozen OOF predictions; replay outputs are ONLY integrity evidence. No refit.
Paired intervals use2000 fixed-seed cluster bootstrap replicates. Equal-group
mean latency and observation-weighted MAE have distinct declared weighting.
Generalization to unmeasured QPack-dominant parent population is not justified.
QPU paired reanalysis uses identical panel.csv target_seconds for all methods;
saved target text has float serialization differences of at most
1.78e-15s. Original labels/predictions untouched.

Graph full-overhead comparison has NOT been answered: its raw-input gate
failed. No result supports choosing Graph by full B2 efficiency. Original
B1 Graph–MLP conclusions and all full-cohort accuracy remain unchanged.
Dense context median reference alone is not a competitive frontier or a
measurement of dense neural B2. Saved FP32/FP64 execution timings still have
different software/time provenance; no operational break-even claim.

Full-cohort accuracy authority remains
artifacts/am1_postpilot_audit/v1/accuracy_population_metrics.csv and original
producing OOF files. In particular QPU Graph MAE0.834133s versus MLP0.893423s
on4515; dense Graph MAE0.001715940s/0.001730218s versus MLP0.001577722s/
0.001590082s on150 each. These are not latency-population scores.

## Exploratory size/complexity description

| covariate | n | spearman_with_B2_ms | interpretation |
|---|---|---|---|
| logical_width_g44 | 45 | 0.6537486381178172 | exploratory selected_sample; source/family/size confounded |
| logical_depth_g45 | 45 | 0.44062325214336806 | exploratory selected_sample; source/family/size confounded |
| saved_graph_node_count_covariate | 45 | 0.8495009721661098 | exploratory selected_sample; source/family/size confounded |
| raw_QASM_bytes | 45 | 0.6832674571805006 | exploratory selected_sample; source/family/size confounded |

| source | n | median_B2_ms | min_B2_ms | max_B2_ms |
|---|---|---|---|---|
| mali_real_qpu | 38 | 224.19284900000002 | 102.0111605 | 271.508448 |
| qonductor_single_circuit_ibm | 7 | 2.0565745 | 1.9173244999999999 | 2.2919385 |


Đây là mô tả sau đo, không confirmatory scaling law. Dùng logical width từ
g44, không dùng width physical/allocated của cột selection cũ. Source và family
cùng thay đổi với size; không suy diễn luật2^n, quan hệ nhân quả hay ngoại suy
width. Graph node count chỉ là covariate cấu trúc, không feature của MLP.
Model dispatch/fold routing, imports và model load đã làm offline. B2 có
instrumentation và CUDA stage synchronizations; không trừ timer overhead để
biến scalar access thành nanosecond deployment claim. Shared bundle setup
recorded ở worker environments, không phải standalone B3 benchmark.
