> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# RQ4 — Accuracy–cost trade-off and scalability

## Question

Chi phí tạo một runtime prediction là bao nhiêu, và accuracy có tương xứng với chi phí đó không?

## Methodology

Full-B2 là warm online raw-input-to-prediction path: parsing, features, Graph construction khi cần, frozen transforms, serialization/IPC, tensor/H2D, inference/inverse transform/D2H và aggregation. Không bao gồm offline training/model load/cold start. So sánh genuine per-method paths ở cùng external boundary, không ép hai methods trả cùng intermediate work. Accuracy và estimator latency là hai quantities khác nhau; không lấy MAE trừ B2.

## Evidence and result

| context | method | n | accuracy_unit | mae | rmse | r2 | median_B2_ms |
|---|---|---|---|---|---|---|---|
| QPU exact45 | graph | 45 | s | 0.724177 | 0.94754 | 0.897687 | 332.557 |
| QPU exact45 | mlp | 45 | s | 0.812521 | 1.25757 | 0.81978 | 149.514 |
| Dense FP32 exact30 | graph | 30 | ms | 1.75343 | 3.59679 | 0.678366 | 7.25009 |
| Dense FP32 exact30 | mlp | 30 | ms | 1.52333 | 3.3867 | 0.714842 | 3.68971 |
| Dense FP64 exact30 | graph | 30 | ms | 0.807979 | 1.62501 | 0.925081 | 7.0761 |
| Dense FP64 exact30 | mlp | 30 | ms | 0.618299 | 1.14641 | 0.962713 | 3.61045 |

QPU exact 45 gồm 38 Ma–Li+7 Qonductor, không QPack. Graph−MLP MAE−0.088344266 s, conditional 95% interval[−0.354865074,+0.137886216] s chứa zero. Ratio of marginal B2 medians 2.224255631,[2.067538851,2.408525542]. Median paired ratio 2.370999 là statistic khác.

Dense FP32 exact 30: Graph−MLP MAE 0.230099699 ms,[0.056416868,0.459004446]; B2 ratio 1.964950273,[1.916280158,2.013973472]. FP64 exact 30: delta 0.189680237 ms,[0.027736609,0.413599796]; ratio 1.959891936,[1.897087796,2.010438803]. MLP dominates Graph trên hai criteria này, conditional vào hai measured subsets.

Preparation shares của accumulated B2: QPU Graph 76.1086%, MLP 97.2488%; Dense FP32 Graph 67.6274%, MLP 86.1642%; FP64 Graph 62.7999%, MLP 83.0888%. Đây là sum(stage durations)/sum(B2), không percentage của marginal median. Stage medians không được cộng hoặc trừ để suy causal overhead. Parsing/globals cũng là work thật của MLP.

## Interpretation

Prediction-only inference bỏ qua substantial online preparation. Trên QPU subset, extra Graph cost không đi cùng statistically established accuracy advantage. Dense measured pair cho lower accuracy error và lower B2 ở MLP. Không suy ra mọi Graph model vô ích.

## Limitation

Exact 45≠full 4515; exact 30≠full 150. Full 150 Dense neural R² khoảng 0.13–0.14, không dùng high subset R² làm full performance. Historical simulator execution cross-environment, không matched operational break-even. B1/B2 khác occasions/instrumentation, không ratio matched slowdown. Strong classical fitted-state overhead thiếu; chưa complete Pareto frontier, cold deployment, cross-hardware scaling hoặc scheduler gain.

## Thesis takeaway

RQ4 **ANSWERED WITHIN MEASURED SCOPE** ở cả tracks, mã hóa `ANSWERED WITH QUALIFICATIONS` theo closure vocabulary. Bounded accuracy–cost finding đã đủ, không yêu cầu universal scalability.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); p4 (historical source; not bundled); p5 (historical source; not bundled); [dense_full](../../results/accuracy_cost/dense/full_context_accuracy.csv); [verdict](../../results/research_questions/closure/experiment_completion_decision.md).
