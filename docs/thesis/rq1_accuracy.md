> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# RQ1 — Predictive accuracy and comparative effectiveness

## Question

Các estimators dự đoán runtime chính xác đến đâu so với simple, analytical và learned baselines?

## Methodology

Track A dùng chung 4515 observations và five outer/four inner grouped partitions. Fits/transforms/tuning/calibration không dùng outer-test labels. Graph/MLP dùng protocol-defined per-row median-of-three, không chọn best seed. Parent neural fits chọn epoch theo inner 0, fit inner 1–3; regressors dùng inner selection rồi full outer refit. Vì vậy đây là pipeline comparison, không matched-budget hoặc topology-only ablation. Track B so trên đúng context và shared hashes/rungs.

## Evidence and result

Table B ghi exact metrics và nguồn. Pooled MAE của Graph 0.834133328 s, MLP 0.893422980 s, classical selector 0.886656328 s; global train-median 1.138275552 s. Graph−MLP conditional paired 95% interval[−0.329074,+0.484567] s; Graph−selector[−0.317858,+0.457908] s. Point differences favor Graph nhưng cả hai intervals chứa zero. Source-balanced MAE là Graph 0.898029392, MLP 0.976019206, selector 0.878982346 s. Equal-group MAE đổi thứ tự Graph/MLP thành 0.633242057/0.580696886 s.

Analytical comparison giữ raw và outer-train-calibrated predictions riêng. QCRE/Qiskit schedule có signal nhưng không đồng nghĩa service seconds. Scholten proxy có 3399 eligible observations, không phải full 4515 original estimator. Calibration improvement phải so với references trên chính eligible shared rows, không với full-cohort reference.

Table D giữ context-specific results. Transpiled Aer GNN MAE 0.300764122 s tốt hơn source-view 0.449658482 s, nhưng XGBoost transpiled 0.297035322 s là comparator cạnh tranh; không có resolved GNN superiority. Dense RF có lower local MAE hơn Graph/MLP; fixed-MPS Graph MAE 0.296091793 s so Ridge 1.275667370 s. Family conditioning chưa cho stable runtime advantage. Maestro CPU component, TN native EST và Pasqal companion có clock/applicability riêng.

## Interpretation

Graph có skill so với simple constant, nhưng competitive classical baseline làm lợi thế nhỏ hơn. Source/group weighting là khác estimand, không công cụ chọn bảng đẹp. Simulator input giống nhau chưa đủ: output task và measurement boundary cũng phải khớp.

## Limitation

165 groups không tương đương 4515 independent structures. Bootstrap conditional theo fitted predictions, không bao gồm future-workload guarantee hoặc đầy đủ training variability. Different feature/capacity/refit procedures không cho causal topology claim.

## Thesis takeaway

RQ1 **ANSWERED WITH QUALIFICATIONS** cho cả tracks: conditional pipeline comparison có bằng chứng; universal winner không có.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); [qpu](../../results/analysis/qpu_metrics.csv); qpu_pairs (historical source; not bundled); [controls](../../results/analysis/analytical_controls.csv); [sim](../../results/interpretation/thesis/tables/track_b_findings.csv); [sim_pairs](../../results/analysis/simulator_paired.csv); [claims](../../results/research_questions/closure/final_claim_matrix.csv).
