> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# RQ2 — Generalization

## Question

Estimators hoạt động thế nào ngoài training/calibration distribution?

## Methodology

Family/workload holdouts, backend-context holdouts và QPack size holdouts giữ riêng. Full grid có 9 family+10 backend+2 size cases,21 CPU/21 neural cases,126 model×seed cells. Test/train observation IDs disjoint trong từng case. Backend transfer cố ý cho phép đã thấy circuit structure; family embargo không cùng nghĩa. Các cases có thể dùng lại observations:6684 test assignments phủ 4515 unique IDs, không pool như 6684 independent samples. Generalization runner chọn epoch internally rồi fresh full outer-train refit; degradation versus parent cũng thay training procedure.

## Evidence and result

Table C giữ 42 neural method rows, support, train-median/CPU selector và paired uncertainty. QNN 26 observations/14 groups và Random 30/16 groups có severe overprediction: Graph/MLP MAE 36.272171/50.763748 s và 145.708260/188.814714 s. Train-median MAE 2.658/3.627 s. Không gọi relative Graph win là useful OOD prediction.

Kyoto 148/148 MLP fit-vector collisions, Osaka 149/192 và Quito 1051/1051 không phải observation-ID overlap. Kyoto là known-circuit context transfer; Osaka có 43 new-group rows; Quito chứa 1009 QPack+42 Qonductor rows trong 7 groups. Graph full inputs có nominal T1/T2 nên không cùng full-input collision. Context effect vẫn confounded bởi source/workload/shots và calibration assumptions; không isolate hardware generation.

Track B có grouped-hash OOF, separate family-held-out MPS ladder và historical width study. Trained FP32/FP64 riêng không phải precision transfer. Descriptive source-balanced parent slices không phải source-held-out training experiment.

## Interpretation: ba loại evidence

1. **Evaluated failure:** QNN/Random ngoài supported feature/range distribution.
2. **Evaluated but confounded transfer:** backend-context với repeated structures hoặc source/workload changes; low-support QPack size probes.
3. **Unevaluated transfer:** controlled source/shot shift, isolated generation/topology shift và broad current simulator-engine/hardware portability.

## Limitation

Small group counts giới hạn uncertainty; không tự tạo CI khi frozen rule không cho. Per-case performance không được average R² thành global score. Historical Twolocalrandom unauthorized-at-execution disclosure vẫn giữ dù later admission.

## Thesis takeaway

RQ2 **PARTIALLY ANSWERED, NON-BLOCKING** cho hai tracks. Evaluated failures và context limitations đã trả lời phần có evidence; universal portability không cần để bảo vệ central benchmark claim.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); [generalization](../../results/interpretation/thesis/tables/generalization_robustness.csv); [grid](../../results/generalization/report.md); [sim](../../results/interpretation/thesis/tables/track_b_findings.csv); [claims](../../results/research_questions/closure/final_claim_matrix.csv).
