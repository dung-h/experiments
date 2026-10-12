> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# RQ3 — Error behavior, robustness and failure modes

## Question

Estimators thất bại ở đâu, theo kiểu nào, và failures ảnh hưởng selection ra sao?

## Methodology

Giữ MAE/MedAE/RMSE/R², signed residual, tails, coverage, support và train-only controls. Không bỏ outliers, chọn seeds/metrics hoặc điều chỉnh quality threshold theo test performance. Table F liên kết failure class với evidence; Table B/C giữ denominators và distributions.

## Evidence and result

**Prediction failure:** QNN/Random overpredict mọi test observation ở cả neural pipelines, có extremely negative R². Đây là systematic error, không chỉ một outlier. Point winner tương đối không đủ practical usefulness.

**Representation failure/limitation:** Kyoto raw và post-transform MLP vectors 148/148 đã có trong fit, nhưng observation IDs không trùng. Global branch không có backend/calibration context; Graph có nominal node T1/T2. Collision khác label leakage và không chứng minh learned hardware adaptation.

**Aggregation and selection sensitivity:** Source-balanced/equal-group estimands đổi ranking. Dense selected 30 R² cao hơn full 150; subset qualification không được biến thành full-context performance. Median-reference MedAE tốt hơn neural trên Dense timed subsets, còn MAE/RMSE/R² kém hơn. Tail errors và target variance giải thích vì sao metrics không đồng thuận, không cho phép chọn metric hậu nghiệm.

**Quality-safety failure:** Trong MPS c44, mỗi frozen family-conditioned và agnostic selector vi phạm fidelity gate trên 64/140 observed choices;10/150 unassessed. Không cộng hai branches thành 128 independent failures và không áp 64/140 cho family-holdout variant khác. Chạy simulator xong hoặc dự đoán runtime tốt không đảm bảo state đủ fidelity.

**Context/cost failure:** TN EST/warm ratio có bias trên successful same-plan slices; Pasqal 17/18 quality passes không xác nhận analytical coefficients đã identifiable. OOM, timeout hoặc unsupported input là coverage/resource/applicability outcomes, không tự là model prediction error.

## Interpretation

Negative findings là kết quả của evaluation có protocol hợp lệ. Cần phân biệt model extrapolation, missing-context collision, unavailable engine support và preprocessing incompatibility. Không quy tất cả poor scores cho GNN yếu hay implementation sai.

## Limitation

Range/support shifts chưa là causal explanation. Không có prospective uncertainty/safety guarantee hoặc complete feature intervention study.

## Thesis takeaway

RQ3 **ANSWERED WITH QUALIFICATIONS**. Report failures cùng practical usefulness; không chỉ báo ai có MAE thấp hơn.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); [generalization](../../results/interpretation/thesis/tables/generalization_robustness.csv); [grid](../../results/generalization/report.md); [qpu](../../results/analysis/qpu_metrics.csv); [mps](../../results/analysis/mps_selection_safety.csv); [tn](../../results/analysis/tn_same_plan_slices.csv); [pasqal](../../results/analysis/pasqal_stage_quality.csv); p5 (historical source; not bundled).
