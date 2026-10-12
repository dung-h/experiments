> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Diễn giải khoa học cuối cùng

READY WITH LIMITATIONS. Final Scientific Freeze v2 giữ nguyên. Cohort gồm
340 Ma–Li, 230 Qonductor và 3.945 QPack observations, trong 165 conservative
structural groups. Target là archived observed execution/service time theo
boundary của từng nguồn, không phải một pulse clock đồng nhất. QPack chiếm
87,4% số hàng nhưng chỉ có sáu cấu trúc. Input có các mức exact logical và
reconstruction-qualified; không gọi toàn bộ là 4.515 historical logical
instances đã phục hồi chính xác.

Các mục dưới là reporting questions. [Crosswalk](README.md) nối chúng với
bốn RQ của Research Contract v1.1. [Registry](estimator_fidelity_registry.md)
và [claim matrix](evidence_to_claims.md) quy định cách gọi method và giới hạn.

## RQ1 — Predictive accuracy và baseline skill

**Question.** Pipeline nào dự đoán observed service time hiệu quả, hơn reference nào?

**Evidence.** Grouped five-outer/four-inner evaluation dùng chung test IDs.
Graph/MLP dùng protocol-defined median-of-three predictions, không chọn seed
theo test. Bảng 2 giữ toàn bộ 34 metric rows cùng tail errors. Graph có MAE
0.83413333 s và R² 0.46809088; MLP
0.89342298 s/0.36315832; selector
0.88665633 s/0.41130738 [C01–C04].

**Finding.** Skill = 1 − MAE_method/MAE_reference. Graph đạt
26.7195604% so với global train median và 5.9237157% so với inner-selected
classical regressor trên cùng 4.515 hàng. Graph−median có paired 95% CI
[−0.72887, −0.04188] s. Graph−selector có CI [−0.31786, +0.45791] s;
Graph−MLP có CI [−0.32907, +0.48457] s. Hai đối chiếu sau chưa resolve superiority.

**Interpretation.** Graph khai thác được signal vượt global constant trong study
này. Lợi thế giảm trước comparator mạnh hơn. Equal-source MAE là Graph 0.89803,
MLP 0.97602 và selector 0.87898 s. Equal-group MAE đổi thứ tự Graph/MLP:
0.63324/0.58070 s. Per-source scores là slices của unified fits, không phải
các model được train riêng từng nguồn.

**Limitation.** Source median biết provenance, nên không phải equal-information
reference. Graph nhận node/calibration và có khoảng 842k parameters; MLP
khoảng 369k, chỉ dùng globals. Parent neural fit inner folds 1–3, chọn epoch
trên inner 0 và không full-outer refit. Regression dùng bốn inner validations
rồi full-outer refit. Same test rows không đồng nghĩa same information/budget.
Bootstrap conditional theo groups và fitted predictions, không đo training
variability hoặc future prediction coverage.

**Thesis takeaway.** Trình bày point advantage kèm comparator, weighting và CI.
Không biến pooled ranking thành kết luận neural superiority phổ quát.

## RQ2 — Estimator fidelity và target alignment

**Question.** Những thành phần nào được dùng nguyên bản, adapted hay proxied?
Outputs có cùng ý nghĩa với target đánh giá không?

**Evidence.** Registry và Bảng 1/index bao phủ 34 Track A metric rows, 49 Track B
rows và native/analog companions [C09–C13, C20]. QCRE gọi pinned original
timing component. Qiskit gọi native API. Original Scholten đòi hỏi kernel/QV
ensemble effective depth; verified eligibility là 0/4.515.

**Finding.** Graph/MLP là architecture/core adaptations. Qonductor giữ counter
gốc gọi là "swap", thực tế đếm CX/CZ/ECR, năm statistics và sáu regressor
families; search được giới hạn ở 307 candidates. Nominal snapshot khớp tên
backend không chứng minh calibration đúng ngày đo. QCRE unitary critical path
bỏ terminal measurement; Qiskit schedule có semantics riêng. Hyb chia cost
cho survival probability, nên biểu diễn useful-work/resource cost chứ không
tự động là thời gian thực thi một lần.

**Interpretation.** Accuracy tốt không xóa semantic mismatch. Raw, shot-scaled,
affine và log-affine service bridges là các estimators khác nhau. QCRE/Qiskit
shot-affine MAE 1.05127/1.04970 s không thắng source-median point reference
0.97436 s trên matched rows. Scholten generalized proxy có raw MAE 610.288,
affine 0.74835 và log-affine 0.72660 s trên 3.399 rows. Eligible global median
là 0.82007 s, source median 0.69207 s. Proxy cải thiện reference thứ nhất,
không reference thứ hai. Hyb finite-raw coverage 4.175 và log-cost coverage
4.367 phải được trình bày riêng; undefined/overflow không thay bằng 0.

**Limitation.** Original Scholten accuracy chưa được đánh giá. Native component
không đồng nghĩa full-paper workflow reproduction. Calibration học statistical
bridge từ labels, không đo được physical overhead. QCEC chứng minh zero-input
prepared-state agreement và measured-output agreement cho 32 pairs theo
declared mappings. Full-unitary chỉ có tám proofs và 24 counterexamples.
Đây không phải exact historical logical recovery.

**Thesis takeaway.** Nêu method identity, fidelity và clock trước khi so error.
Đổi dataset không tự làm method kém faithful; thay representation/procedure
phải được khai báo cụ thể.

## RQ3 — Feature và structural contribution

**Question.** Evidence hỗ trợ signal, reliance và incremental value của input nào?

**Evidence.** Ledger 51 raw fields, 257 saved transforms và historical controls
trong Bảng 4 [C05–C08, C31]. Raw schema gồm 44 gate counters, hai size và
năm structural summaries. Refinement dựa trên positive-sum support, không lấy
41 vị trí đầu tiên.

**Finding.** Original simulator corpus có 41 supported positions. Released
340 real rows được recompute có 40; mọi current parent fit cũng giữ 40.
Ngoài mười zero simulator fields, rccx không có support trong cohort hiện tại.
RQ4 final tensor widths là 11/22/40 tùy fit population. Một positive constant
có thể được giữ trong tensor nhưng bị variable mask zero. Author executed real
tensor còn chưa xác minh: notebook example 12, constructor default 41 và
released-corpus support 40 không phải cùng một bằng chứng.

**Interpretation.** Historical fixed-model no-edge perturbation làm MAE tăng
0.4783→5.9652 s, cho thấy reliance. Retrained coarse node-only có MAE 0.81454 s
so với true DAG 0.84483 s; delta CI [−0.0165, +0.0762] chưa resolve topology
gain. Retrained static/timing blocks có predictive signal; critical-subgraph
additions không cho stable MAE benefit. Không có complete T0–T3 ladder trên
4.515 rows. Paper component scores là external evidence, không local replication.

**Limitation.** Graph–MLP khác information và capacity. Intervention sensitivity
không phải retrained benefit. Historical 340-row/coarse controls không thay
thế native full-4.515 topology ablation. Random test-only fields và QNN range
shifts là descriptive diagnostics, chưa cô lập nguyên nhân thất bại.

**Thesis takeaway.** Bảo vệ useful compiled/timing information và verified
schema. Không claim isolated DAG topology là nguyên nhân thắng hoặc ép width 41.

## RQ4 — Generalization và robustness

**Question.** New family, backend-context và size được kiểm chứng đến đâu?

**Evidence.** Full grid gồm 21 CPU cases, 21 neural cases và 126 cells:
chín family, mười backend-context, hai QPack size holdouts [C14–C19].
Bảng 3 giữ per-case MAE, MedAE, RMSE, R², residual/tail errors, reference,
paired CI và support. Có 6.684 test assignments phủ 4.515 unique IDs.
Train/test observation IDs disjoint trong từng case.

**Finding.** QNN có 26 rows/14 groups: Graph/MLP MAE 36.272/50.764 s,
train-only median 2.658 s. Random có 30/16: MAE 145.708/188.815 s,
median 3.627 s. Cả hai models overpredict mọi test row, R² âm rất lớn.
Đây không phải một outlier. Kyoto có 148/148 raw và transformed MLP inputs
trùng fit vectors nhưng không trùng observation IDs. Osaka có 149/192 collisions
và 43 new-group rows. Quito có 1.051/1.051 collisions, gồm 1.009 QPack
và 42 Qonductor observations trong bảy groups. Complete Graph inputs không
exactly collide do nominal node context khác.

**Interpretation.** Family holdouts có structural embargo và 14–27 groups,
khác backend transfer cố ý cho repeated structures. MLP có MAE thấp hơn
Graph ở 9/10 backend cases không chứng minh đã học hardware adaptation.
QPack size probes có 1.321 rows/3 groups và 498/2. Case sau có Graph/MLP
R² −4.685/−13.060; cả hai underpredict mọi row. QNN giữ 40 globals;
Random giữ 22 và masks 18 test-only fields. Extreme scaled counts mô tả
support shift, không causal attribution.

**Limitation.** Chỉ Kyoto/Osaka có ít nhất mười groups trong backend cases để
tính CI theo frozen rule. Source, workload, clocks và shots confound generation.
Không pool cross-case errors. Không có systematic real-QPU shot-held-out hoặc
source-held-out neural study. Source-stratified scores/reweighting không phải
source-OOD evaluation. Twolocalrandom vẫn được ghi unauthorized-at-execution,
later independently admitted; không viết lại authorization history.

**Thesis takeaway.** Full grid trả lời bounded transfer questions và phát hiện
catastrophic failures. Không bảo đảm universal generalization hoặc isolated
hardware-generation effect.

## RQ5 — Reliability, scaling và operational limitations

Mục này là reporting dimension bổ sung, không phải RQ thứ năm của frozen contract.

**Question.** Runtime predictions đã đủ an toàn cho scheduler, quality và scale chưa?

**Evidence.** Saved coverage/tails, conditional CI, MPS quality gates,
TN/Pasqal và width probes [C18–C20, C27–C30, C32]. Chưa có prospective
calibrated prediction-interval study.

**Finding.** Runtime completion không bảo đảm required state quality. Fixed-MPS
có 147/150 quality passes và ba failures. Joint selectors fail fidelity ở
64/140 assessed choices; mười assigned hashes chưa assessed. Trong 140 hashes
đã đo, 138 có passing rung. Hyb finite-cost và nominal-throughput missingness
có coverage riêng. Width-dependent API zero-work cost không phải isolated
physical launch latency.

**Interpretation.** Operational reporting phải giữ riêng runtime, quality,
failure và coverage. Negative safety finding là kết quả đã được evaluated,
không INCONCLUSIVE chỉ vì model thất bại. Chưa có universal entanglement,
shot hoặc hardware scaling law.

**Limitation.** Conditional error bootstrap không phải 95% future-job guarantee.
TN internals opaque; Pasqal coefficients chưa identifiable. Limited width,
bond và structural support giới hạn extrapolation.

**Thesis takeaway.** Benchmark chỉ ra nơi estimator có thể fail dù average
error tốt. Chưa thiết kế universal production estimator hoặc chứng minh safe scheduling.

## Track B — Simulator runtime prediction theo context

**Question.** Predictor nào có giá trị trong cùng engine/task/context?

**Evidence.** Bảng 5 giữ 49 predictive metric rows trong bảy contexts và native
TN/analog diagnostics, frozen shared-hash/rung protocols cùng paired uncertainty
[C21–C30].

**Finding.** Dense FP32/FP64 warm sample: RF MAE 0.821/0.813 ms so với
median 1.749/1.753 ms và Graph 1.716/1.730 ms. Fixed-chi16 FP64 warm
get_state: Graph 0.29609 s so với Ridge 1.27567 s; paired CI
[−1.88582, −0.33726] s. Aer FakeSherbrooke Opt1: source/hybrid/transpiled
GNN MAE 0.44966/0.40636/0.30076 s; compiled XGBoost 0.29704 s.
Source→transpiled gain được resolve; hybrid→transpiled và GNN→XGBoost
chưa resolve. Family-conditioned/agnostic MAE 0.31163/0.30752 s, family
holdout 0.31101/0.30879 s không có resolved runtime gain.
Maestro-style QCSim component có MAE 0.10370 ms so với Ridge 0.13361 ms
trên 143 supported hashes, saved paired CI âm. Native TN có 603/612 plan
rows với median EST/warm 3.734 cho core và 4.132 cho frontier. Pasqal có
18 cells/17 quality passes nhưng
chưa có identified/promoted predictor.

**Interpretation.** Graph có useful signal cho fixed MPS nhưng không thắng
dense RF theo MAE. Compiled views giúp Aer mà chưa chứng minh cần GNN.
Component calibration có local value trong QCSim. Joint quality selection
là endpoint riêng với runtime accuracy.

**Limitation.** Sample, state, Aer execution, engine-reported time, scalar
contraction và analog TDVP khác clocks; cùng đơn vị seconds không đủ để pool.
Ladder có 840 rungs từ 140 hashes, không 840 independent circuits.
Aer chỉ q2–9, một backend/level; không full Azizov replication. Closed
Composer/engine selection chưa reproduced. Không có universal paper winner.

**Thesis takeaway.** Hai domain dùng chung nguyên tắc benchmark nhưng không
một leaderboard. Report per-context errors, matched controls, quality,
coverage và fidelity.

## Kết luận thesis

Benchmark cung cấp evidence có thể truy vết về năng lực và giới hạn của
runtime-estimation pipelines trên qualified multi-source QPU data và local
simulator contexts. Lợi thế dự đoán phụ thuộc comparator, representation,
nguồn và clock. Strong simple baselines, negative OOD results và quality-safety
tests làm rõ giới hạn ứng dụng. Không cần experiment mới để bảo vệ các kết
luận có điều kiện này. Các open questions ở Bảng 6 cần amendment riêng nếu
muốn đưa ra stronger claims.
