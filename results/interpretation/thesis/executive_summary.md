> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Tóm tắt khoa học

**Verdict: READY WITH LIMITATIONS.** Benchmark hoàn tất trong Final Scientific
Freeze v2. Không thêm thí nghiệm, thay dữ liệu hoặc sửa kết quả. Các findings
dưới đây được xếp theo ý nghĩa khoa học, không theo độ phức tạp của model.

1. **FACT [C01–C04].** Trên 4.515 observations, Graph/MLP/selector có MAE
0.83413/0.89342/0.88666 s. Graph cải thiện 26,7% so với global train median,
nhưng 5,9% so với selector. Paired CI của Graph−selector và Graph−MLP chứa 0.
**INTERPRETATION:** có signal vượt constant reference, chưa resolve superiority
trước stronger comparators.

2. **FACT [C03].** Equal-source MAE selector 0.87898 s thấp hơn Graph
0.89803 s. Equal-group weighting đảo thứ tự Graph/MLP. QPack chiếm 87,4%
rows nhưng chỉ sáu structures. **INTERPRETATION:** ranking phụ thuộc estimand
và source distribution.

3. **FACT [C14–C18].** Full RQ4 có 21 CPU/21 neural cases, 126 cells.
QNN/Random là held-out families, không baseline algorithms. Graph/MLP MAE
36.272/50.764 s và 145.708/188.815 s; cả hai overpredict mọi row, kém
train-only median nhiều lần. Kyoto 148/148 MLP inputs trùng fit vectors,
không trùng IDs. **INTERPRETATION:** catastrophic OOD failures và context-blind
collisions giới hạn practical/hardware-transfer claims.

4. **FACT [C07,C08].** Ma–Li có 51 raw positions, support-refined thành 41
trong original simulator corpus và 40 ở mọi parent fit. Có 257 transforms
đã kiểm tra; RQ4 widths tùy fit population. **OPEN QUESTION:** notebook 12,
default 41 và released support 40 chưa xác định author executed real tensor.
**PROJECT DECISION:** giữ adaptation, không hard-code 41 hoặc retrain chỉ để
khớp dimension label.

5. **FACT [C05,C06,C31].** Fixed-model perturbations cho thấy reliance lên
edges/wire identities. Retrained coarse controls chưa resolve topology gain;
static/timing blocks có signal. **INTERPRETATION:** reliance khác retrained
increment. Graph−MLP không isolated topology ablation.

6. **FACT [C09–C13].** QCRE/Qiskit duration khác observed service. Hyb cost
điều chỉnh survival, không literal execution time. Original Scholten thiếu
verified kernel/QV inputs; current route là generalized proxy. Calibration
là predictor mới. **INTERPRETATION:** low error sau calibration không chứng
minh native clock phù hợp hoặc physical overhead đã được đo.

7. **FACT [C21–C29].** Simulator findings theo context: dense RF có MAE
thấp hơn Graph; fixed-MPS Graph tốt hơn Ridge; Aer transpiled GNN cải thiện
source GNN nhưng chưa resolve hơn XGBoost. Maestro component có local gain,
không full engine-selector reproduction. **INTERPRETATION:** không có universal
Track B winner hoặc clock chung với Track A.

8. **FACT [C27].** Joint MPS selectors fail fidelity 64/140 assessed choices,
mười unassessed, dù 138/140 hashes có passing rung. **INTERPRETATION:** runtime
accuracy hoặc quality probability score không đồng nghĩa safe configuration
selection.

9. **FACT [C20].** QCEC có 32/32 zero-input prepared-state proofs và
measured-output matches. Full-unitary có 8/32 proofs, 24 counterexamples
theo declared maps. Có 230 observations liên kết 32 pairs.
**INTERPRETATION:** hỗ trợ functional recipe qualification, không exact historical
recovery hoặc 230 independent proofs.

**Đóng góp có thể bảo vệ:** qualified multi-source data với provenance;
shared-test pipeline comparisons cùng strong simple controls; clock/fidelity/
coverage discipline; negative OOD và quality-safety findings. Không claim
thuật toán estimator mới hoặc paper-exact replication. Bootstrap hiện có
conditional theo groups/fits, không prospective prediction guarantee.

**PROJECT DECISION:** khóa experimental scope. **OPEN QUESTIONS:** individual
feature/topology attribution, unseen shots, isolated hardware-generation effects,
prospective intervals và broader engine/width transfer. Không cần experiment
bổ sung để bảo vệ bounded claims hiện tại. Ownership, rights, metadata và
exact-public-payload verification vẫn là release gates riêng.
