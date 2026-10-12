> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Claims, đóng góp và giới hạn

## Phân loại phát biểu

FACT là số liệu hoặc sự kiện được source-linked. INTERPRETATION là suy luận
có phạm vi. PROJECT DECISION là lựa chọn đã duyệt, không phải finding thực nghiệm.
OPEN QUESTION là điều evidence chưa trả lời. Claim registry dùng bốn strength
labels; negative outcome không mặc định INCONCLUSIVE.

| Loại | Ví dụ hợp lệ |
| --- | --- |
| FACT | C14/C15: mọi QNN/Random test row bị overpredict; không lọc outlier |
| INTERPRETATION | Absolute errors không hữu ích dù Graph có MAE thấp hơn MLP |
| PROJECT DECISION | Giữ frozen results và nhãn adaptations; không thêm experiment |
| OPEN QUESTION | Topology riêng lẻ có retrained gain trên full 4.515 không? |

## Đóng góp thesis có thể bảo vệ

- Qualified multi-source panel gồm 340 Ma–Li, 230 Qonductor và 3.945 QPack
  observations trong 165 groups, có filter/reconstruction/provenance và
  missingness policies truy ngược được. Không gọi toàn bộ là exact historical
  logical circuits.
- Shared-test pipeline benchmark cho learned, analytical, calibrated và simple
  controls, cùng pooled/source/group weighting và conditional paired uncertainty.
- Local simulator comparisons theo clock/configuration, có quality gates và
  failures, không ghép các APIs thành leaderboard.
- Full frozen family/backend/size grid, phát hiện systematic overprediction và
  representation collisions. Negative findings là đóng góp, không che giấu.
- Feature-fidelity trace từ raw 51 đến supported 41/40 và fit-dependent RQ4
  widths; tách source audit khỏi author executed tensor chưa được xác minh.

Đây là empirical/methodological benchmark contributions. Không tuyên bố
thuật toán runtime estimator mới, phát hiện internal NVIDIA formula hoặc giải
quyết universal hardware generalization. QCRE/Tremba là một source family,
không hai kết quả độc lập.

## Threats to validity

**Construct validity.** Provider/source clocks có service components khác nhau.
Raw schedule, cost và throughput proxy không phải service identity. Nominal
snapshot không chứng minh historical calibration/routing. Shot multiplication
có assumptions chưa được kiểm chứng bằng shot-held-out study.

**Input validity.** Qonductor recipe qualification dựa trên 32 pairs và
zero-input mappings, không historical recovery. QPack representative angles
không phải historical optimizer angles. Missing features/costs/failures không
được thay bằng 0 để tăng coverage. Graph nominal wire-index calibration không
phục hồi physical placement.

**Internal validity.** Graph/MLP khác node/context/capacity. Regression dùng
compiled statistics, search và refit policy khác. Same outer test không bảo đảm
causal attribution. Historical controls không thay full-4.515 controls; paper
scores không phải local replication. Upstream global normalization và local
fit-only per-feature moments khác nhau.

**Statistical validity.** QPack có 3.945 rows nhưng sáu structures. RQ4 cases
overlap; không pool 6.684 assignments thành IID data hoặc average R².
Bootstrap conditional theo fits, không training variability, simultaneous
multiple-comparison guarantee hoặc prospective coverage. Không CI dưới mười
groups. Median-of-three tính per row trước scoring; không average seed metrics
hay chọn seed tốt nhất theo test.

**External validity.** Local hardware/software/engine contexts, small-width
panels, Aer một backend/level, low-support size probes, unseen shots và causal
hardware-generation effects hạn chế extrapolation. Backend-context cases có thể
chủ yếu là known structures. Source slices không phải source holdout.

**Provenance.** Twolocalrandom unauthorized-at-execution, later audited/admitted
vẫn giữ lịch sử. Saved-evidence replay không phải clean-checkout full-QPU refit.
Reduced public payload thiếu private QPU row materials; local hash/path không
đồng nghĩa distribution permission.

## Các câu không được dùng

- “Graph/GNN thắng mọi baseline hoặc mọi backend.” C02/C04 uncertainty và C03
  weighting reversal không hỗ trợ.
- “Topology là nguyên nhân thắng.” Whole-pipeline contrast không isolated control.
- “Tác giả dùng 40 global features cho real model.” Executed real tensor chưa verified.
- “51 actual current inputs” hoặc “fixed 41.” Raw/support/tensor/varying widths khác nhau.
- “Original Scholten đã được benchmark faithful.” Current route là generalized proxy.
- “QCRE/Qiskit raw seconds là observed execution seconds.” Clock mismatch.
- “Affine intercept đo được launch overhead.” Statistical bridge không physical decomposition.
- “QNN/Random là baseline predictors.” Đó là held-out circuit families.
- “Kyoto MLP học hardware adaptation trên unseen circuits.” Inputs 148/148 collide.
- “Có thể bỏ quality failures vì runtime predictor đúng.” Safety là endpoint bắt buộc giữ.
- “230 independent QCEC proofs” hoặc “exact historical recovery.” Chỉ 32 pairs/zero-input scope.
- “21-case completion chứng minh universal generalization.” Support/confounding vẫn còn.

## Kết luận có thể đưa vào thesis

Trong benchmark đã khóa, runtime-estimation pipelines khai thác được signal
hữu ích ở một số contexts nhưng hiệu quả phụ thuộc baseline, nguồn,
representation và timing boundary. Graph có pooled advantage so với global
median, chưa có resolved superiority trước stronger classical/global controls.
Compiled information hỗ trợ local noisy-Aer prediction, còn quality-constrained
MPS selection và family-OOD neural prediction bộc lộ failures đáng kể.
Việc giữ reconstruction qualification, source/group support, coverage và
negative results cho phép đưa ra các kết luận giới hạn nhưng có thể bảo vệ.

## Việc tiếp theo

Viết Results/Discussion từ C01–C32 và sáu bảng. Mở đầu bằng data qualification
và timing semantics, sau đó accuracy/controls, features, transfer và safety.
Đưa open questions vào Future Work, không mở experiment queue. Metadata,
ownership và exact release payload là gates riêng. Không đổi frozen contract
để khớp reporting headings.
