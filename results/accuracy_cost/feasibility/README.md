> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 feasibility: estimator overhead

**CONDITIONAL GO — planning only, 2026-10-11.** Có thể đo một subset của
estimators đã đóng băng mà không train/refit. Không đủ assets để đo toàn bộ
leaderboard. AM1 chưa được chạy; đây không phải authorization hoặc timing harness.

| Track | READY | CONDITIONAL | BLOCKED | NOT APPLICABLE | Total |
|---|---:|---:|---:|---:|---:|
| A | 26 | 0 | 9 | 0 | 35 |
| B | 13 | 1 | 36 | 1 | 51 |

Đơn vị đếm là **method × execution context/variant**, không phải số papers,
checkpoints hay independent methods: 34 QPU metric rows + original Scholten;
49 simulator metric rows + native cuTensorNet + analog companion. Raw/affine/
log-affine và FP32/FP64 được đếm riêng. READY nghĩa là assets và code có vẻ đủ
qua static inspection, không phải model đã load hay replay/inference đã PASS.
READY của analytical chỉ áp dụng trên eligibility đã lưu, không biến missing
rows thành available. 39 READY rows không tạo thành 39 independent methods.

**Phát hiện quyết định:** QPU Graph/MLP, dense neural và joint-MPS có 180
checkpoint files với hashes khớp receipts (150 runtime heads + 30 family
classifiers). Qonductor six regressors/selector, Hyb log-Ridge, dense classical,
fixed-MPS learned, Azizov và Maestro learned comparators không có fitted states
đủ trong output authoritative. Hyperparameters/predictions không thay cho model.
Không được refit để đo rồi ghép latency mới với accuracy cũ.

AM1 đáng làm để bổ sung **một phần** working RQ4 Accuracy–Cost Trade-off;
không đóng toàn bộ RQ4/scalability/production scheduling. Tên RQ4 trong freeze
cũ là generalization; ở task này dùng working RQ1–RQ5 của closure review.
Không đổi tên hay nghĩa của historical protocol.

## Đọc theo thứ tự

1. [Inventory](estimator_inventory.md) và [CSV](estimator_inventory.csv).
2. [Measurement boundaries](measurement_boundaries.md).
3. [Fair comparison](fair_comparison_protocol.md).
4. [Existing timing](existing_timing_evidence.md).
5. [Accuracy–cost analysis](accuracy_cost_analysis_plan.md).
6. [Implementation plan](implementation_plan.md), [decision](decision_memo.md).
7. [Validation](validation_report.md), pins (historical source; not bundled).

Nguồn: [closure](../../research_questions/historical/README.md);
[fidelity](../../interpretation/thesis/estimator_fidelity_registry.md);
freeze manifest (historical source; not bundled).
Không cập nhật manuscript/presentation; không commit/push.
