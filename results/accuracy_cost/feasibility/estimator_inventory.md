> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Executability inventory

| Track | READY | CONDITIONAL | BLOCKED | NOT APPLICABLE | Total |
|---|---:|---:|---:|---:|---:|
| A | 26 | 0 | 9 | 0 | 35 |
| B | 13 | 1 | 36 | 1 | 51 |

Exact row-level paths, required fields và eligible accuracy denominators nằm
trong [CSV](estimator_inventory.csv). Fidelity card/class được chép từ saved
method-context index, không nâng adaptation/proxy thành reproduction. Các path
có `{0..4}`/seed braces là
**notation**, không phải file literal; expand đúng năm folds/ba seeds rồi kiểm
source pins. Full absolute file/hash list ở manifest (historical source; not bundled).

## READY: assets sufficient by static inspection

- QPU: `mali_logical_graph`, `mali_global_mlp`; ba `outer_train_*` references;
  raw/affine/log-affine của bảy bases: `qcre_source_estimator`,
  `qcre_source_estimator_shot_scaled`, `qiskit_estimate_duration_snapshot`,
  `qiskit_shot_scaled_schedule_seconds`, `hyb_nominal_r2_effective_cost_v1`,
  `hyb_nominal_r2_shot_effective_cost_v1`, `scholten_unified_nominal_throughput`.
- Dense FP32 và FP64: `mali_style_dense_graph`, `matched_metadata_mlp`,
  `outer_train_median` trong từng context.
- Fixed MPS chi16: chỉ `train_fold_median`.
- Joint MPS: `family_conditioned`, `family_agnostic_ablation` trong `c44` và
  `family_holdout`; conditioned includes the saved family classifiers.
- QCSim: `maestro_style_cpu_sv_component_optimizer_off_adaptation` và
  `outer_train_median`. Đây là saved independent synthetic calibration knots,
  không fit thêm trên evaluation targets. QCSim không cần chạy để predict.

Parent QPU cần **best_model_state**, không final `model_state`. Dense cần final
500-epoch `model_state`; joint cần saved `model` sau 160 epochs. Không load
optimizer hoặc resume epoch. Native precision của simulator target không xác
định precision của neural estimator (frozen heads vẫn FP32).

## BLOCKED: không có fitted model

Track A: sáu Qonductor families + budgeted selector (7); Hyb log-cost Ridge (1);
original Scholten (1, thiếu kernel/effective-depth eligibility). QPU regressors
chỉ ghi selections/scores/predictions; Hyb Ridge receipts không có coef/scaler.

Track B: 6 dense learned classical routes × 2 precision = 12; fixed-MPS Ridge,
Graph, family residual/agnostic = 4; Azizov 3 GNN + 15 regression-view routes =
18; Maestro nested Ridge/source-DAG Graph = 2. Tổng 36. Authoritative roots đã
kiểm file extensions và producing serialization code. Không tìm thấy model
khác đủ identity để thay vào. Không kết luận rằng upstream method không thể
deploy; đây là **artifact retention limitation của benchmark hiện tại**.

Không đủ coefficients để “tái tạo” fitted Ridge từ alpha; không thể khôi phục
trees/support vectors từ predictions. Không dùng estimator cùng tên ở cohort,
split, engine, chi hay historical run khác. Nếu owner chỉ ra serialization
đúng frozen identities, được audit read-only rồi xin duyệt mở coverage; không
âm thầm refit. Các missing serialization không làm mất giá trị saved accuracy.

## CONDITIONAL và NOT APPLICABLE

cuTensorNet có native pipeline, không learned inference. Read attribute sau
planning khác toàn bộ chi phí tạo estimate. Historical selected paths không
được serialize trong producer; seed/optimizer settings không chứng minh path
identity. Không ghép fresh-plan latency với historical same-plan accuracy như
một paired point. Native extension chỉ sau explicit approval, đúng libraries,
workspace và plan-trace policy; tuyệt đối không execute contraction.

Pasqal có 18 analog diagnostic cells, không promoted predictor hay identified
resource coefficients; không time MPSBackend.run rồi gọi là estimator overhead.

## Additional frozen evidence, không nhân đôi primary count

21 generalization cases/126 Graph–MLP seed cells là case-specific refits của
hai architecture families, không 126 methods. Chúng không được gộp với parent
OOF models hoặc dùng làm parent timing. Source routing/identity authority là
cell lineage (historical source; not bundled);
full-grid manifest (historical source; not bundled).
AM1 initial proposal không time các deployment instances này; nếu bổ sung cần
inventory riêng theo case, hash/parity gates và cùng paired support. Historical
PyTorch width/RX launch studies đo simulator work, không là current estimators.

## Integrity scope

Verified: 30 QPU checkpoints + 60 dense + 60 joint runtime heads + 30 joint
family classifiers; 25 transform files; associated saved predictions/receipts.
Source pins preserve measured and adapted method semantics. File hashes chỉ
chứng minh byte identity, không prediction parity; parity là gate của amendment
tương lai. Model internals chưa deserialize. Scalars được đối soát single-valued
theo fold hoặc fold/source từ saved predictions và ghi nguyên giá trị trong
manifest, không tính lại train statistic/fit model ở bước này.
