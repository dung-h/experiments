> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# Reproducibility findings

## Ba mức equivalence

**Semantic circuit equivalence** xét mathematical operation hoặc declared input/output distributions. **Representation equivalence** xét graph/feature structure dưới mapping. **Frozen-model input equivalence** yêu cầu consumed tensors, order, values, masks và transforms đúng với fitted checkpoint. Hai mức đầu không tự bảo đảm mức cuối.

## Feature support

Raw schema 51 không đồng nghĩa executed width 51. Support refinement tạo 41 cho original simulator corpus và 40 cho current parent QPU. Generalization fits có width 11/22/40 theo train population; một positive constant vẫn có thể tồn tại trong tensor nhưng variable mask zero. Matching feature count không chứng minh author processed tensor được phục hồi.

## Operation order

QPU raw parser instruction sequence/wires khớp 45/45 giữa environments.38 Ma–Li cases diverge ở model-consumed order qua RemoveBarriers DAG-to-circuit reconstruction. Terra 0.25.2 và Qiskit 2.5.2 khác deterministic topological tie ordering; node_index 177 làm order visible cho model.45 named dependency graphs equivalently relabeled không đủ để gọi ordered tensors compatible. RemoveFinalMeasurements giữ version-dependent order; no retrospectively selected per-row permutation.

## RCCX compatibility

Historical/library RCCX definitions U2/U1/CX và H/T/TDG/CX khác representation trước lowering. Adapter không được silently accept unvalidated basis để ép inference PASS. Declared historical dispatch/environment và frozen transforms khôi phục exact consumed arrays. Fix dựa vào independent frozen inputs, không target/prediction performance.

## Verified scope

Phase 3 phục hồi 210/210 independently encoded model-ready inputs:90 QPU method inputs (45×Graph/MLP) và 120 Dense inputs (30× 2 methods× 2 precision contexts). Frozen NPZ bytes không được overwrite; transformed arrays được so exact. Numeric raw-global tolerance 1e−12 chỉ cho decimal CSV storage. No model forward được dùng làm equivalence test.

## Implication

Environment-aware adapter có thể cần để replay một frozen estimator, ngay cả khi quantum circuit semantics không đổi. Original Dense producing package versions và author real tensor vẫn thiếu metadata; demonstrated compatibility không claim exact historical environment. Đây là reproducibility lesson, không causal proof topology làm accuracy tốt hơn.

## Provenance erratum

Final closure đã resolve 6 README checks bằng correct package-relative namespace và 182 named declarations (180 SHA 256+2 Git revisions). Không chỉnh historical hashes hay raw evidence. Generic validator PASS là engineering evidence, không scientific sufficiency tự động.


Nguồn: [feature](../../results/interpretation/benchmark/mali_feature_schema_audit.md); [recovery](../../results/reproducibility/preprocessing/root_cause_analysis.md); [replay](../../results/reproducibility/preprocessing/validation_report.md); [provenance](../../results/research_questions/closure/provenance_resolution.md).
