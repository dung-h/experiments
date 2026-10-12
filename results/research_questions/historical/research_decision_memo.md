> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Research decision memo

## Verdict: PARTIALLY SUFFICIENT

**Core benchmarking completeness:** đủ bảo vệ comparisons của declared
estimators trên qualified QPU cohort và matched local simulator contexts,
cùng observed negative findings. Không hoàn tất mọi paper-exact objective.

**Broader research completeness:** chưa đóng toàn bộ năm working RQ. RQ2 vẫn
partial ở source/shot/hardware/size/engine transfer. RQ4 estimator computational
overhead có material measurement gap. Không đánh đồng completed 21-case grid
của historical RQ4 với completed accuracy–cost question hiện tại.

**Publication readiness:** không đánh giá/duyệt release ở task này. Existing
metadata/rights/controlled-export gates giữ nguyên, không giải bằng experiment.

## Có thể đóng ngay

- RQ1A/B: conditional accuracy/comparator findings, không universal ranking;
  Track B giữ Pasqal predictive exception và Maestro component boundary.
- RQ3A/B: actual failures/tails/source effects/collisions/quality violations.
  QNN/Random thất bại là scientific answers, không unanswered hypotheses.
- RQ5A/B: input/clock/fidelity applicability và conditional selection rules,
  không automated cost-optimal/safe production policy.

## Chưa đóng broad question

- RQ2A/B: completed observed transfers không isolate hardware generation;
  no source/shot-transfer guarantee; current simulator context portability
  untested. G3 không cần thêm model để khắc phục ngay.
- RQ4A/B: target scaling và historical training loop duration không cung cấp
  full estimator overhead frontier. Có partial evidence, không efficiency claim.

Không mâu thuẫn với previous READY WITH LIMITATIONS: verdict trước xét
defensibility của bounded findings, verdict này xét coverage năm questions
rộng hơn. Reporting verification chỉ kiểm fidelity/consistency, không bỏ phiếu
cho research sufficiency. Không numerical contradiction mới trong frozen scores.

## Practical method-selection framework

1. Chọn exact endpoint trước: QPU service, schedule, simulator sample/state,
   engine component, scalar contraction hoặc analog run. Reject mismatched clock.
2. Kiểm input trước execution: logical/physical circuit tier, shots/native
   counters, nominalTarget/durations/errors, configured chi, hoặc selected plan.
   Measured fidelity/observed bond/target runtime không pre-execution features.
3. Không có train/calibration labels: native schedule/EST chỉ áp dụng đúng
   native target; không tự cộng physical overhead hoặc gọi service prediction.
4. Có suitable historical labels: strong constant/classical controls trước;
   compare eligible shared rows, sources/groups and uncertainty. Graph pooled
   gain nhỏ hơn trước selector và reversed khi equal-source weighting.
5. Simulator: dense RF có local accuracy; fixed-MPS Graph có signal; Aer compiled
   views useful nhưng GNN over XGB unresolved; Maestro component scope: CPU engine;
   family selectors không quality-safe, TN same-plan only, Pasqal descriptive.
6. OOD/collisions/quality unmet: công bố limitation/abstention, không choose
   method chỉ vì pooled winner. Chưa có measured cost đủ để tối ưu deploy.

## Next decision

**Giữ nguyên freeze; chỉ phân tích saved evidence thêm khi có câu hỏi cụ thể.**
30-cell loop inventory và 603-session TN phase summary đã hoàn tất trong review
này, vẫn không tạo ra missing inference measurements. Không fit/calibrate/execute
model mới. Quyết định cần chủ dự án chọn:

- Giữ freeze, nhận RQ2/RQ4 partial và đóng bounded benchmark; hoặc
- Duyệt riêng AM1 nếu full accuracy–cost là required thesis contribution.

Không đề xuất rerun neural models để tăng score, thêm source labels giả lập,
revive old queue hoặc sửa split sau khi thấy failures. AM1 cần approval mới;
không hành động tiếp chỉ vì review file đã lưu. Không manuscript/Canva/release
work ở bước này.
