> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Minimal experimental amendment — planning only

## AM1: Đo estimator overhead trên frozen estimators

**Status: NOT AUTHORIZED.** Một amendment, áp dụng hai tracks theo contexts,
không thêm model hoặc retrain để cải thiện leaderboard. Chỉ cần nếu thesis
muốn đóng đầy đủ RQ4 accuracy–cost; bounded accuracy/failure claims không cần.
Không suy ra phải chạy AM1 chỉ vì file này tồn tại.

**Hypothesis.** Khi tính parsing/features/transpilation/model loading và
prediction, một accuracy advantage nhỏ có thể không bù được estimation cost,
đặc biệt ở simulator targets dưới millisecond. Hypothesis này có thể bị bác:
matched end-to-end cost có thể negligible, hoặc amortize tốt với repeated queries.

**Exact question.** Trên cùng frozen query/input boundaries, incremental cost
của Graph/MLP/classical/analytical/component estimate là bao nhiêu, đổi lấy
held-out accuracy nào đã lưu? Không đo quantum execution trong amendment này.

**Tracks/data/unit.** Track A: 4.515 parent, dùng từng frozen outer-fold checkpoint
và transform; predeclare tối đa 10 queries/fold, chọn label-free theo hash và
structural-size strata (50 query identities), report size/group coverage.
Track B: tối đa 30 hashes/context chọn label-free từ shared core cho dense FP32,
fixed-MPS, Aer và QCSim, chỉ những contexts có đủ serialized estimators.
Đơn vị là estimator invocation tại declared input/config, không circuit job.
Các numbers này là proposed resource/sample limits, chưa frozen protocol.

**Target.** End-to-end estimator wall seconds, stage seconds, peak CPU RSS/
CUDA memory, model artifact bytes; offline training/calibration time riêng.
Existing quantum/simulator runtime labels chỉ dùng làm denominator/accuracy
reference, không remeasure. GPU synchronization trước/sau timed neural calls.

**Baselines.** Train-only median, frozen MLP, frozen selected classical model,
QCRE/Qiskit/raw-and-calibrated routes cùng query where eligible. Track B dùng
frozen matched Ridge/RF/component theo context; không force mọi model qua task
không phù hợp. Không choose comparator sau khi xem cost.

**Preparation gate trước execution approval.** Inventory serialization và
hashes. Không giả rằng predictions+params là trained checkpoint. Nếu classical
model chưa serialize, dừng route và xin duyệt bounded selected-model refit
trên original outer-train với original selected params; không rerun search,
không neural retrain, không thay frozen predictions hoặc score. Nếu không duyệt
refit, chỉ báo supported subset, không claim full cost comparison.

**Split.** Reuse outer-test identities và corresponding outer-trained models;
no tuning/inference rule fitted on test targets. Existing accuracy giữ nguyên.
Calibration reuse frozen coefficients; chỉ đo invocation, không chọn calibrator
theo latency. Preprocessing cost dùng circuits known before execution.

**Controls.** Same host/software/threads/precision, no concurrent workloads.
Một worker; predeclare cold process/model-load versus warm reuse; model-cache,
parsed/compiled input reuse và batch 1/32 boundaries; ten warmups, 30 timed repeats
where feasible. Include parsing/features/target transpilation/planning if route
needs them, and separately show reusable-prepared-input scenario. Include
median-of-three cost, không time one seed rồi gán cho ensemble. Quality reference,
original simulator execution và measured telemetry không được predictor inputs.

**Metrics.** Median/p90/p99 latency and stage share per structural group/context,
coverage/failures, memory, ratio estimator latency / existing target runtime,
paired same-query cost deltas. Combine with existing MAE/relative errors and
conditional intervals, không compute new leaderboard accuracy from timed reruns.
Offline cost amortization for assumed query counts 1/100/10.000 phải ghi là
scenarios, không actual scheduler throughput hoặc measured economic gain.

**Expected computation/value.** No new GPU training, QPU job, simulator execution
or QCEC. New feature/planning/inference measurements are experiments and need
approval. Proposed ceiling 2 hours, 1 worker, preserve 3 GiB disk floor and no automatic
resource growth. Đây là budget ceiling, không measured ETA. Existing cost logs
được kiểm trước để giảm số measurements. AM1 có thể thay recommendation về
expensive predictors cho tiny targets, không thay frozen accuracy/failure facts.

**Stop criterion.** Stop on identity/model/checkpoint/feature mismatch, missing
serialization, protocol ambiguity, cap exhaustion or inadequate disk/RAM.
Report incomplete coverage, không đổi dataset/mappings hay tune implementation
để có tốc độ đẹp. Stop once predeclared query/repeat grid terminates. Không
escalate thành simulator scaling sweep hoặc new training queue.

## Vì sao không đề xuất thêm experiments lúc này?

Source/hardware causal transfer cần crossed compatible labels, không chỉ train
thêm 21 cases. Original Scholten/Pasqal/closed Maestro cần khác target/input scope.
Topology-only gain không mục tiêu chính của benchmark existing pipelines.
Các câu hỏi này được ghi G3, không lén yêu cầu rerun để tăng model count hoặc
đẹp scores. G1 cost/scale reanalysis và target-threshold analysis không cần
experiment nhưng chỉ dùng evidence/decision thresholds có thật.

Nếu không muốn mở freeze, lựa chọn khoa học hợp lệ là giữ RQ4 PARTIALLY ANSWERED
và không claim deployment efficiency. Không tự đổi nó thành ANSWERED bằng
wording hạn chế đến mức bỏ qua câu hỏi chi phí.
