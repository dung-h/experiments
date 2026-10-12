> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Estimator identity và fidelity

Final Scientific Freeze v2. Registry diễn giải dẫn xuất, không thay registry/protocol lịch sử. Phạm vi bao phủ: mọi variant trong 34 dòng Track A, 49 dòng Track B, native TN/analog và original Scholten không evaluable. Không coi số variant là số paper. Các controls lịch sử được nhóm riêng, không tráo vào cohort hiện tại.

Các category mô tả implementation: native component có thể được gọi nguyên bản trong một pipeline vẫn có nominal-context adaptation. Calibrated adaptation là predictor mới có học nhãn. Faithful reproduction chỉ dùng khi toàn bộ method/procedure được chứng minh; không gán nhãn đó chỉ vì sử dụng cùng data hoặc có số gần paper. `UNVERIFIED / not evaluated` giữ bất định thay vì ép category.

[Bảng định danh](tables/estimator_identity.md) và [index đầy đủ method × context](tables/method_context_index.csv). Các constant references dùng clock của context tương ứng; không mang clock QPU sang simulator.

## mali_graph: Ma–Li-style Graph + globals

- Source/paper: https://arxiv.org/html/2411.15631v2.
- Task gốc: Observed circuit runtime.
- Observation unit gốc: circuit execution.
- Target gốc: observed runtime.
- Input gốc: Circuit node identity/wires/T1T2 and globals.
- Mechanism gốc: TransformerConv DAG + global MLP.
- Input benchmark: DAG, raw178 node/calibration; raw51 globals, parent40; no shots/backend ID.
- Implementation: TransformerConv + global MLP, three-seed median.
- Khác biệt: Fit-only per-feature moments, nominal wire-index T1/T2, qualified recipes, grouped splits; no active acquisition; parent retains inner1–3 fit.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: So sánh pipeline trên cùng test IDs.
- Không được kết luận: Exact reproduction; topology-only gain; recovered job-day placement.

Evidence: [mali_feature_schema_audit.md](../benchmark/mali_feature_schema_audit.md); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## mali_mlp: Matched global-feature MLP

- Source/paper: https://arxiv.org/html/2411.15631v2.
- Task gốc: Observed circuit runtime.
- Observation unit gốc: circuit execution.
- Target gốc: observed runtime.
- Input gốc: Circuit node identity/wires/T1T2 and globals.
- Mechanism gốc: TransformerConv DAG + global MLP.
- Input benchmark: raw51 / parent40; no graph/calibration/shots/backend.
- Implementation: Global-only network, matched neural splits/seeds.
- Khác biệt: No author processed real tensor; input support varies by fit.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Global-only pipeline contrast.
- Không được kết luận: Identical-information topology ablation; hardware adaptation from collisions.

Evidence: [mali_feature_schema_audit.md](../benchmark/mali_feature_schema_audit.md); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## qonductor: Qonductor-style six regressors + inner selector

- Source/paper: https://arxiv.org/html/2408.04312v2.
- Task gốc: Job resource planning.
- Observation unit gốc: job; restricted here to one circuit.
- Target gốc: observed job time.
- Input gốc: Counter called swap (CX/CZ/ECR); operand-stack depth; touched width; shots; circuit count.
- Mechanism gốc: Six regressor families and polynomial features.
- Input benchmark: swap counts CX/CZ/ECR; operand-stack depth; active width; shots; circuit_count=1.
- Implementation: ET/RF/GB/Ada/HGB/polynomial; 307 bounded candidates; four inner folds then full outer refit.
- Khác biệt: Pinned extraction/constructors; bounded rather than full upstream search; qualified nominal compiled QPack view; mitigation not reproduced.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Six-family and inner-only selector comparison.
- Không được kết luận: Literal SWAP; full search/mitigation reproduction; source-local fits masquerading as unified.

Evidence: [track_a_fidelity.csv](../benchmark/track_a_fidelity.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## qcre: QCRE/Tremba pinned timing component

- Source/paper: https://github.com/mtkgv/qcre.
- Task gốc: Gate-aware duration.
- Observation unit gốc: compiled unitary circuit.
- Target gốc: unitary critical-path seconds.
- Input gốc: Device-compiled BQSKit circuit and native duration map.
- Mechanism gốc: Per-wire max arrival time + location-dependent native gate duration.
- Input benchmark: BQSKit unitary view and nominal native duration map.
- Implementation: Pinned estimate_runtime, max per-wire arrival time.
- Khác biệt: Native component only; terminal measurement/barriers omitted; historical routing unknown.
- Fidelity của implementation: Original/native estimator.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Original component invoked under declared nominal context.
- Không được kết luận: Full QCRE compilation/paper reproduction; raw service-time predictor.

Evidence: qcre_source_receipt.json (historical source; not bundled); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## qiskit: Qiskit scheduled-duration reference

- Source/paper: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit#estimate_duration.
- Task gốc: Scheduled circuit duration.
- Observation unit gốc: scheduled circuit.
- Target gốc: scheduled single-shot seconds.
- Input gốc: Circuit, backend Target/durations.
- Mechanism gốc: Instruction durations and circuit scheduling.
- Input benchmark: Frozen circuit and nominal Target.
- Implementation: QuantumCircuit.estimate_duration.
- Khác biệt: Backend-matched snapshot not verified job-day; possible measurement/reset semantics differ from QCRE.
- Fidelity của implementation: Original/native estimator.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Native API timing reference.
- Không được kết luận: Scheduled duration equals observed service or QCRE unitary clock.

Evidence: [track_a_fidelity.csv](../benchmark/track_a_fidelity.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## shot_scale: Deterministic shot multiplication

- Source/paper: Frozen internal protocol.
- Task gốc: Local comparator.
- Observation unit gốc: one circuit at shots S.
- Target gốc: S × scheduled one-shot seconds.
- Input gốc: Native timing and shots.
- Mechanism gốc: Multiply duration by S.
- Input benchmark: Native timing and shots.
- Implementation: Multiply duration by S.
- Khác biệt: No measured inter-shot/readout/host/batch cost decomposition.
- Fidelity của implementation: Generalized proxy.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Declared repeat-cost proxy.
- Không được kết luận: Verified shot extrapolation or provider runtime identity.

Evidence: [runtime_estimator_interpretation.md](../benchmark/runtime_estimator_interpretation.md); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## scholten_proxy: Scholten-inspired generalized proxy

- Source/paper: https://arxiv.org/html/2307.04980v1.
- Task gốc: Quantum-kernel job time.
- Observation unit gốc: generic one-circuit observation (original: kernel job).
- Target gốc: M*S*D_eff/C original; generic-depth nominal seconds locally.
- Input gốc: Kernel template repetitions, kernel/QV depth expectations, system throughput.
- Mechanism gốc: Shots × circuit count × QV-normalized effective layers / CLOPS.
- Input benchmark: Compiled wire depth + admitted nominal throughput.
- Implementation: Generic depth/shots/throughput, 3399/4515 eligible.
- Khác biệt: Generic wire depth is not ensemble/QV-normalized D_eff; 1116 unavailable; no CLOPS_h/v conversion.
- Fidelity của implementation: Generalized proxy.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Evaluate explicitly generalized proxy on its eligible rows.
- Không được kết luận: Scholten original accuracy; manufactured CLOPS; full-panel ranking.

Evidence: [track_a_fidelity.csv](../benchmark/track_a_fidelity.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## scholten_original: Scholten original (not evaluable)

- Source/paper: https://arxiv.org/html/2307.04980v1.
- Task gốc: Quantum-kernel job time.
- Observation unit gốc: kernel job.
- Target gốc: M*S*D_eff/C.
- Input gốc: Kernel template repetitions, kernel/QV depth expectations, system throughput.
- Mechanism gốc: Shots × circuit count × QV-normalized effective layers / CLOPS.
- Input benchmark: Missing kernel/template and QV ensemble expectations with compatible throughput.
- Implementation: D_eff=E[depth(U†(x)U(y))]/E[depth(QV(v,v))]*v; v=ceil(sqrt(2Dn)).
- Khác biệt: 0/4515 verified eligibility; not executed as original.
- Fidelity của implementation: UNVERIFIED / not evaluated.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Explain input eligibility and method definition.
- Không được kết luận: Use proxy scores as original-paper reproduction.

Evidence: [runtime_estimator_interpretation.md](../benchmark/runtime_estimator_interpretation.md).

## hyb: Hyb-HANAS identifiable cost component

- Source/paper: https://arxiv.org/html/2603.00625v1.
- Task gốc: Reliability-aware hybrid NAS cost.
- Observation unit gốc: reliability-adjusted circuit/training work.
- Target gốc: useful-work/resource effective seconds.
- Input gốc: Native durations/errors, T2, routing, parameters/steps.
- Mechanism gốc: Gate/routing cost divided by survival; full paper scales gradient/steps.
- Input benchmark: Native timing/errors/T2/routing and survival/log-cost.
- Implementation: Gate cost divided by survival; identifiable one-circuit terms.
- Khác biệt: No full NAS/gradient/step model; finite4175; log-cost4367; undefined values explicit.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Cost-component and missingness study.
- Không được kết luận: Literal execution-time identity; full hybrid NAS reproduction.

Evidence: [track_a_fidelity.csv](../benchmark/track_a_fidelity.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## service_bridge: Outer-train analytical-to-service bridge

- Source/paper: Frozen internal protocol.
- Task gốc: Local comparator.
- Observation unit gốc: eligible circuit observation.
- Target gốc: native timing/cost/throughput.
- Input gốc: Raw or shot-scaled predictor plus eligible outer-train labels.
- Mechanism gốc: Affine/log-affine; separately named log-cost Ridge.
- Input benchmark: Raw or shot-scaled predictor plus eligible outer-train labels.
- Implementation: Affine/log-affine; separately named log-cost Ridge.
- Khác biệt: New fitted predictor; coverage depends on route.
- Fidelity của implementation: Calibrated adaptation.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Train-only service calibration and matched reference comparison.
- Không được kết luận: Physical launch/host overhead identified; native estimator fidelity restored.

Evidence: [analytical_controls_v1.csv](../../analysis/analytical_controls.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## references: Train-only constant references

- Source/paper: Frozen internal protocol.
- Task gốc: Local comparator.
- Observation unit gốc: held-out observation.
- Target gốc: observed service/runtime reference.
- Input gốc: Outer-train target labels; source identity only for source median.
- Mechanism gốc: Mean/global median/source median; fallback rules explicit.
- Input benchmark: Outer-train target labels; source identity only for source median.
- Implementation: Mean/global median/source median; fallback rules explicit.
- Khác biệt: Retrospective secondary controls; source-provenance information advantage.
- Fidelity của implementation: Benchmark-defined diagnostic.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Measure skill beyond simple constants.
- Không được kết luận: Preregistered primary models or equal-information source reference.

Evidence: [qpu_predictive_skill.csv](../benchmark/qpu_predictive_skill.csv); [qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv).

## sim_mali: Context-specific Graph/global controls

- Source/paper: https://arxiv.org/html/2411.15631v2.
- Task gốc: Simulation runtime.
- Observation unit gốc: circuit/configuration.
- Target gốc: simulator observed execution time.
- Input gốc: Circuit graph + globals.
- Mechanism gốc: DAG/global learned predictors.
- Input benchmark: Local DAG/node/globals per frozen context.
- Implementation: Graph/global networks.
- Khác biệt: Different engines, schemas and fit policies; not exact upstream model run.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: context-specific warm sample/get_state or engine-reported time.
- Được kết luận: Within-context pipeline comparison.
- Không được kết luận: Universal simulator winner; isolated topology benefit.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv).

## azizov: Azizov-described source/hybrid/transpiled GNN

- Source/paper: https://arxiv.org/html/2609.12980v1.
- Task gốc: Noisy Aer runtime after transpilation.
- Observation unit gốc: circuit × noise/transpilation context.
- Target gốc: noisy Aer after-transpile execution.
- Input gốc: Source/compiled graph, gate/structural globals, backend noise, opt-level.
- Mechanism gốc: Source/hybrid/transpiled two-branch GNN + five regressors.
- Input benchmark: Source/compiled graph and globals; FakeSherbrooke Opt1 shots1024.
- Implementation: Three two-branch GNN representations; three-seed medians.
- Khác biệt: Paper full encoding/code unavailable; one backend/level and local hardware/software.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: warm Aer execution seconds after transpilation.
- Được kết luận: Local source-to-transpiled pipeline contrast.
- Không được kết luận: Exact authors encoding or full two-backend/four-level reproduction.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [simulator_paired_v1.csv](../../analysis/simulator_paired.csv).

## sim_classical: Context-specific classical/complexity controls

- Source/paper: internal frozen protocol.
- Task gốc: Local runtime baseline.
- Observation unit gốc: circuit/configuration.
- Target gốc: context-specific measured runtime.
- Input gốc: Width, gate counts, shots and complexity-inspired features.
- Mechanism gốc: Median, six tabular/linear models, work-scaled Ridge.
- Input benchmark: Local source/compiled counters or 2^n×gate work features.
- Implementation: Linear/Ridge/SVR/RF/XGBoost; frozen context-specific fits.
- Khác biệt: Different feature views, budgets and clocks; not a single pooled method.
- Fidelity của implementation: Benchmark-defined diagnostic.
- Target đánh giá: same context as each neural comparator.
- Được kết luận: Matched-context empirical/complexity baseline.
- Không được kết luận: Physical launch coefficient or universal scaling law.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv).

## family_joint: Family-aware joint MPS adaptation

- Source/paper: https://arxiv.org/html/2606.11620v1.
- Task gốc: Joint approximation/runtime prediction.
- Observation unit gốc: circuit × approximation rung.
- Target gốc: joint runtime/approximation target.
- Input gốc: Circuit features and approximation/runtime-quality labels.
- Mechanism gốc: Family classifier + FiLM/residual + runtime/threshold branches.
- Input benchmark: Family context + local six-chi ladder, fidelity>=0.99.
- Implementation: Family classifier, residual/FiLM and joint heads; agnostic control.
- Khác biệt: Quantum Rings threshold replaced by local bond/fidelity policy; 840rungs/140hashes.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: warm get_state seconds plus quality probability.
- Được kết luận: Runtime/quality/safe-selection evaluated separately.
- Không được kết luận: Original engine reproduction; Brier improvement proves safe selection.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv); [mps_selection_safety_v1.csv](../../analysis/mps_selection_safety.csv).

## family_fixed: Fixed-MPS family residual control

- Source/paper: https://arxiv.org/html/2606.11620v1.
- Task gốc: Joint approximation/runtime prediction.
- Observation unit gốc: circuit at fixed chi16.
- Target gốc: family-conditioned runtime.
- Input gốc: Circuit features and approximation/runtime-quality labels.
- Mechanism gốc: Family classifier + FiLM/residual + runtime/threshold branches.
- Input benchmark: Fixed-state context family/global inputs.
- Implementation: Family residual versus family-agnostic network.
- Khác biệt: Runtime-only fixed-bond control, not full joint approximation model.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: warm get_state at FP64 chi16.
- Được kết luận: Fixed-context family-conditioning contrast.
- Không được kết luận: Joint model reproduction or quality guarantee.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv).

## maestro: Maestro-style local component

- Source/paper: https://arxiv.org/html/2512.04216v1.
- Task gốc: Engine selection from runtime estimates.
- Observation unit gốc: circuit/engine/hardware.
- Target gốc: calibrated runtime supporting engine choice.
- Input gốc: Width, gate mix, shots, backend/calibration; MPS bond surfaces.
- Mechanism gốc: Hardware-calibrated operation complexity sum + sampling term.
- Input benchmark: Width/gate/sampling calibration, shots1000.
- Implementation: Local operation-complexity sum + sampling terms.
- Khác biệt: Optimizer off, one engine; closed Composer selection not reproduced; 143/150 hashes.
- Fidelity của implementation: Architecture/core adaptation.
- Target đánh giá: QCSim CPU FP64 engine-reported seconds; fresh processes.
- Được kết luận: Component skill versus same-target Ridge/median.
- Không được kết luận: Persistent warm wall-clock or full automatic selection reproduction.

Evidence: [simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv).

## tn: cuTensorNet native RUNTIME_EST

- Source/paper: https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornetcontractionoptimizerinfoattributes-t.
- Task gốc: Contraction plan performance.
- Observation unit gốc: selected contraction plan.
- Target gốc: all-slice selected-plan contraction seconds.
- Input gốc: Network, selected path, slices, workspace, GPU.
- Mechanism gốc: Native architecture-aware RUNTIME_EST.
- Input benchmark: Network/path/slices/workspace/GPU/complex64.
- Implementation: Native architecture-aware cost estimate.
- Khác biệt: Scalar not full state/sample; internals opaque; build/search/first excluded from comparison.
- Fidelity của implementation: Original/native estimator.
- Target đánh giá: same-plan warm scalar contraction.
- Được kết luận: Native same-plan calibration diagnostic.
- Không được kết luận: Internal formula recovered; predicts whole simulator API.

Evidence: [tn_same_plan_slices_v1.csv](../../analysis/tn_same_plan_slices.csv).

## pasqal: Pasqal analog MPS/TDVP companion

- Source/paper: https://pasqal-io.github.io/emulators/v2.4.0/emu_mps/advanced/resource_estimation/.
- Task gốc: Analog MPS/TDVP resources.
- Observation unit gốc: analog program/configuration.
- Target gốc: analog MPS/TDVP runtime resource model.
- Input gốc: Analog pulse program, atoms, bond/TDVP settings.
- Mechanism gốc: Work terms N²χ³ and N³χ² with fitted coefficients.
- Input benchmark: Atoms, pulse program, chi and TDVP settings.
- Implementation: Documented N²chi³/N³chi² work terms; measured quality/timing grid.
- Khác biệt: 18cells; coefficients not identifiable/promoted; no validated predictor.
- Fidelity của implementation: Benchmark-defined diagnostic.
- Target đánh giá: host MPSBackend.run, excluding quality capture.
- Được kết luận: Analog timing/quality observations.
- Không được kết luận: Digital leaderboard or identified extrapolation coefficients.

Evidence: [pasqal_stage_quality_v1.csv](../../analysis/pasqal_stage_quality.csv).

## historical_controls: Historical structural/transfer/width controls

- Source/paper: Frozen internal protocol.
- Task gốc: Local comparator.
- Observation unit gốc: historical circuit or kernel population.
- Target gốc: study-specific service/log-runtime/kernel time.
- Input gốc: Coarse graphs, static blocks, interventions or PyTorch work features.
- Mechanism gốc: Fixed-model intervention or retrained controls, explicitly distinguished.
- Input benchmark: Coarse graphs, static blocks, interventions or PyTorch work features.
- Implementation: Fixed-model intervention or retrained controls, explicitly distinguished.
- Khác biệt: 340/170/coarse representation or separate kernel; topology scores mean predictions, not mean seed MAEs or current median ensemble.
- Fidelity của implementation: Benchmark-defined diagnostic.
- Target đánh giá: historical clocks kept separate from parent4515.
- Được kết luận: Bounded reliance/incremental evidence.
- Không được kết luận: Full4515 topology causality or universal pretraining/scaling result.

Evidence: [feature_contribution.md](../benchmark/feature_contribution.md).

## holdout_families: QNN / Random held-out families (not estimators)

- Source/paper: Frozen internal protocol.
- Task gốc: Local comparator.
- Observation unit gốc: family holdout.
- Target gốc: observed service seconds.
- Input gốc: Frozen family test IDs.
- Mechanism gốc: Evaluate existing Graph/MLP/CPU models.
- Input benchmark: Frozen family test IDs.
- Implementation: Evaluate existing Graph/MLP/CPU models.
- Khác biệt: These labels identify workloads, not QNN or random-prediction baseline algorithms.
- Fidelity của implementation: Benchmark-defined diagnostic.
- Target đánh giá: archived observed service/execution seconds; source boundaries retained.
- Được kết luận: Systematic generalization failures.
- Không được kết luận: Invent a quantum-neural predictor or random estimator score.

Evidence: [generalization.md](../benchmark/generalization.md).

## Quy tắc runtime boundary

QCRE unitary path, Qiskit single-shot schedule, shot multiplication, Hyb reliability cost và Scholten proxy đều khác observed service. Chỉ các bridges có học outer-train labels mới xuất dự đoán service. Dense sample, MPS get_state, Aer sau transpile, QCSim engine time, scalar contraction và analog TDVP phải đọc ở context riêng. Native `RUNTIME_EST` không phải learned service bridge.
