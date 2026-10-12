> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Khung câu hỏi và tiêu chí đóng nghiên cứu

## Working framework hiện tại

Mục tiêu là benchmark existing runtime estimators cho real QPU và simulator,
không phát triển neural architecture mới hoặc chứng minh Graph phải thắng.
Các RQ dưới đây là framework được người dùng yêu cầu cho assessment này.
Không sửa Research Contract v1.1 hoặc đổi tên frozen producing authority.

| Working RQ | Nội dung | Nối với historical authority |
| --- | --- | --- |
| RQ1 | Predictive accuracy/comparative effectiveness | Contract RQ1 pipelines, RQ2 analytical, RQ3 local simulator |
| RQ2 | Generalization | Contract RQ4; `artifacts/final_research/rq4_report.md` vẫn giữ tên cũ |
| RQ3 | Error/robustness/failures | Cross-cutting residual, support và quality analyses |
| RQ4 | Accuracy–cost/scalability | Target scaling evidence + estimator-overhead question; không phải old RQ4 grid |
| RQ5 | Fidelity/applicability/practical selection | Method/input/clock/quality identity, không new selection model |

Các headings trong Thesis Interpretation dùng numbering khác nữa. Khi trích
claim IDs C01–C32 phải giữ source ID và dùng crosswalk, không suy luận cùng
RQ number là cùng question. Historical contract còn ghi 10/21; full-grid report
là current coverage authority 21/21. Hai văn bản không được silently đồng bộ.

## Operational definitions

- ANSWERED: minimum evidence đầy đủ cho đúng population/endpoint đã nêu.
- ANSWERED WITH QUALIFICATIONS: có câu trả lời bảo vệ được nhưng phải giữ
  giới hạn về population, clocks, fidelity hoặc uncertainty.
- PARTIALLY ANSWERED: một phần quan trọng của broad question chưa evaluated.
- UNANSWERED: không có direct evidence cho question cụ thể, không phải vì
  method trả kết quả xấu. Pasqal predictive accuracy khác analog timing.
- CLOSE đóng bounded question, không cấp public-release approval.
- REANALYZE SAVED EVIDENCE không chạy model, checker hoặc workload mới.
- MINIMAL EXPERIMENTAL AMENDMENT là đề xuất có điều kiện, chưa authorized.
- DEFER WITH EXPLICIT LIMITATION phải nêu stronger claim nào bị cấm.

Unseen observation ID khác unseen consumed input, structural group, family,
backend, source và hardware. Exact hash grouping bảo vệ shared-circuit splits
nhưng không chứng minh tất cả near-duplicate structures độc lập. Backend
holdout được phép lặp cấu trúc có chủ ý; family holdout dùng structural embargo.
Case assignment không là independent observation. Ranking là conditional theo
test population, target, estimator information và training procedure.

Target-runtime scaling là thời gian simulator/QPU thực thi. Estimator overhead
là chi phí tạo estimate: parsing/features/transpile/planning/calibration/load/
prediction. Training là offline cost, không chia tổng training duration cho số
test rows để giả làm inference latency. Calibration intercept không physical
launch overhead. Native estimator availability có thể cần plan trước execution.

## Đối soát từng RQ và Track

### RQ1 — Track A

**Câu hỏi.** Existing estimators dự đoán archived circuit service time chính xác đến đâu?

**Vì sao quan trọng.** Kiểm tra predictive benefit, không lấy độ phức tạp model làm giá trị nghiên cứu.

**Minimum evidence.** Shared held-out IDs, train-only preprocessing/tuning/calibration, strong simple/classical controls, errors/coverage và paired uncertainty.

**Dataset và observation unit.** 4.515 observations = Ma–Li 340 + Qonductor 230 + QPack 3.945; 165 conservative groups.

**Runtime target.** Archived observed execution/service seconds; source boundaries retained, không pulse duration chung.

**Evaluation protocol.** 5 outer / 4 inner grouped folds; seed 42. Neural seeds 42/1234/31415, frozen per-row median; regression six families/307 candidates.

**Estimators.** Ma–Li-style Graph/global MLP; Qonductor-style six regressors/selector; QCRE, Qiskit, Hyb, Scholten proxy và raw/calibrated variants; train-only references.

**Evidence hiện có.** 34 metric rows; matched analytical controls; saved paired intervals và source/group-balanced metrics.

**Finding.** Graph MAE 0,83413 s, MLP 0,89342 s, selector 0,88666 s. Graph skill 26,7% vs global median nhưng 5,9% vs selector; selector/MLP CIs chứa 0. Equal-source ranking đổi.

**Threats to validity.** Reconstructed tiers, heterogeneous clocks, 87,4% QPack rows/sáu structures; unequal inputs/capacity/tuning/refit procedures. Đây là pipeline benchmark, không equal-information architecture contest.

**Bằng chứng thiếu.** Original Scholten eligibility; exact-paper reproductions; universal model ranking. Không thiếu primary shared-row scores.

**Ảnh hưởng đến kết luận.** Không chặn bounded pipeline comparison; chặn claim mọi original method đã evaluated hoặc Graph universally best.

**Metrics và đơn vị.** Coverage, MAE/MedAE/RMSE, R², log1p/relative errors, p90/p99/max, skill versus matched references và paired MAE-delta intervals; analytical eligible subsets riêng.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [qpu](../../analysis/qpu_metrics.csv); [qpu_pairs](../../analysis/qpu_shared_reference_comparisons.csv); qpu_shared (historical source; not bundled); [controls](../../analysis/analytical_controls.csv); profile (historical source; not bundled); parent_protocol (historical source; not bundled); [registry](../../interpretation/thesis/estimator_fidelity_registry.md).

### RQ1 — Track B

**Câu hỏi.** Estimator nào có accuracy hữu ích trong từng simulator task/context?

**Vì sao quan trọng.** Engine, API và approximation thay đổi target; cùng đơn vị seconds không bảo đảm comparability.

**Minimum evidence.** Same exact hashes/folds/engine/precision/shots/clock; references và paired comparisons; quality/terminal denominator.

**Dataset và observation unit.** 150 exact core hashes q2–9; Maestro 143/150; joint ladder 840/900 measured rungs thuộc 140/150 hashes; TN 603/612 paired plan sessions; analog 18 cells.

**Runtime target.** Warm sample, warm get_state, after-transpile Aer, QCSim engine-reported time, scalar contraction và analog run tách riêng.

**Evaluation protocol.** Context-specific grouped five-fold OOF; three-seed medians where frozen; separate family-component holdout for ladder; native TN same-plan pairing.

**Estimators.** Graph/global models; Azizov source/hybrid/transpiled; family-conditioned/agnostic; Maestro-style component; native RUNTIME_EST; classical/complexity controls; Pasqal diagnostic only.

**Evidence hiện có.** 49 metrics/seven predictive contexts, paired tables, TN ratios và analog quality/time; không cross-context leaderboard.

**Finding.** Dense RF thấp MAE hơn Graph; fixed-MPS Graph tốt hơn Ridge; Aer compiled view giúp nhưng GNN superiority over XGBoost chưa resolve; Maestro local component có skill; family conditioning chưa resolve gain.

**Threats to validity.** Small-width/resource-qualified survivor panel; incomplete ladders; different information/budgets; QCSim engine clock không full wall-clock. Pasqal chưa có predictive validation.

**Bằng chứng thiếu.** Full Maestro selector/closed engine, original Quantum Rings reproduction, Pasqal predictive coefficients; broad common wall-clock engine comparison.

**Ảnh hưởng đến kết luận.** Không chặn evaluated matched-context findings; chặn original-paper/full-wall-clock/global engine winner claims.

**Metrics và đơn vị.** Coverage, MAE/MedAE/RMSE, R², log1p/relative errors, p90/p99/max, skill versus matched references và paired MAE-delta intervals; analytical eligible subsets riêng.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [sim](../../analysis/simulator_metrics.csv); [sim_pairs](../../analysis/simulator_paired.csv); [quality](../../analysis/mps_selection_safety.csv); [tn](../../analysis/tn_same_plan_slices.csv); [analog](../../analysis/pasqal_stage_quality.csv); sim_producing (historical source; not bundled); [registry](../../interpretation/thesis/estimator_fidelity_registry.md).

### RQ2 — Track A

**Câu hỏi.** Generalize đến circuits, families, sizes, backends hoặc sources chưa thấy đến đâu?

**Vì sao quan trọng.** Random/grouped interpolation không thay thế domain transfer.

**Minimum evidence.** Leakage-safe groups và predeclared OOD splits; support/composition, repeated-input checks, train-only references, paired uncertainty.

**Dataset và observation unit.** Parent 165 groups; full grid chín family/mười backend-context/hai size cases; 6.684 assignments nhưng 4.515 unique IDs.

**Runtime target.** Source-retained service seconds trong từng case, không pool overlapping cases.

**Evaluation protocol.** 21 frozen cases; family structural embargo; intentional structural overlap for backend transfer; QPack sizes có 3/2 test groups; 126 Graph/MLP×seed cells.

**Estimators.** Graph/MLP median-of-three; CPU selectors/analytical routes và train-only references, case-specific fresh fits.

**Evidence hiện có.** 21/21 CPU + neural cases; test-ID hashes, lineage, shared-row and known/new support tables.

**Finding.** QNN/Random neural transfer thất bại nghiêm trọng; CPU selector vẫn hữu ích. Kyoto là known-circuit backend context; Osaka mixed known/new; Quito mostly repeated QPack structures.

**Threats to validity.** Source/family/clock/shots/generation confounding; low backend/size groups; no source-held-out study; nominal-index calibration; overlapping case assignments.

**Bằng chứng thiếu.** Systematic source/shot transfer và isolated hardware-generation effect; representative large-size evaluation.

**Ảnh hưởng đến kết luận.** Không phủ nhận completed negative OOD findings; không đóng toàn bộ generalization agenda chỉ bằng 21/21 completion.

**Metrics và đơn vị.** Per-case MAE/MedAE/RMSE/R² và reference skill, shared-ID deltas, group support, known/new structure counts, input collision counts, train/test support distributions; không average R².

**Status:** PARTIALLY ANSWERED. **Action:** DEFER WITH EXPLICIT LIMITATION.

**Exact source paths:** [freeze](../../generalization/report.md); generalization (historical source; not bundled); generalization_pairs (historical source; not bundled); generalization_support (historical source; not bundled); generalization_sources (historical source; not bundled); generalization_protocol (historical source; not bundled); profile (historical source; not bundled).

### RQ2 — Track B

**Câu hỏi.** Generalize qua circuits, families, width và simulator/hardware contexts đến đâu?

**Vì sao quan trọng.** Hardware-calibrated and learned predictors có thể chỉ nội suy trong calibration support.

**Minimum evidence.** Hash-grouped splits; dedicated OOD held-outs; train-only calibration and matched clocks/tasks; structural support.

**Dataset và observation unit.** Core q2–9 hashes; family-component MPS ladder; historical PyTorch dense width corpus và CUDA-Q RX-only precision/shot pilot, không nhập làm common-panel scores.

**Runtime target.** Separate engine/task clocks; historical kernel target khác current sample/get_state.

**Evaluation protocol.** Primary hash OOF; separate ladder family holdout; historical width holdouts. Precision-specific training không phải precision transfer.

**Estimators.** Per-context learned/empirical models, family conditioning; historical work-scaled models; native TN diagnostics.

**Evidence hiện có.** Ladder family-holdout scores, width slices, historical width-held-out errors, same-plan TN core/frontier.

**Finding.** Family-conditioning advantage chưa resolve. Historical dense width holdout có boundary failures; không thể chuyển score đó sang current CUDA-Q/Aer/engine khác.

**Threats to validity.** Một local resource profile, little width support; context-specific survivor sets; no common trained cross-engine transfer; descriptive width slices không OOD experiment.

**Bằng chứng thiếu.** Current-context systematic width/engine/hardware transfer và unseen hardware calibration efficacy.

**Ảnh hưởng đến kết luận.** Core local accuracy còn hợp lệ; broad portable simulator predictor/general scaling chưa answered.

**Metrics và đơn vị.** Per-case MAE/MedAE/RMSE/R² và reference skill, shared-ID deltas, group support, known/new structure counts, input collision counts, train/test support distributions; không average R².

**Status:** PARTIALLY ANSWERED. **Action:** REANALYZE SAVED EVIDENCE.

**Exact source paths:** [sim](../../analysis/simulator_metrics.csv); sim_slices (historical source; not bundled); sim_widths (historical source; not bundled); joint_protocol (historical source; not bundled); width_study (historical source; not bundled); launch (historical source; not bundled); tn_scaling (historical source; not bundled).

### RQ3 — Track A

**Câu hỏi.** Errors tập trung ở đâu, khi nào estimators unreliable?

**Vì sao quan trọng.** MAE trung bình có thể che catastrophic failures và deployment limits.

**Minimum evidence.** Signed errors, quantiles/max, absolute/relative errors, per-source/case support, simple references, collision/preprocessing audits.

**Dataset và observation unit.** Parent 4.515 rows và từng independent declared holdout; QNN 26 rows/14 groups, Random 30/16, Kyoto 148/117.

**Runtime target.** Residual = prediction minus observed service; route-specific finite coverage.

**Evaluation protocol.** Saved parent OOF + frozen OOD cases; no seed/row removal after errors.

**Estimators.** All parent routes; Graph/MLP/CPU within each case.

**Evidence hiện có.** Tails, MedAE/RMSE/R², overprediction counts; QNN/Random distributions; raw/post-transform collisions; Hyb missingness.

**Finding.** Cả neural models overpredict mọi QNN/Random row; R² âm nghiêm trọng. Kyoto 148/148 MLP collisions không ID leakage. Source weighting changes ranking; analytical missing values không zero-filled.

**Threats to validity.** Support-shift association không causal feature mechanism; fitted-model conditional uncertainty; model failure khác representation limitation.

**Bằng chứng thiếu.** Prospective intervals, causal per-feature failure attribution, scheduler loss; không thiếu evidence rằng observed failures xảy ra.

**Ảnh hưởng đến kết luận.** Negative findings close bounded failure questions; no prospective safety or isolated mechanism claim.

**Metrics và đơn vị.** Signed residuals, over/underprediction counts, p90/p99/max, relative-error valid counts, per-source/group metrics, missingness/terminal/quality failure rates; EST/warm ratio ở TN.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [qpu](../../analysis/qpu_metrics.csv); [freeze](../../generalization/report.md); [feature_support](../../interpretation/benchmark/generalization_field_support.csv); generalization_support (historical source; not bundled); [controls](../../analysis/analytical_controls.csv); [thesis](../../interpretation/thesis/final_scientific_interpretation.md).

### RQ3 — Track B

**Câu hỏi.** Simulator estimators sai hoặc không an toàn dưới điều kiện nào?

**Vì sao quan trọng.** Runtime accurate vẫn có thể chọn approximation không đạt fidelity.

**Minimum evidence.** Signed/tail errors per-context; terminal/quality coverage; same-rung target; support/slicing/first-warm boundaries.

**Dataset và observation unit.** Core contexts, joint ladder 140 assessed/150 assigned hashes, TN plan sessions, analog grid; historical frontier outcomes riêng.

**Runtime target.** Each context residual; quality violations separate from seconds errors; EST/warm dimensionless ratio.

**Evaluation protocol.** Saved OOF and family-holdout predictions; quality measured outside timing; repeated rungs cluster by hash.

**Estimators.** Runtime predictors, conditioned/agnostic selectors, native EST và diagnostic controls.

**Evidence hiện có.** Context error slices, 64/140 selected quality failures, ten unassessed, 138/140 achievable hashes; TN bias and resource records.

**Finding.** Runtime performance không bảo đảm quality selection. TN core median EST/warm 3,734, frontier 4,132; under locked plan/precision, không whole-API time estimate.

**Threats to validity.** Quality-incomplete hashes and survivor support; small analog grid; internal TN model opaque; no physical overhead decomposition.

**Bằng chứng thiếu.** Prospective quality-safety guarantees, broad hardware regime and exact failure causality.

**Ảnh hưởng đến kết luận.** Observed negative safety/bias findings đã answered; missing guarantees limit deployment claims.

**Metrics và đơn vị.** Signed residuals, over/underprediction counts, p90/p99/max, relative-error valid counts, per-source/group metrics, missingness/terminal/quality failure rates; EST/warm ratio ở TN.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [sim](../../analysis/simulator_metrics.csv); sim_slices (historical source; not bundled); [quality](../../analysis/mps_selection_safety.csv); [tn](../../analysis/tn_same_plan_slices.csv); tn_scaling (historical source; not bundled); [analog](../../analysis/pasqal_stage_quality.csv); joint_protocol (historical source; not bundled).

### RQ4 — Track A

**Câu hỏi.** Accuracy có đáng đổi lấy feature/transpile/train/calibration/inference cost không?

**Vì sao quan trọng.** Scheduler phải trả chi phí estimator; target circuit runtime không phải overhead của predictor.

**Minimum evidence.** Matched-query end-to-end and stage timings, cold/warm, CPU/GPU/RAM, serialized models/calibrators, prediction batch; accuracy trên cùng rows; amortization assumptions.

**Dataset và observation unit.** Frozen parent 30 neural cells và datasets/compiled views; 21-case receipts là execution records, không matched latency benchmark.

**Runtime target.** Estimator computation seconds, peak memory and preparation/training costs; separate from service labels.

**Evaluation protocol.** Historical elapsed fields + producing-code timer inspection; không có dedicated matched cost evaluation split.

**Estimators.** Graph/MLP; six regressors/selector; native analytical routes and calibrated bridges.

**Evidence hiện có.** Parent training-loop elapsed receipts cho 15 Graph +15 MLP cells; no separated test inference timers; regression score receipts thiếu stage timing.

**Finding.** Có descriptive historical training-loop cost, không measured accuracy–deployment-cost frontier. Timer dừng trước outer-test prediction.

**Threats to validity.** Cache/checkpoint/validation overhead, unequal fit rows and budgets, concurrent CPU activity; sum-cell time không overall campaign wall time.

**Bằng chứng thiếu.** Matched feature/transpilation/calibration/single-query and batch inference overhead, model load/cold start, deploy amortization.

**Ảnh hưởng đến kết luận.** Material cho full RQ4 closure hoặc efficiency/deployment-superiority claim; không invalidates accuracy/negative OOD findings.

**Metrics và đơn vị.** Historical recorded loop min/median/max/sum và target-runtime width/bond errors có sẵn; matched estimator-stage/prediction latency, peak memory và amortized cost chưa có đầy đủ.

**Status:** PARTIALLY ANSWERED. **Action:** MINIMAL EXPERIMENTAL AMENDMENT.

**Exact source paths:** parent_runner (historical source; not bundled); parent_protocol (historical source; not bundled); profile (historical source; not bundled).

### RQ4 — Track B

**Câu hỏi.** Accuracy–estimation-cost/scaling trade-off trong từng simulator context là gì?

**Vì sao quan trọng.** Microsecond execution targets có thể nhỏ hơn overhead estimator; planner cost có thể chi phối TN deployment.

**Minimum evidence.** Separate predictor preparation/inference/calibration/planning clocks, matched query timings and memory; target runtime scaling scored separately.

**Dataset và observation unit.** Seven predictive contexts, same-plan TN telemetry, 18 analog cells, historical dense/launch diagnostics.

**Runtime target.** Estimator overhead versus context-specific execution runtime; không so network build/path-search trực tiếp với warm contraction error.

**Evaluation protocol.** Saved measurements and width/bond probes; no unified end-to-end estimator-cost protocol.

**Estimators.** Graph/tabular/component predictors, TN native planner/EST, Pasqal unpromoted resource formula.

**Evidence hiện có.** TN raw fields t_network_build_s/t_path_search_s/t_first_contract_s/t_warm_contract_s/t_end_to_end_s; historical kernel width/precision controls; no matched predictor latency panel.

**Finding.** Target-runtime scaling có bounded evidence; estimator scalability/accuracy-cost ranking chưa có direct matched measurements. TN telemetry hỗ trợ reanalysis, không identify internal cost-model evaluation latency.

**Threats to validity.** Build/search may be shared prerequisites, cached paths and repeated query amortization; engine time versus API wall time; saturated chi 32/64 for q<=9.

**Bằng chứng thiếu.** Prediction/preparation/calibration cost per-context and amortized decision overhead; broad high-width/high-entanglement evidence.

**Ảnh hưởng đến kết luận.** Material cho full cost RQ; không chặn current per-context error comparisons hoặc native same-plan conclusions.

**Metrics và đơn vị.** Historical recorded loop min/median/max/sum và target-runtime width/bond errors có sẵn; matched estimator-stage/prediction latency, peak memory và amortized cost chưa có đầy đủ.

**Status:** PARTIALLY ANSWERED. **Action:** MINIMAL EXPERIMENTAL AMENDMENT.

**Exact source paths:** tn_telemetry (historical source; not bundled); [tn](../../analysis/tn_same_plan_slices.csv); [sim](../../analysis/simulator_metrics.csv); joint_protocol (historical source; not bundled); launch (historical source; not bundled); width_study (historical source; not bundled); [analog](../../analysis/pasqal_stage_quality.csv).

### RQ5 — Track A

**Câu hỏi.** Method nào áp dụng được với input/metadata/clock nào trước execution?

**Vì sao quan trọng.** Unavailable input và incompatible semantics có thể làm score vô nghĩa khi deploy.

**Minimum evidence.** Pinned method/input/output identity, pre-execution availability, qualification/calibration, eligibility and failure coverage, matched baseline strength.

**Dataset và observation unit.** Qualified 4.515 cohort; 230 Qonductor observations không full original one-circuit inventory; QCEC 32 pairs/11 recipes.

**Runtime target.** Observed service target versus scheduled/native/reliability/nominal output clocks.

**Evaluation protocol.** Method-source audit + same-row accuracy + frozen eligibility; selection recommendations conditional, không trained meta-selector.

**Estimators.** Learned adaptations; original QCRE/Qiskit components; Hyb component; Scholten generalized proxy; service bridges and references.

**Evidence hiện có.** Fidelity cards, 51/41/40 support audit, 257 transforms, snapshots and QCEC boundary.

**Finding.** Native timing suitable cho schedule questions, service bridges require labels. Neural omits shots/backend inputs; regression uses shots/native counters. Original Scholten unavailable, proxy không original accuracy.

**Threats to validity.** Nominal calibration and reconstructed recipe/angles/layout; historical author tensor uncertain; native component không full-paper replication.

**Bằng chứng thiếu.** Prospective deployment validation and guaranteed selector; exact historical circuits/calibration unavailable.

**Ảnh hưởng đến kết luận.** Không chặn conditional method-applicability framework; không được claim exact paper reproduction hoặc universal practical optimum.

**Metrics và đơn vị.** Eligible/assigned coverage, input/clock/fidelity class, qualification level, shared-row accuracy và quality violations; không có universal selection score hoặc deployment-cost optimum.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [registry](../../interpretation/thesis/estimator_fidelity_registry.md); parent_protocol (historical source; not bundled); profile (historical source; not bundled); qcec (historical source; not bundled); [transforms](../../interpretation/benchmark/effective_input_widths.csv); [controls](../../analysis/analytical_controls.csv).

### RQ5 — Track B

**Câu hỏi.** Lựa chọn estimator theo engine/API/precision/quality và known inputs như thế nào?

**Vì sao quan trọng.** Full-state, sampling, contraction và analog simulation là different tasks; cần estimate đúng endpoint.

**Minimum evidence.** Execution identity, pre-execution feature availability, quality/configuration policy, coverage, native/adapter fidelity; no post-run oracle inputs.

**Dataset và observation unit.** Core 150 hashes and limited frontier, QCSim 143, ladder140, TN188 unique paired hashes, analog18 cells.

**Runtime target.** Explicit per-context clocks; configured chi/cutoff known trước run, observed fidelity/bond/status không predictor inputs.

**Evaluation protocol.** Frozen per-context predictions/protocols; same-plan pairing; selection decision evaluated separately from oracle rung runtime.

**Estimators.** Graph/classical, Azizov adaptations, family-aware local adaptation, Maestro-style component, native TN, Pasqal diagnostic.

**Evidence hiện có.** Registry, context-specific metrics, ladder target definitions and safety controls.

**Finding.** Start from task/clock/required accuracy then eligible methods; retain quality violations/abstention. Maestro component không Composer; TN requires selected plan; Pasqal cannot enter predictive ranking.

**Threats to validity.** One machine/engine configurations and different survivor sets; no universal policy or production safety proof.

**Bằng chứng thiếu.** Prospective multi-engine selector and high-width quality transfer; optional for estimator benchmark, not required to reproduce closed selector.

**Ảnh hưởng đến kết luận.** Conditional applicability answered; cost-optimal/practically safe automatic selection chưa established.

**Metrics và đơn vị.** Eligible/assigned coverage, input/clock/fidelity class, qualification level, shared-row accuracy và quality violations; không có universal selection score hoặc deployment-cost optimum.

**Status:** ANSWERED WITH QUALIFICATIONS. **Action:** CLOSE.

**Exact source paths:** [registry](../../interpretation/thesis/estimator_fidelity_registry.md); [sim](../../analysis/simulator_metrics.csv); joint_protocol (historical source; not bundled); [quality](../../analysis/mps_selection_safety.csv); [tn](../../analysis/tn_same_plan_slices.csv); [analog](../../analysis/pasqal_stage_quality.csv).
