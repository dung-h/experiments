# Comparison with the original papers

The completed benchmark tests local adaptations of published ideas. It supports
claims about those implemented pipelines on their declared held-out rows; it does
not establish numerical reproduction of the papers' scores. The strongest current
QPU result is a lower MAE for the graph model than its matched global MLP. The
graph–polynomial comparison is unresolved, and the simulator findings depend on
representation, engine, clock and quality target.

Local numbers below come from the independently checked
[scientific review](scientific_review.md), [results](results.md), and the separate
[QPU](../results/real_qpu/method_comparison.csv) and
[simulator](../results/simulator/method_comparison.csv) reader tables.
Paper mechanisms and original targets are linked directly to primary sources.
The current QPU comparison contains 7,350 archived observations, rather than
7,350 distinct circuits. The simulator predictor core contains 150 distinct
exact-QASM hashes at q2–q9. Neither population matches every original paper.

## What each local result tests

| Method or route | Published mechanism and original target | What the local implementation keeps and changes | Completed local evidence | Supported conclusion and limit |
| --- | --- | --- | --- | --- |
| Ma–Li graph transformer | Combines a circuit graph with global features to predict execution time on simulators and IBM QPUs. The original study selects 340 QPU samples through active learning and reports average QPU R² above 0.90 in its own evaluation. See [Ma and Li, §§4–6](https://arxiv.org/html/2411.15631v2) and the [author artifact](https://github.com/mooselab/Quantum-Execution-Time-Prediction). | Retains graph-plus-global prediction and the source feature schema. The current unified adaptation uses logical inputs, 51 raw global positions and 178 node positions, nominal T1/T2 context, grouped nested validation and three-seed median predictions. It combines three archived sources and reconstruction tiers; it does not repeat active-learning acquisition or recover job-day calibration. | On 7,350/7,350 rows, graph MAE is 1.2186 s, R² 0.4291; matched global MLP MAE is 1.7322 s, R² 0.1873. Graph-minus-MLP MAE is −0.5136 s, 95% grouped interval [−0.9074, −0.1748]. | The full graph/node-context branch adds predictive value over this matched global-only MLP on this panel. It does not isolate topology from node features or T1/T2, establish unseen-family/backend transfer, or reproduce the paper's R². The separate source-native 340-row result is not a unified-panel method. |
| Qonductor polynomial | Uses regression over circuit/shot features as part of a resource estimator for hybrid cloud planning, including error mitigation. The paper reports execution-time R² 0.998 under its own cross-validation; scheduling and workflow objectives extend beyond a one-circuit label. See [Giortamis et al., §6](https://arxiv.org/html/2408.04312v2) and the [author artifact](https://github.com/manosgior/Qonductor-SC25). | Retains the upstream five-input polynomial route: CX/CZ/ECR count under the field name `swap`, operand-stack depth, touched width, shots and circuit count fixed to one. Uses compiled/submitted inputs and nested grouped degree selection on the unified panel; every outer fold selects degree two. No source-wide verified mitigation setting is available. | On 7,350/7,350 rows, MAE is 1.4006 s, R² −0.5867. Graph-minus-polynomial MAE is −0.1819 s, interval [−0.4883, +0.1968]. Polynomial has lower source-slice MAE on Ma–Li and Qonductor; graph has lower MAE on QPack. | There is no resolved pooled MAE advantage between these two pipelines. Negative R² describes squared-error performance and tail sensitivity on the local target; it neither invalidates polynomial regression nor refutes Qonductor's original estimator, scheduler or mitigation-aware resource planning. |
| Azizov transpilation-aware GNN | Compares source, hybrid and transpiled graph views with conventional regressors for noisy Aer runtime after transpilation. The paper uses two fake-backend configurations and four optimization levels and finds that graph benefits depend on the setting. See [Azizov et al., §§3–5](https://arxiv.org/html/2609.12980v1). | Retains the three representation views and graph-versus-tabular comparison. Fits a local common-core adaptation with its own hardware, noise/configuration context, exact-hash folds and three neural seeds; it does not reproduce the full two-backend/four-level experiment. | On the same 150 hashes, transpiled GNN MAE is 0.3008 s, hybrid GNN 0.4064 s, source GNN 0.4497 s; transpiled XGBoost is 0.2970 s. Transpiled-GNN-minus-XGBoost MAE is +0.0037 s and its paired interval includes zero. | Post-transpilation information is useful in this local Aer context. A general GNN advantage over regression is not established. Source versus transpiled compares complete representation pipelines, not topology alone, and these Aer labels are not QPU execution times. |
| Scholten CLOPS model | Estimates job execution time using circuit count, shots, system CLOPS and an effective number of quantum-volume layers. The effective-depth definition and validation concern quantum-volume and quantum-kernel jobs through Qiskit Runtime. See [Scholten et al., §§2–3](https://arxiv.org/html/2307.04980v1). | Retains throughput/depth/shot reasoning but substitutes admissible nominal same-backend CLOPS-like context and a compiled-depth proxy for missing historical CLOPS and paper effective depth. Sets the single-circuit job factors to one. Outer-train affine calibration is a separate learned service-time adaptation. | The current nominal affine adaptation predicts 5,199/7,350 rows, with successful-subset MAE 1.364 s and R² 0.708. The other 2,151 observations remain assigned and unavailable. | This quantifies a nominal single-circuit sensitivity adaptation. It cannot validate the original effective-depth construction or historical throughput. Its successful-subset score is not a full-panel win; CLOPS_h is not converted into CLOPS_v. |
| QCRE and Qiskit scheduled duration | QCRE accumulates device gate durations over circuit dependencies; its author repository requires a device-compiled circuit and a native-gate duration map. The associated paper studies gate-aware depth and compiled-circuit runtime comparison. See [QCRE](https://github.com/mtkgv/qcre) and [Tremba, Hovland and Liu](https://arxiv.org/html/2505.16908v3). | Uses pinned nominal same-backend duration maps. Keeps raw single-shot schedule, shot-scaled schedule and outer-train service-time calibration as distinct outputs. The exact original gate-aware-depth ranking task is not the current benchmark task. | QCRE and Qiskit single-shot values agree to floating-point precision under the frozen Target. The shot-scaled QCRE affine adaptation scores 7,350/7,350 rows at MAE 1.555 s, R² 0.675. | Agreement checks the two scheduling implementations under the same inputs. It does not provide two independent predictive wins. Calibration supports a learned service-time pipeline; raw scheduled duration is not observed provider service time or a job-day reconstruction. |
| Hyb-HANAS cost model | Models calibrated gate cost, routing and reliability penalties within a hybrid neural architecture search objective. Its published validation compares the gate-time component with Qiskit scheduling; effective cost and HQNN training cost have broader multipliers. See [Kashif, Marchisio and Shafique, §§2.4 and 4.1](https://arxiv.org/html/2603.00625v1). | Retains identifiable one-circuit gate/cost terms, with declared aggregation and nominal calibration. Omits the full NAS/training workflow. Current log-cost/gate-time Ridge variants learn service-time calibration inside the new outer splits; these are not the raw analytical cost. | Nominal log-cost Ridge predicts 7,202/7,350 rows, MAE 1.370 s. Kyoto composite log-cost and gate-time Ridge variants predict 7,350/7,350 rows, MAEs 1.369/1.371 s. Earlier raw effective-cost overflow/zero-survival evidence remains historical. | Cost-derived features can be evaluated as service-time predictors after calibration. These scores do not validate NAS quality, HQNN training cost, reliability penalties or raw effective-cost units. Nominal snapshot anomalies and missing support prevent attributing historical raw failures solely to the paper's mechanism. |
| Family-Aware residual architecture | Uses a learned family classifier, family-conditioned residual corrections and FiLM to jointly predict approximation threshold and runtime. The paper's Quantum Rings target uses fidelity ≥0.75; its 36-circuit evaluation spans CPU/GPU and single/double precision. See [Xing et al., §§3–4](https://arxiv.org/html/2606.11620v1). | Retains family conditioning and joint runtime/quality prediction, with a fold-local classifier and a matched family-agnostic ablation. Changes engine to CUDA-Q MPS FP64, target to warm state-producing `get_state()`, quality to dense-reference state fidelity ≥0.99 and approximation grid to six configured bond dimensions on q2–q9. | Of 900 assigned hash×rung rows, 840 have complete measured labels. Family-conditioned/agnostic runtime MAEs are 0.31163/0.30752 s; delta +0.00411 s, interval [−0.02660, +0.03923]. Brier scores are 0.08159/0.08631; selected-rung quality violations are 64/140, or 45.7%. | A runtime benefit from family conditioning is unresolved; the rung quality probabilities improve modestly under the paired comparison, but selected configurations often fail the local quality gate. The local selector is not dependable. This does not reproduce the paper's threshold accuracy or establish cross-simulator transfer. |
| Maestro component model | Sums operation complexities with hardware-calibrated coefficients; uses width-dependent statevector curves, MPS width/bond surfaces and a shot-sampling term to select among simulators. See [Bertomeu et al., §3.C](https://arxiv.org/html/2512.04216v1). | Retains local synthetic component calibration and operation-cost summation. The completed follow-up uses CPU QCSim FP64, 1,000 shots and circuit optimization disabled. It tests one component estimator, not the full published backend selector; the earlier optimizer-on pilot remains failed. | On 143 supported/150 assigned hashes, component MAE is 0.0001037 s, R² 0.9490; nested Ridge MAE is 0.0001336 s. Component-minus-Ridge MAE is −0.0000299 s, interval [−0.0000625, −0.0000016]. Labels pool 15 finite engine-reported samples per hash, each from a fresh process. | The component formula improves over the matched Ridge on this narrow CPU cell. It does not reproduce automatic backend-selection accuracy, persistent warm-process execution, process wall-clock, or a physical launch-overhead intercept. The original Composer estimator and fitted artifacts are not available in the pinned public checkout. |
| Pasqal EMU-MPS resource formula | Official documentation derives analog two-site TDVP runtime contributions proportional to N²χ³ and N³χ², with coefficients dependent on the workload and implementation. Bond/Krylov limits constrain memory and approximation error. See [Pasqal resource estimation](https://pasqal-io.github.io/emulators/v2.4.0/emu_mps/advanced/resource_estimation/). | Retains the documented complexity form and tests a separately calibrated local analog companion. Analog evolution is not converted into a generic digital circuit label. Candidate and reference quality thresholds remain explicit. | Eighteen analog cells: 17 quality-pass, one quality-fail. The `quench\|4x4\|chi16` cell has fidelity 0.95147 against candidate threshold 0.99; four failed stage/repetition records belong to that one cell. Reference threshold is 0.9999. The fitted formula is unpromoted. | This provides bounded analog timing/quality evidence. It does not establish generic digital-panel predictor accuracy or a validated scaling coefficient; repeated records do not represent four independent failed cells. |
| cuTensorNet native runtime estimate | NVIDIA documents experimental `RUNTIME_EST` as a time-objective estimate in seconds for one contraction pass over all slices. See [NVIDIA API types](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornetcontractionoptimizerinfoattributes-t) and the [26.06 binding](https://docs.nvidia.com/cuda/cuquantum/26.06.0/python/bindings/generated/cuquantum.bindings.cutensornet.ContractionOptimizerInfoAttribute.html). | Calls the native estimator for the selected scalar-contraction plan on the local GPU and pairs it with that same plan's warm contraction. Network building, path search and end-to-end simulation remain separate clocks. | There are 603 successful/612 attempted same-plan sessions. Median estimate/warm ratios are 3.7340 on core and 4.1324 on frontier. | The native estimate overpredicts matching warm contraction in these cells. This diagnoses the selected-plan estimate under the local context; it does not estimate noisy Aer, dense sampling or complete simulator wall-clock. |
| Dense work-scaled Ridge and classical controls | Internal empirical comparison; no paper-exact or native NVIDIA runtime model is claimed. Complexity-inspired operation counts are motivated by statevector work, but their fitted coefficients are benchmark constructs. See the [local protocol](../protocol/cudaq_dense_statevector_predictor_comparison.json). | Tests nine fixed predictors separately on CUDA-Q FP32/FP64 32-shot warm `sample()` labels. Work-scaled Ridge uses 2^n, width-scaled gate counts and shots. Its intercept is empirical; no independent kernel launch measurement identifies it. | Every predictor scores 150/150 hashes in each precision. Random Forest has lowest MAE, 0.000821/0.000813 s for FP32/FP64; work-scaled Ridge has highest R², 0.8728/0.8649. | Classical empirical controls outperform the neural adaptations on MAE here. Highest R² and lowest MAE identify different methods. q2–q9 fits do not validate asymptotic exponential scaling, width extrapolation, universal coefficients or a physical launch-overhead law. |

## Reading the comparisons scientifically

The current QPU graph–MLP comparison is the closest local test of an incremental
architectural contribution: both consume the same transformed global schema,
use the same outer folds and neural selection procedure, and differ through the
full graph/node branch. That branch also supplies node context, including nominal
T1/T2. Its improvement therefore supports the complete added branch, rather
than an isolated causal effect of graph topology. The comparison with polynomial
is between complete pipelines with different input representations, input counts
and model-selection procedures. It answers a practical prediction question on
the shared observations but cannot isolate architecture alone.

MAE and R² emphasize different parts of the error distribution. In the current
panel, polynomial p99/max absolute error is 4.0035/187.0537 s, compared with
graph 7.0822/76.0941 s. A lower typical error can coexist with a rare extreme
error and a negative R². Source-balanced metrics, source slices and tails must
accompany pooled scores; all source slices reuse the unified fits and are not
independent transfer experiments. The [current QPU guide](real_qpu_benchmark.md)
also reports the reconstruction sensitivity subset, which reuses full-panel
fits rather than defining another trained benchmark.

On the FakeSherbrooke/Opt1 Aer cell, source-view GNN MAE exceeds transpiled-view
GNN MAE by 0.1489 s, with paired interval [+0.0284, +0.3039]. This is consistent
with the Azizov paper's emphasis on
information about the circuit that the simulator actually executes, but the
local GNN–XGBoost comparison remains unresolved. Transpilation can change
structural features and node inputs together. This result does not show that
source circuits universally lack runtime information or that a GNN is always
needed once transpiled features are available.

Family-conditioned MPS runtime and quality must be assessed separately. The
joint adaptation's Brier-score difference is −0.00473, with paired interval
[−0.00850, −0.00082], while its runtime interval includes zero. Better rung
probabilities do not guarantee a successful decision at the selection threshold:
45.7% of selected rungs violate the local fidelity condition. The separate
fixed-χ=16 graph model has MAE 0.2961 s over 150 finite labels and 0.2908 s over
147 quality-pass labels; that fixed-configuration runtime score is not a test of
joint minimum-rung selection. At q2–q9, the maximum possible Schmidt rank is at
most 16, so configured χ=32/64 test saturated small-circuit settings, not
large-bond asymptotic behavior. See the
[joint runtime/quality contract](../protocol/family_aware_joint_runtime_quality_mps.json).

Scheduled gate duration, effective hardware cost and nominal throughput time
are distinct from the archived provider service-time label. Their raw errors
against that label diagnose a proposed clock bridge. Fitting an outer-train
affine or Ridge bridge creates a new prediction pipeline and must be reported as
such. A successful-subset MAE from Scholten or nominal Hyb cannot be compared
directly with a full-panel graph/polynomial score; comparisons require identical
successful held-out rows and the full assigned denominator. Scheduler agreement
also cannot validate reliability penalties that the scheduler does not model.

The original papers' reported R², threshold accuracy, scheduling gains and
backend-selection accuracy are different endpoints under different circuits,
splits, hardware and timing boundaries. They are context for the mechanisms,
not numerical equivalence targets for these adaptations. A lower local score
does not by itself contradict a paper, and a higher local score does not
replicate or exceed its claim. The present evidence is retrospective and
exploratory: grouped intervals condition on saved fits, omit refitting
uncertainty and are pointwise rather than multiplicity-adjusted. The
[reproduction guide](reproduction.md) verifies saved-evidence/table replay;
it does not claim fresh paper-level fitting or simulator retiming.
