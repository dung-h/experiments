# Comparison with the original papers

The QPU benchmark evaluates declared implementations on a shared dataset of
4,515 archived observations and fresh grouped folds. Execution and independent
saved-evidence QA are complete. The comparisons below distinguish QPU service-time
prediction from local simulator clocks. Changing the dataset does
not by itself change a method, but altered representation, nominal context,
bounded search or calibration must be disclosed.

Paper mechanisms and targets are linked to primary sources. Local simulator
numbers come from [scientific review](scientific_review.md) and the
[simulator reader table](../results/simulator/method_comparison.csv).
QPU input evidence is defined in the [dataset report](data_preprocessing.md).
QPU numbers are in the [method table](../results/real_qpu/method_comparison.csv)
and [same-row comparisons](../results/real_qpu/shared_row_comparisons.csv).

## What each local evaluation tests

| Method or route | Published mechanism and original target | What the local implementation keeps and changes | Local evidence / status | Supported conclusion and limit |
| --- | --- | --- | --- | --- |
| Ma–Li graph transformer | Combines a circuit graph with global features to predict execution time on simulators and IBM QPUs. The original study selects 340 QPU samples through active learning and reports average QPU R² above 0.90 in its own evaluation. See [Ma and Li, §§4–6](https://arxiv.org/html/2411.15631v2) and the [author artifact](https://github.com/mooselab/Quantum-Execution-Time-Prediction). | Retains the upstream graph-plus-global architecture, 51 raw globals and 178 node positions. Uses source/qualified logical inputs, nominal-index T1/T2, fresh grouped validation and three-seed median predictions. It does not repeat active-learning acquisition or recover job-day calibration. | Graph MAE 0.8341 s / R² 0.4681; matched MLP 0.8934 s / 0.3632 on all 4,515 observations. MLP-minus-graph MAE +0.0593 s, interval [−0.4846,+0.3291]. | Complete graph/node branch beyond matched globals has a lower point estimate, but a robust advantage is unresolved. It does not isolate topology alone or numerically reproduce the paper's original score. |
| Qonductor polynomial | Uses regression over circuit/shot features as part of a resource estimator for hybrid cloud planning, including error mitigation. The paper reports execution-time R² 0.998 under its own cross-validation; scheduling and workflow objectives extend beyond a one-circuit label. See [Giortamis et al., §6](https://arxiv.org/html/2408.04312v2) and the [author artifact](https://github.com/manosgior/Qonductor-SC25). | Retains the five-input route: CX/CZ/ECR counter named `swap`, operand-stack depth, touched width, shots and circuit count = 1. Six regressor families use bounded grouped inner searches on compiled/submitted inputs; mitigation metadata is unknown. | Polynomial MAE 0.9177 s / R² 0.4024; inner-selected family 0.8867 s / 0.4113 on all 4,515. Selector source-balanced MAE 0.8790 s; graph 0.8980 s. | Regression remains competitive. Bounded search and nominal compiled reconstructions are disclosed; no original-paper score or scheduler claim is inferred. Family selection uses inner scores, not these test errors. |
| Azizov transpilation-aware GNN | Compares source, hybrid and transpiled graph views with conventional regressors for noisy Aer runtime after transpilation. The paper uses two fake-backend configurations and four optimization levels and finds that graph benefits depend on the setting. See [Azizov et al., §§3–5](https://arxiv.org/html/2609.12980v1). | Retains the three representation views and graph-versus-tabular comparison. Fits a local common-core adaptation with its own hardware, noise/configuration context, exact-hash folds and three neural seeds; it does not reproduce the full two-backend/four-level experiment. | On the same 150 hashes, transpiled GNN MAE is 0.3008 s, hybrid GNN 0.4064 s, source GNN 0.4497 s; transpiled XGBoost is 0.2970 s. Transpiled-GNN-minus-XGBoost MAE is +0.0037 s and its paired interval includes zero. | Post-transpilation information is useful in this local Aer context. A general GNN advantage over regression is not established. Source versus transpiled compares complete representation pipelines, not topology alone, and these Aer labels are not QPU execution times. |
| Scholten CLOPS model | Estimates job execution time using circuit count, shots, system CLOPS and an effective number of quantum-volume layers. The effective-depth definition and validation concern quantum-volume and quantum-kernel jobs through Qiskit Runtime. See [Scholten et al., §§2–3](https://arxiv.org/html/2307.04980v1). | Uses nominal same-backend throughput and compiled wire-depth proxy when original effective-depth/template inputs are absent. Single-circuit factors are fixed; outer-train calibration is a separately labeled learned pipeline. | Nominal coverage 3,399/4,515; log-affine MAE 0.7266 s / R² 0.3780 on that subset. Original effective-depth route 0/4,515. | Lower partial-coverage error is not a full-panel win. Original effective-depth and historical throughput remain unverified. No CLOPS_h/v conversion. |
| QCRE and Qiskit scheduled duration | QCRE accumulates device gate durations over circuit dependencies; its author repository requires a device-compiled circuit and a native-gate duration map. The associated paper studies gate-aware depth and compiled-circuit runtime comparison. See [QCRE](https://github.com/mtkgv/qcre) and [Tremba, Hovland and Liu](https://arxiv.org/html/2505.16908v3). | QCRE calls the pinned original estimator through BQSKit on the compiled/submitted unitary view with nominal native durations; final measurement/barriers are omitted. Qiskit uses its declared schedule semantics. Raw, shot-scaled and fitted service-time outputs remain separate. | Both cover 4,515/4,515. Shot-scaled affine QCRE MAE 1.0513 s / R² 0.2661; Qiskit 1.0497 s / 0.2686. All raw/calibrated variants remain reported. | Same-observation evaluation compares their clocks with provider labels. A fitted service-time bridge is not the original duration estimator or proof of historical compilation. |
| Hyb-HANAS cost model | Models calibrated gate cost, routing and reliability penalties within a hybrid neural architecture search objective. Its published validation compares the gate-time component with Qiskit scheduling; effective cost and HQNN training cost have broader multipliers. See [Kashif, Marchisio and Shafique, §§2.4 and 4.1](https://arxiv.org/html/2603.00625v1). | Retains identifiable nominal one-circuit cost/gate terms and explicitly fitted service-time bridges; does not implement the full NAS/training workflow or select snapshots using test error. | Raw cost coverage 4,175/4,515; log-cost Ridge 4,367/4,515, MAE 0.9864 s / R² 0.2867. Log representation retains Osaka overflow inputs; Kyoto zero-survival remains failed. | The identifiable component can be tested; HQNN/NAS objectives and historical calibration are not reproduced. Coverage and identical-row comparisons are required. |
| Family-Aware residual architecture | Uses a learned family classifier, family-conditioned residual corrections and FiLM to jointly predict approximation threshold and runtime. The paper's Quantum Rings target uses fidelity ≥0.75; its 36-circuit evaluation spans CPU/GPU and single/double precision. See [Xing et al., §§3–4](https://arxiv.org/html/2606.11620v1). | Retains family conditioning and joint runtime/quality prediction, with a fold-local classifier and a matched family-agnostic ablation. Changes engine to CUDA-Q MPS FP64, target to warm state-producing `get_state()`, quality to dense-reference state fidelity ≥0.99 and approximation grid to six configured bond dimensions on q2–q9. | Of 900 assigned hash×rung rows, 840 have complete measured labels. Family-conditioned/agnostic runtime MAEs are 0.31163/0.30752 s; delta +0.00411 s, interval [−0.02660, +0.03923]. Brier scores are 0.08159/0.08631; selected-rung quality violations are 64/140, or 45.7%. | A runtime benefit from family conditioning is unresolved; the rung quality probabilities improve modestly under the paired comparison, but selected configurations often fail the local quality gate. The local selector is not dependable. This does not reproduce the paper's threshold accuracy or establish cross-simulator transfer. |
| Maestro component model | Sums operation complexities with hardware-calibrated coefficients; uses width-dependent statevector curves, MPS width/bond surfaces and a shot-sampling term to select among simulators. See [Bertomeu et al., §3.C](https://arxiv.org/html/2512.04216v1). | Retains local synthetic component calibration and operation-cost summation. The completed follow-up uses CPU QCSim FP64, 1,000 shots and circuit optimization disabled. It tests one component estimator, not the full published backend selector; the earlier optimizer-on pilot remains failed. | On 143 supported/150 assigned hashes, component MAE is 0.0001037 s, R² 0.9490; nested Ridge MAE is 0.0001336 s. Component-minus-Ridge MAE is −0.0000299 s, interval [−0.0000625, −0.0000016]. Labels pool 15 finite engine-reported samples per hash, each from a fresh process. | The component formula improves over the matched Ridge on this narrow CPU cell. It does not reproduce automatic backend-selection accuracy, persistent warm-process execution, process wall-clock, or a physical launch-overhead intercept. The original Composer estimator and fitted artifacts are not available in the pinned public checkout. |
| Pasqal EMU-MPS resource formula | Official documentation derives analog two-site TDVP runtime contributions proportional to N²χ³ and N³χ², with coefficients dependent on the workload and implementation. Bond/Krylov limits constrain memory and approximation error. See [Pasqal resource estimation](https://pasqal-io.github.io/emulators/v2.4.0/emu_mps/advanced/resource_estimation/). | Retains the documented complexity form and tests a separately calibrated local analog companion. Analog evolution is not converted into a generic digital circuit label. Candidate and reference quality thresholds remain explicit. | Eighteen analog cells: 17 quality-pass, one quality-fail. The `quench\|4x4\|chi16` cell has fidelity 0.95147 against candidate threshold 0.99; four failed stage/repetition records belong to that one cell. Reference threshold is 0.9999. The fitted formula is unpromoted. | This provides bounded analog timing/quality evidence. It does not establish generic digital-panel predictor accuracy or a validated scaling coefficient; repeated records do not represent four independent failed cells. |
| cuTensorNet native runtime estimate | NVIDIA documents experimental `RUNTIME_EST` as a time-objective estimate in seconds for one contraction pass over all slices. See [NVIDIA API types](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornetcontractionoptimizerinfoattributes-t) and the [26.06 binding](https://docs.nvidia.com/cuda/cuquantum/26.06.0/python/bindings/generated/cuquantum.bindings.cutensornet.ContractionOptimizerInfoAttribute.html). | Calls the native estimator for the selected scalar-contraction plan on the local GPU and pairs it with that same plan's warm contraction. Network building, path search and end-to-end simulation remain separate clocks. | There are 603 successful/612 attempted same-plan sessions. Median estimate/warm ratios are 3.7340 on core and 4.1324 on frontier. | The native estimate overpredicts matching warm contraction in these cells. This diagnoses the selected-plan estimate under the local context; it does not estimate noisy Aer, dense sampling or complete simulator wall-clock. |
| Dense work-scaled Ridge and classical controls | Internal empirical comparison; no paper-exact or native NVIDIA runtime model is claimed. Complexity-inspired operation counts are motivated by statevector work, but their fitted coefficients are benchmark constructs. See the [local protocol](../protocol/cudaq_dense_statevector_predictor_comparison.json). | Tests nine fixed predictors separately on CUDA-Q FP32/FP64 32-shot warm `sample()` labels. Work-scaled Ridge uses 2^n, width-scaled gate counts and shots. Its intercept is empirical; no independent kernel launch measurement identifies it. | Every predictor scores 150/150 hashes in each precision. Random Forest has lowest MAE, 0.000821/0.000813 s for FP32/FP64; work-scaled Ridge has highest R², 0.8728/0.8649. | Classical empirical controls outperform the neural adaptations on MAE here. Highest R² and lowest MAE identify different methods. q2–q9 fits do not validate asymptotic exponential scaling, width extrapolation, universal coefficients or a physical launch-overhead law. |

## Reading the comparisons scientifically

The QPU graph–MLP comparison holds global inputs, outer folds and
neural selection procedure constant while adding the complete graph/node
branch, including nominal T1/T2. It does not isolate graph topology alone.
Graph versus regression compares complete pipelines rather than identical
feature sets. Source-balanced errors, source slices and tails must accompany
pooled scores, because QPack supplies 87.4% of observations but only six groups.

MAE and R² emphasize different error properties. Both must be reported on the
same successful rows, with coverage and tail error. Source slices reuse unified
fits rather than defining separate source benchmarks. The QPU graph–MLP and
graph–selector intervals include zero; point-estimate order is not a resolved
advantage. The source-balanced ranking also differs from the pooled ranking.

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
[reproduction guide](reproduction.md) describes public file verification and
saved simulator-score checks. It does not claim QPU score recomputation from
excluded inputs, fresh paper-level fitting or simulator retiming.
