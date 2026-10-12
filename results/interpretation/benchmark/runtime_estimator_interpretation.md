> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Runtime-estimator interpretation

## Track A: separate native outputs from scheduler-oriented prediction

The evaluation target is archived observed execution/service seconds, queue
excluded under each source's declared boundary. Harmonized row units are not
proof that provider host/readout/service components are identical. The qualified
cohort is340 Ma–Li +230 Qonductor +3945 QPack,165 conservative structural groups.
QPack supplies87.4% of observations but only six structures. The reconstruction
qualification and source boundaries remain unchanged.
The 32 QCEC zero-input prepared-state/measured-output checks support the declared
230-observation recipe qualification under declared mappings; they do not prove
230 independent pairs, exact historical logical recovery or full-unitary equivalence.

`track_a_fidelity.csv` gives original definition, inputs, current component,
fidelity, native target and calibration rules. Different evaluation data alone
is not a method-fidelity failure; extraction, architecture, calibration, search
and training changes are disclosed separately.

### Learned methods and baseline value

Parent MAE/R²: Graph .83413s/.46809; MLP .89342/.36316; Qonductor budgeted
selector .88666/.41131. Graph skill is .26720 vs outer-train global median,
.14392 vs source median and .05924 vs the inner-selected classical regressor.
MLP skill is .21511/.08307/−.00763 against these same references.

The Graph minus global-median MAE CI is [−.72887,−.04188]s; Graph minus source
median [−.37198,.17473]; Graph minus selector [−.318,.458]. Thus there is clear
conditional evidence beyond the global median, not a resolved advantage over
all stronger controls. Source-balanced MAE is Graph .89803, MLP .97602,
selector .87898: Graph skill against selector becomes−.02167. Equal-group
MAE reverses Graph/MLP (.63324/.58070). This is a weighting-sensitive benchmark,
not a universal neural ranking.

The baseline ladder uses saved outer-train global median, provenance-aware
source median and circuit-statistics regressors/selector. Source median has
information absent from methods without source input; its skill is a contextual
reference, not an equal-information competition. Qonductor's five native
statistics are `swap` (the upstream CX/CZ/ECR counter, **not literal SWAP**),
operand-stack depth, touched width, shots and circuit_count=1. Six original
regressor families and 307 frozen candidates are compared; bounded search is
not full original grid reproduction. No backend/source one-hot is added.
There is no separately matched raw-width/depth-only linear QPU rung in this
saved4515 study; do not invent it or silently import another cohort's score.

Graph receives node/calibration information beyond the global-only MLP and
five-counter regressors. Parent neural gradient fits retain inner1–3 and select
on inner0, whereas regressors refit complete outer train after four inner folds.
RQ4 neural models freshly refit complete outer train. Shared test rows do not
mean identical information or training procedures. These differences constrain
attribution, not the validity of comparing the declared complete pipelines.

### QCRE and Qiskit

QCRE invokes the pinned `estimate_runtime` component on admitted BQSKit
unitary circuits and location-specific nominal duration maps. It is more than
a home-written depth substitute, but not full QCRE compilation/evaluation
reproduction. Terminal measurement/barriers are omitted from this component;
unsupported nonunitary/classical semantics remain explicit. Qiskit uses its
frozen nominal Target schedule/duration semantics, potentially including
measurement/reset. The two raw clocks are not assumed identical.

One-shot unitary/scheduled duration, deterministic `shots × duration`, and
outer-train affine/log-affine service-time predictions are separate outputs.
Shot multiplication assumes repeated circuit cost and omits inter-shot reset,
readout/classical processing, batching and host/provider overhead unless those
terms are explicitly included. It is not validated observed-time extrapolation.
The [Qiskit API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit#estimate_duration)
provides a timing reference, not a provider-service prediction model.

Raw one-shot MAE is about 6.116s for both components on 4515, not evidence of
equally accurate service prediction. Shot-scaled affine MAE is QCRE1.05127s
and Qiskit1.04970s versus same-row source median .97436s. Calibration fits
eligible outer-train labels only and is an additional predictor, not original
estimator behavior or identification of physical overhead.

### Scholten: original not evaluable; proxy separately useful to assess

The [original paper](https://arxiv.org/html/2307.04980) defines
`D_eff = E[depth(U†(x)U(y))] / E[depth(QV(v,v))] × v`,
`v = ceil(sqrt(2 D n))`, and `T = M × S × D_eff / C`.
Kernel template repetitions D, encoding width n, transpiled parameter/QV
ensemble expectations and compatible throughput are required. Ordinary
compiled wire depth is not that D_eff. None of the qualified4515 rows has
all verified original-method semantics: **original not evaluable**.

The frozen `scholten_unified_nominal_throughput` route uses generic compiled
wire depth and nominal throughput. Reader label: **Scholten-inspired generalized
proxy**, not original-style reproduction or a newly invented effective depth.
It predicts3399 rows;1116 have no admitted throughput. Its raw MAE610.288s
falls to .74835 affine/.72660 log-affine, versus eligible train median .82007s
and source median .69207s. The calibrated proxy improves on the former, not
the latter (which has source provenance unavailable to the proxy). Much of the apparent correction can come from
label/context fitting; native throughput accuracy is not established. Historical
same-width QV-normalized extensions are different studies and not these4515
predictions. No CLOPS_h-to-v conversion or calibration-to-CLOPS inference is made.

### Hyb-HANAS

The [paper's cost component](https://arxiv.org/html/2603.00625v1) combines gate
timing/routing and divides by a survival probability including noise/decoherence;
its full workflow additionally scales parameter-gradient/training work. A
survival-adjusted cost estimates useful-work resource demand, not literal time
taken by one circuit execution. A numerical seconds unit does not fix this
semantic difference. The local identifiable component is not the entire NAS model.

Finite raw cost covers 4175 rows;192 Osaka observations overflow exponentiation,
148 Kyoto observations are unavailable. The separately named log-cost Ridge
can use finite log costs for 4367 rows, still excluding148. Undefined/overflow
results are not zeros. Raw finite cost MAE5.361s; shot-linear effective cost
MAE2600.514s. Affine/log-affine cost MAE1.19770/1.16836s is worse than eligible
intercept1.06955s and source median .93555s. The log-cost Ridge's .98642s on
4367 is a separately trained adaptation on a different successful set.
These results do not validate reliability-adjusted cost as direct service time.

## Track B: context-specific answers, not a universal leaderboard

Use `simulator_predictive_skill.csv`, existing paired intervals and
`benchmark_question_matrix.csv`. Seconds below remain context-specific.

| Context and operation | Saved finding | Scope/limitation |
| --- | --- | --- |
| CUDA-Q dense FP32/FP64 warm sample,150 hashes each | RF MAE .821/.813ms; skill vs median .530/.536. Graph1.716/1.730ms; MLP1.578/1.590ms | Tabular RF strongest of these; Graph near median. No topology-only claim |
| CUDA-Q FP64 fixed chi16 warm get_state,150 hashes | Graph .29609s vs Ridge1.27567 and median1.87060; skill .768/.842 |147 pass quality,3 fail; fast completion does not imply adequate state |
| Noisy Aer FakeSherbrooke Opt1 shots1024,150 hashes | GNN source/hybrid/transpiled .44966/.40636/.30076s; compiled XGBoost .29704s | Source→transpiled delta CI excludes0; hybrid→transpiled and GNN→XGB intervals include0 |
| Joint family-aware MPS chi ladder,840 rungs/140 hashes | Conditioned .31163s vs matched agnostic .30752; family-held-out .31101/.30879 | No resolved runtime increment.64/140 selected-quality failures +10 unassessed despite138/140 having a passing rung |
| Maestro-style QCSim CPU FP64 shots1000,143 hashes | Component .10370ms vs Ridge .13361 and median .30578; skill .224/.661 | Engine-reported execution across fresh processes, optimizer off; not persistent warm wall-clock or full Composer |
| cuTensorNet same-plan scalar contraction |603/612 plan rows,201 members/188 hashes; median EST/warm3.734 (core4.132) | Native all-slice estimate only; no simple learned reference, no general simulator API claim |
| Pasqal analog MPS/TDVP companion |18 cells,17 quality pass; chi16 quench fails | Analog clock/workload separate; work-term coefficients not identified/promoted |

The Aer result supports useful compiled representation locally and shows a
strong classical model matching the transpiled GNN, not all GNNs being inferior
or complete Azizov encoding reproduction. Family conditioning can help a
quality score without yielding safe operational quality selection.

Historical PyTorch dense208-row evidence found grouped HGB log-R² .9794, but
width holdout analytical Ridge .8754 vs HGB .7295. That is a distinct kernel,
not a CUDA-Q common-panel extrapolation result. The saved CUDA-Q RX pilot
likewise shows a width-dependent zero-work API baseline: q30 FP32 zero-gate
warm34.2471ms vs q16 .0967ms. Width-aware leave-q30-out MAE14.905ms improves
on work-only 42.297ms, but does not identify a physical launch coefficient or
a general simulator scaling law. No new measurement was performed.

## Evidence and uncertainty

All skill calculations use exact shared test IDs, unchanged targets and saved
predictions. `skill = 1 − MAE_method / MAE_reference`; zero means equal MAE,
negative means worse than reference. Equal-source skill is the ratio of macro
MAEs, not the mean of source ratios or R². Eligible-subset analytical controls
use their own frozen fit populations; compare their exact shared rows.

Existing paired group-bootstrap intervals quantify conditional error
differences, not seed-selection/retraining uncertainty or prediction-interval
coverage. Correlation/rank metrics do not make raw timing a service-time
estimator. No missing outcome is filled with zero and no failed/unsupported
case is discarded to create an all-method winner. Paper attribution, admitted
reconstruction and nominal hardware context remain explicit.
