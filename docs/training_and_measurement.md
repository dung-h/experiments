# Training and measurement

This guide describes how the saved benchmark observations and predictions were
produced. The supported reader workflow is [saved-evidence verification](reproduction.md):
it checks file integrity and simulator metrics using retained labels and
out-of-fold predictions, requires no GPU or QPU account, and performs no training
or simulator measurement. It cannot recompute QPU scores from this public selection.
The procedures below
are execution specifications for separately authorized refits or new measurements.
They do not turn source-local scores, analytical clocks and different simulator
engines into one leaderboard.

The [data dictionary](data_and_feature_dictionary.md) defines the fields and
representation stages. The [methodology](methodology.md) defines comparisons;
[results](results.md) reports the measured outcomes. Protocol status fields may
describe a design-time state. Completion is established by attempt ledgers and
run manifests, not by the existence of a protocol or a runnable script.

## Frozen populations and identities

The QPU target is archived one-circuit service/execution time in seconds. Its
7,350-observation common panel contains 340 Ma–Li observations, 3,065 Qonductor
observations and 3,945 QPack observations. Ma–Li has exact logical QASM; the
Qonductor subset contains 230 archive-supported recipe adaptations and 2,835
candidate-recipe sensitivity rows; QPack uses six structural QAOA templates with
representative angles. Reconstruction provides inputs, never replacement runtime
labels. Keep recorded shots: Ma–Li uses 1,024, Qonductor varies from 1 to 20,000,
and QPack uses 4,096. QPack milliseconds are converted to seconds. Provider
service boundaries and missing historical calibration remain qualified by source.

The panel manifest (not bundled),
outer assignments (not bundled),
inner assignments (not bundled) and
target ledger (not bundled) are the
authority for these identities. The archived 8,767-observation ledger and earlier
compiled-input comparison are separate populations. Excluded bridge rows still
contribute equality links to transitive leakage groups; removing a bridge from
the panel does not sever its identity link. There are 206 groups: 148 Ma–Li, 52
Qonductor and six QPack groups. Source slices and the 4,515-row reconstruction
sensitivity subset reuse full-panel fits; they are not new training runs.

The digital simulator [member manifest](../data/simulator/circuits/sim_common_q16_manifest.csv)
selects existing Ma–Li source QASM at the revision in
[the upstream lock](../provenance/upstream.json). It contains 204 members,
191 exact-QASM SHA-256 hashes, 13 duplicate-hash groups and 22 families. Its core
has 162 members / 150 hashes at widths 2–9; the frontier has 42 members at widths
10–16. Members are filenames; hashes are deduplicated prediction observations.
The same exact hash and every alias share one fold. The frozen simulator folds
contain 37, 28, 26, 25 and 34 hashes. Frontier circuits do not contribute to core
training, preprocessing, vocabulary, model selection or prediction accuracy.

No circuits were generated to fill missing family/width combinations. Every width
2–16 occurs somewhere, but family support is irregular. The following absent
widths are obtained directly from the member manifest; they are absent inputs,
not failed simulator executions.

| Family | Absent widths within 2–16 |
| --- | --- |
| ae | 11, 13–16 |
| dj | 8, 12–15 |
| ghz | 16 |
| graphstate | 2, 10, 11, 14–16 |
| grover-noancilla | 9–16 |
| grover-v-chain | 6, 8, 10–16 |
| portfolioqaoa, portfoliovqe | 2, 11–16 |
| qaoa | 2, 15, 16 |
| qft | 11–16 |
| qftentangled | 12–16 |
| qnn, qpeexact, qpeinexact | 11–16 |
| qwalk-noancilla | 2, 10–16 |
| qwalk-v-chain | 2, 4, 6, 8, 10–16 |
| random, realamprandom, su2random, twolocalrandom | 11–16 |
| vqe | 2, 11 |
| wstate | 16 |

Before a refit or measurement, resolve source paths by content hash, verify exact
member bytes and width, and join targets/features/splits by canonical observation
ID or exact QASM hash as appropriate. A filename, matching width, family name or
backend capacity is insufficient identity evidence. A new environment or repaired
adapter creates a separately identified measurement context; it cannot inherit
historical timings under a new fingerprint.

## QPU graph and matched global MLP

The [common-panel execution contract](../protocol/common_panel_completion.json)
specifies the current neural runs. Its representation override is logical-recipe
input, rather than the older compiled-input population in the reusable
[full-feature contract](../protocol/mali_full_features.json).
[The runner](../methods/real_qpu/train_mali.py) explicitly loads
the current partitions instead of invoking the older corpus loader unchanged.

For outer fold `f`, reserve every row assigned to `f` as test. Partition its
outer-training groups into four frozen inner folds: fit gradients and transforms
on inner folds 1–3 and select the model on inner fold 0. The graph and MLP have
identical fit, validation and test IDs, seeded example order, effective batch and
raw-seconds mean-squared-error objective. Neither fitting, transforms nor
checkpoint selection reads outer-test labels. Test labels are joined only after
prediction. There is no final refit incorporating inner validation for this QPU
experiment.

Both methods start with 51 raw global fields: 44 named operation counts, allocated
quantum width, depth and five structural descriptors. Fit-only positive-column-sum
masking retains 40 fields in every fold. Means and unbiased sample standard
deviations use one weight per fit observation in float64, with transformed
storage in float32. A standard deviation at or below `1e-6` zeroes that column
in fit, validation and test. Apply exactly the saved mask and statistics to held-out
rows; do not remask the whole dataset or substitute seven summary features.

The graph additionally uses 178 node fields. Barriers and final measurements are
removed; directed DAG dependencies and complete graphs are retained. The node
schema includes gate type, compact used-wire positions, nominal T1/T2 by logical
wire index and node enumeration. Its source encoding quirks, including zero
wire/T1/T2 slots for one-operand operations, are retained and documented in the
feature contract. Pool node statistics across fit nodes in float64, so their
weighting differs from row-weighted globals. Eleven node columns are constant
and zeroed; 167 vary. Nominal T1/T2 is not execution-day calibration. Source IDs,
explicit backend IDs, shots and numerical gate error/duration are not neural
inputs. Complete graph/node/global signatures and fit-only statistics are retained
in the input index (not bundled)
and fold transforms (not bundled).

The graph branch has three `TransformerConv(178,178)` layers, one head, ReLU after
each and global mean pooling. The global branch is retained-width→64→64. The
concatenated 242-vector passes through 512→512→128→1, with ReLU except at the
output. The matched MLP uses retained-width→64→64→512→512→128→1. The final
output is unconstrained. Both use Adam, learning rate `0.0005`, weight decay
`0.0001`, constant learning rate, raw-seconds MSE and 500 complete epochs.
Select the lowest inner-validation MSE state, choosing the earliest exact tie.
There is no early termination, target log transform, hyperparameter search or
post-hoc clipping of neural seconds.

The actual execution telemetry (not bundled)
records effective batch 32 and physical graph microbatch node budget 250,000 for
all folds. Whole-graph gradient accumulation implements that effective batch;
nodes are never truncated, sampled or split across graph examples. The profile
considered batches 2, 8, 32, 128, 512, 2,048 and 4,096 and node budgets 250,000
and 500,000 with synthetic zero targets. It chose the smallest shared feasible
batch within 5% of best measured graph throughput, then the smallest node budget
in that band. This is a compute adaptation from the source batch-2 regime.

Actual cells record RTX 5070 Ti, PyTorch `2.7.1+cu128`, CUDA runtime `12.8`, FP32
and AMP disabled. The runner sets 12 PyTorch intra-op threads and four inter-op
threads. The execution contract budgets one GPU worker, 12 GiB used GPU memory,
4 GiB free reserve, 8 GiB host cache, 36 GiB aggregate RSS and 8 GiB available
host RAM; auxiliary CPU work has a four-thread aggregate cap. These budgets are
requirements, not measured memory usage for every cell. Preserve cell identity,
imported helper hashes, transform, profile and environment. An atomic checkpoint
includes model, optimizer, scheduler, RNG, completed epoch and best validation
state. Resume restores those identities and RNG and replays an incomplete epoch;
it does not restart with a changed optimizer or profile.

There are 30 cells: two models × five outer folds × seeds 42, 1234 and 31415.
The reader prediction is the median of all three finite seconds predictions per
row. A missing seed is unavailable; never select a favorable seed. Retain seed
outputs and dispersion. The cell manifests (not bundled)
pin the actual code, model, profile and partitions. Technical canaries verify
transform parity, complete-largest-graph memory disposition, full-batch versus
microbatch loss/gradient/Adam parity and checkpoint continuation. Fold-0 acceptance
checks identities and finite outputs, not prediction quality. Saved terminal
receipts are authoritative where progress telemetry still says `training`.

## Polynomial, Ridge and analytical calibration

The common-panel polynomial is a CPU adaptation with its own compiled/submitted
input route. Its ordered fields are upstream `swap`, operand-stack `depth`,
touched `num_qubits`, recorded `shots` and `circuit_count=1`. Here `swap` counts
CX/CZ/ECR operations, unlike the literal SWAP count in neural globals. Preserve
the upstream counter and depth semantics. Do not apply a logarithm, scaler or
constant-field pruning. Within each outer-training population, fit polynomial
degrees 2, 3 and 4 in each of the four grouped inner folds; select highest mean
raw-seconds R², breaking exact ties toward lower degree. Refit the selected
polynomial on complete outer-train and predict outer-test. Every fold selected
degree 2. Finite negative predictions are retained in MAE and R². The
polynomial manifest (not bundled)
records Python `3.10.21`, CPU execution and two threads. This is distinct from
the older five-log-feature polynomial experiment.

Analytical input compilation and calibration are also separate stages. Reuse a
raw calculation only after verifying circuit representation, snapshot, row IDs,
units, shots and method hashes. Matching an observation ID alone does not verify
its inputs. Nominal same-backend FakeBackend assets are not job-day calibration.
Raw single-shot schedule, shot-scaled schedule, effective cost and nominal
throughput retain distinct output clocks. None is silently relabelled observed
service time.

For affine service-time calibration, fit OLS `y=a+b*x` on eligible successful
outer-training rows only. For log-affine calibration, fit
`log1p(y)=a+b*log1p(x)` there and invert before scoring. The declared calibrated
prediction floor is zero. Preserve raw failures and their reasons through
calibration; an unavailable schedule does not acquire a fabricated prediction.
No outer-test values choose coefficients, snapshot, formula or eligibility.
The analytical manifest (not bundled)
and [panel contract](../protocol/real_qpu_common_panel.json) specify
the projection and calibration variants.

For Hyb-style cost, the [analytical contract](../protocol/analytical_extension.json)
computes
`log(T_gate)+T_gate/T2_eff−sum_b N_b*log1p(−mean_error_b)` using one-qubit,
two-qubit and readout buckets. Repeated operations contribute repeatedly to each
operation-weighted mean; `T2_eff` is the minimum finite positive T2 over relevant
operands, and `T_gate` is the serial duration sum with routing already present.
Directives skipped by the formula are counted separately. A nonempty bucket
with mean error exactly one has zero survival and an unavailable log cost.
One individual error of one does not automatically mean the entire bucket mean
is one. Missing/invalid durations, probabilities or T2 remain explicit failures.
For shot-scaled raw cost, add `log(shots)` before exponentiation. Finite log cost
remains eligible for calibrated Ridge even when its raw exponential overflows.

Hyb Ridge uses `StandardScaler` and Ridge with alpha candidates
`{0.1,1,10,100}`. Its inputs are `asinh(log_effective_cost_seconds)` and
`log1p(shots)`; target is `log1p(service_seconds)`. For each outer fold, fit the
scaler separately inside each inner-fit, select smallest mean source-balanced
seconds MAE over all four inner folds, breaking exact ties toward smaller alpha,
then refit scaler and model on eligible full outer-train. Output is `expm1`,
floored at zero with nonfinite predictions failed and no upper clipping.
The gate-time control uses its own duration-only eligibility; it must not inherit
noise/T2/zero-survival exclusions. Kyoto composite calibration is a separately
declared sensitivity, not selected for its test score. The nominal log-cost
route has 7,202 eligible rows and 148 zero-survival rows; composite log-cost has
7,350. Hyb run evidence (not bundled)
records Python `3.10.21`, NumPy `2.2.6`, scikit-learn `1.7.2` and a one-thread
CPU limit.

Scholten-style nominal throughput computes shots × compiled quantum-wire depth
÷ declared nominal throughput. Keep the throughput definition, context and
missingness per row: no CLOPS-h/CLOPS-v conversion or calibration imputation.
Scheduled duration, a serial gate-cost formula and a service-time Ridge prediction
are different outputs even when compared diagnostically with one label ledger.

## Digital simulator measurement

The local host evidence identifies Intel Core i7-14700F, 49,239,855,104 bytes
installed RAM and NVIDIA GeForce RTX 5070 Ti with 16 GiB-class VRAM. GPU evidence
records driver `595.91.07` and compute capability 12.0. Run configurations impose
one GPU timing worker; simulator timing and neural fitting do not overlap.
Keep parsing/build, transpilation, planning, first execution, warm execution,
result extraction and end-to-end clocks separate. Timeout and adapter/resource
records remain assigned observations. Repeated timing-stage rows are not additional
independent circuits or additional failed cells.

### Noisy Aer CPU context

The [Aer context](../data/simulator/aer/measurement/environment.json)
and [package freeze](../data/simulator/aer/measurement/packages.txt)
record Python `3.10.21`, Qiskit distribution `0.44.2`, Terra distribution
`0.25.2.1`, module-reported Qiskit `0.25.2` and Aer `0.12.2`. Keep distribution
and module versions separate. This is CPU noisy simulation with method
`automatic`, not the newer Qiskit used by CUDA-Q adapters. The retained context
does not explicitly pin simulator precision; it must not be described as a
verified FP32/FP64 context by importing a current library default.

Instantiate local `FakeSherbrooke`, verify the frozen Target/noise hashes and
transpile with optimization level 1 and seed 1234. Use 1,024 shots and simulator
seed 1234, maximum parallel threads 2, parallel shots 1 and parallel experiments
1. Each of three sessions creates a new Aer simulator, times transpilation and
first execution separately, executes three untimed target warmups, then times
five warm calls. Reduce warm repetitions by median per session and then by
median across sessions. Timeout is 900 seconds. Preserve noise, Target and
compiled-feature identity when using these labels: a newly installed
FakeBackend is not an interchangeable source of old preprocessing.

The checked snapshot hash is
`509ab97f3bf4cd0c2ddbabe2cdc790e2e440726cb2b65b468c05fc54846ec9c1`;
noise hash is
`64d58378f06c47a1bd3d115a5ebea38748c2a1b9702937d497f99d0691b43aa5`.
[The materializer](../methods/simulator/materialize_azizov_common_core.py)
replays source/hybrid/transpiled feature construction against this historical
context; materializing features is not timing Aer again. The core labels support
150-hash predictor evaluation. The wider 204-member Aer measurements are
frontier/coverage evidence and do not extend that predictor denominator.

### CUDA-Q dense sampling

Dense labels use target `nvidia`, `sample()` with 32 shots and separate FP32 and
FP64 contexts. Terminal measurements remain in the normalized convertible IR.
Verify that the same normalized IR is used across precisions. Preserve a first
call separately from three untimed warmups and five timed calls in each of three
sessions; warm targets are medians of the five calls then the three sessions.
Build and result-extraction/end-to-end clocks are descriptive, not added to warm
execution. The dense worker's frozen timeout is 900 seconds per cell. This
sampling target is not MPS state extraction.

The [verified bridge](../protocol/measurement/cudaq_verified_bridge.json)
records the working CUDA-Q `0.15.1` runtime, Python `3.13.15`, CuPy `13.6.0`,
NumPy `2.5.3` and a separate Qiskit `2.5.2` parser path. CUDA-Q initializes before
the parser path is inserted. The general [environment lock](../protocol/measurement/simulator_environment_lock.json)
also describes an intended fresh environment; its presence does not prove that
fresh target discovery worked. The historical bridge was the actual successful
route. Dense [raw timings](../data/simulator/statevector/measurement/cudaq_dense_common_raw.csv)
retain first/warm, precision and failures. Original per-precision coverage is
612 member-session attempts, 594 successful and 18 adapter failures. The selected
repaired core target has 150 hashes per precision; its repaired normalized IR and
target revision must accompany the predictor results rather than being pooled
with the earlier adapter-failure panel.

### Fixed-bond and joint MPS

MPS uses CUDA-Q `tensornet-mps`, FP64, absolute cutoff `1e-10`, SVD algorithm
`gesvdj`, and terminal-measurement stripping only. Any remaining nonterminal
measurement is unsupported rather than silently dropped. The fixed target has
configured maximum bond 16. [The measurement runner](../methods/simulator/run_common_cudaq_mps.py)
creates a fresh child per member/session, separates QASM parse/conversion,
first `get_state()`, three untimed warmups and five timed `get_state()` calls.
Its timeout default is 900 seconds. There are no shots for state extraction.

After timed MPS calls, generate a dense `nvidia` FP64 state on the same stripped
circuit and calculate normalized state fidelity `abs(<dense|mps>)**2`. Reference
execution and amplitude extraction/overlap are separately recorded and excluded
from MPS runtime. A finite fidelity below 0.99 is a quality failure, not a
missing runtime. Invalid norms, nonfinite amplitudes, incorrect vector length or
basis-order disagreement are technical failures. Runtime-only metrics retain
all finite labels; quality-qualified metrics separately require the threshold.
The [original MPS environment](../data/simulator/mps/bridge_measurement/environment.json)
records the bridge versions and package hash, with empty recorded `CUDA_PATH`
and `LD_LIBRARY_PATH`; do not fill those fields from an intended wrapper.

The selected recovered fixed-bond target has 150 finite labels and 147
quality-pass labels. Its earlier target had 144 finite / 142 quality-pass labels.
They are separate revisions. Five drift controls passed the prescribed band;
missing recovered runtime-banner/loader evidence limits binary-equivalence
claims. Recovered-label refits use [the target adapter](../methods/simulator/run_fixed_mps_predictors.py)
and frozen evidence restored by the recovery archive (not bundled).
Previously finite labels are retained, not replaced by a new campaign mean.

The [joint ladder contract](../protocol/family_aware_joint_runtime_quality_mps.json)
uses configured bonds 2, 4, 8, 16, 32 and 64, the same stripped circuit at every
rung, three fresh-process sessions/rung and the same first/three-warmup/five-warm
measurement structure. That is 150 hashes × six rungs × three sessions = 2,700
technical attempts. A rung passes quality only if every session's fidelity is at
least 0.99. The minimum passing rung is the smallest configured passing bond
after a technically complete ladder; never assume monotonic fidelity or infer a
minimum from a partial ladder. Width ≤9 has maximal possible Schmidt rank ≤16,
so bonds 32/64 probe saturation and overhead, not asymptotic high-bond scaling.

The ladder specifies 180-second cell timeout, 3,600-second pilot cap, 24-hour
campaign cap, 4,096 MiB minimum free VRAM and 12,288 MiB maximum used VRAM.
Its [actual campaign](../data/simulator/mps/joint/measurement/run_manifest.json)
observed peak GPU use 9,067 MiB and is partial: 2,520 successful attempts, 72
timeouts and 108 adapter errors. Adapter errors explicitly report unsupported
`rccx` conversion. Ten hashes lack complete ladders, leaving 140 hashes / 840
observed hash×rung targets from 900 assigned. Keep all 900 identities and the
2,700 attempts. Integrity PASS is not an all-successful campaign. Among complete
ladders, 138 hashes have an achievable passing rung and two have none. These
outcomes do not establish that the quality predictor is a reliable selector.

## Simulator predictor fitting

Each predictor learns one engine/configuration/shots/noise/clock/target-revision
context; precision is part of that context only when it was recorded. Aer's
precision is unspecified in the retained measurement environment. Use frozen
exact-hash outer folds and fit transformations and vocabularies on training
hashes. True held-out family, runtime, fidelity, selected bond, status and
post-execution telemetry are never inputs. Seeds are retained
individually; three-seed seconds predictions are reduced by median, never by
best-seed selection. Classical fits may use CPU; neural fits and inference use
the recorded CUDA environment.

The [Azizov-style Aer protocol](../protocol/azizov_common_core_gnn.json)
and [runner](../methods/simulator/run_azizov_common_core_adaptation.py)
use source/hybrid/transpiled views with local raw global widths 51/64/51. The
graph branch has three 178-wide TransformerConv layers and mean pooling; the
global branch is retained-width→64→64 and fusion head 242→512→512→128→1.
Fit-only zero-variance global removal and population-mean/std scaling are
recomputed inside selection and final fit. Sort outer-training hashes by the
declared deterministic inner key and hold out the first ceiling(20%) for epoch
selection. Adam uses learning rate `0.001`, weight decay `0.0001`, log1p MSE,
batch 8, at most 300 epochs, patience 30, gradient clipping 1.0 and FP32 without
AMP. Select earliest lowest validation loss, then refit from scratch on full
outer-train for that selected number of epochs. This differs from the QPU
500-epoch/no-refit rule. Convert output with `max(0,expm1)` and fail nonfinite
outputs. Final medians require all seeds 42, 1234 and 31415.

The Aer five classical estimators per view are linear regression, Ridge alpha 1
with SVD solver, RBF SVR (`C=10`, epsilon `0.01`, gamma `scale`), 300-tree Random
Forest (leaf minimum 2), and CPU histogram XGBoost (300 trees, depth 3, learning
rate `0.05`). Their complete fixed parameters are in the protocol; seed is 1234,
native thread cap two, and global/target preprocessing uses the same training
hashes. The actual [fitting lock](../data/simulator/aer/azizov/environment_and_cuda_lock.json)
records Python `3.10.21`, NumPy `1.26.4`, scikit-learn `1.7.2`, XGBoost `2.1.4`,
PyTorch `2.7.1+cu128`, PyG `2.6.1`, CUDA `12.8`, FP32, TF32 disabled and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Its dependency overlay is distinct from the
historical Aer measurement environment. Source/hybrid/transpiled view roles
follow the source idea, but local dimensions and a single noise/optimization
context do not reproduce the paper-wide experiment.

The [dense predictor contract](../protocol/cudaq_dense_statevector_predictor_comparison.json)
fits nine methods separately in each precision: training median, linear
regression, Ridge alpha 1, RBF SVR, Random Forest, XGBoost, graph, matched MLP
and work-scaled Ridge. Seven globals are log1p transformed and standardized on
outer-train. Graph/MLP use Adam `0.0005`, weight decay `0.0001`, batch 32, 500
epochs, log1p MSE, FP32 without AMP, and no early stopping/HPO. The operation
DAG graph has three 64-wide TransformerConv layers, pooled graph plus 64-wide
global branch and 128→512→512→128→1 head. The work-scaled Ridge instead fits raw
seconds from `2**n`, work-scaled gate counts and shots, with training-only
StandardScaler and alpha 1; floor negative output at zero. Its empirical
intercept is not an identified physical launch overhead. The protocol pins
Python `3.10.x`, NumPy `1.26.4`, scikit-learn `1.7.2`, XGBoost `2.1.4`,
PyTorch `2.7.1+cu128` and PyG `2.6.1`; actual recovered run manifests must be
checked before claiming an exact patch-version environment.

The [fixed-MPS graph runner](../methods/simulator/run_mps_fixed_chi16_runtime_adaptation.py)
uses six structural globals, outer-training log1p/scaling and a source operation
DAG with training vocabulary. Its CUDA graph has three 64-wide layers, Adam
`0.0005`, weight decay `0.0001`, batch 32, 500 epochs, FP32, log1p MSE, no early
stopping/HPO and three-seed seconds median. Baselines are finite outer-training
median and six-feature Ridge alpha 1. Its [actual environment](../data/simulator/mps/fixed/fold_0/environment.json)
records Python `3.10.21`, PyTorch `2.7.1+cu128`, PyG `2.6.1`, CUDA `12.8` and
two CPU threads. The separately adapted
[family residual](../protocol/family_residual_runtime_only.json)
uses six inputs, 128→64 SiLU backbone, dropout `0.2`, predicted-family embedding,
FiLM and a zero-initialized residual plus linear shortcut. Its family classifier
uses 64→64 hidden layers and Adam `0.001`, weight decay `0.0001`, 200 full-fold
epochs; four-way exact-hash cross-fitting provides family predictions for
training rows. Runtime fitting uses AdamW `0.0003`, weight decay `0.001`, 160
full-fold epochs, cosine schedule and gradient clipping 1.0. The family-agnostic
ablation shares initial common weights and fitting inputs. True test family is
used only for post-inference diagnostics.

The joint runtime/quality model adds `log2(chi)` as a pre-execution configuration
input to the six structural values. It uses the same family classifier mechanism,
a 7→128→64 backbone, runtime and quality heads and matched family-agnostic
ablation. Fit runtime on standardized log1p seconds and quality with binary
cross-entropy, fixed equal loss weights, AdamW `0.0003`, weight decay `0.001`,
160 full-table epochs, cosine schedule and clipping 1.0. Finite quality-failed
rungs remain runtime training labels. All rungs of a hash remain together.
Average quality probabilities over three seeds, choose the lowest rung with
mean probability ≥0.5 or abstain, and use that rung's three-seed median seconds
prediction. Evaluate runtime against the identical observed rung, not an oracle
different rung. Oracle minimum-passing-rung metrics are separate diagnostics.
The family-component holdout is a separately frozen diagnostic; it does not
replace primary exact-hash folds. [The training/aggregate manifest](../data/simulator/mps/joint/analysis/manifest.json)
accounts for missing measurements and the selected-rung quality violations.

## Native tensor-network contraction

cuTensorNet converts a measurement-stripped unitary circuit to a scalar
zero-bitstring amplitude contraction in complex64. It does not simulate noisy
shots or return the dense sampling workload. Each session builds a network,
searches a contraction path, obtains native `RUNTIME_EST`, performs a first
contraction, three untimed warmups and five timed contractions. CUDA events time
first/warm contraction; host build/path clocks remain separate. Estimate and
warm contraction use the same selected plan, slicing, precision and workspace.
The [raw records](../data/simulator/tensor_network/measurement/cutensornet_scalar_common_raw.csv)
pin workspace 14,947,909,632 bytes, optimizer samples 32, seed 137 and one
optimizer thread. The recorded [native environment](../data/simulator/tensor_network/measurement/environment.json)
contains Python `3.14.4`, cuQuantum Python `26.6.0`, CuPy `14.2.0`, NumPy
`2.5.3` and Qiskit `2.5.2`, with its package-lock hash. The
[run manifest](../data/simulator/tensor_network/measurement/run_manifest.json)
records planner timeout
1,800 seconds, contraction timeout 900 and parent-cell timeout 1,800.

[Quality assurance](../data/simulator/tensor_network/analysis/qa.json)
reconciles 603 successful and nine timeout sessions out of 612 attempts, and
3,672 stage/repetition records. Preserve timeout denominator and separate native
estimate/warm ratios for core/frontier. The retained
[interference sidecar](../data/simulator/tensor_network/measurement/MEASUREMENT_INTERFERENCE.json)
records possible CPU contention during path search and no GPU overlap. Path-search
wall-clock is therefore descriptive with that flag; the same-plan CUDA-event
estimate/warm diagnostic remains allowed. This is contraction estimation, not a
prediction of whole-simulator wall-clock.

## Pasqal analog pilot

The [analog configuration](../protocol/measurement/pasqal_emu_mps_formula_crossed_pilot.json)
crosses reconstructed official `adiabatic_afm` and `quench` pulse programs with
3×3, 3×4 and 4×4 layouts and configured bonds 16, 32 and 64: 18 analog cells.
Its source revision and document hash pin pulse formulas. This is not a digital
QASM panel. EMU-MPS TDVP GPU uses step `10 ns`, numerical precision setting
`1e-6`, Krylov maximum 40, interaction cutoff zero, no qubit-order optimization
and one GPU worker. That numerical tolerance does not itself attest tensor
floating-point dtype. The recorded runner command combines the CUDA training
interpreter with a separate EMU-MPS package path; the initialized run manifest's
environment fingerprint remains pending, so complete exact-environment replay
is not demonstrated. The separately retained `package_lock_sha256` is the hash
of a failed `pip freeze --all` diagnostic, not an installed-package lock.
Direct versions were recovered from the retained environment, but the complete
transitive CUDA environment is still not locked.

Measure one session, two untimed warmups and two timed warm repetitions with
1,800-second cell timeout. Constructor/network build is separate from host
`perf_counter` around `MPSBackend.run` with observables disabled. Reference state
generation, capture and overlap occur outside the timing boundary. For each
candidate, the record named `first_execute` follows two reference simulations;
it is not a cold-process or first-CUDA-call observation. Warm repetitions use
fresh backend objects after the two untimed candidate warmups. These are host
call durations, not independently profiled pure GPU kernel durations. The field
named `network_build` times only backend construction; pulse-data conversion,
MPS initialization, evolution and result aggregation inside `.run()` remain
inside the execution clock, not inside that constructor field.
EMU-MPS 2.9.1's `MPS.overlap` returns `abs(inner)**2`, the pure-state fidelity
used by Pulser; it must not be squared again. For each
family/layout/session, bond 256 must agree with bond 512 at fidelity ≥0.9999;
candidate threshold against the accepted reference is 0.99. The pilot completed
18 cells: 17 pass and `quench|4x4|chi16` fails, with fidelity about 0.95147.
Four quality-failed timing records belong to that one cell, not four failed cells.
[The final QA](../data/simulator/analog/qa_final/qa.json)
fits warm seconds per sequence step to `alpha*N**2*chi**3 + beta*N**3*chi**2`
with nonnegative least squares on the 17 passing cells and reports rank,
conditioning and unconstrained signs. Those pilot coefficients remain unpromoted;
they are not a held-out digital runtime-estimator score.

## Maestro process-isolated CPU execution

The selected Maestro-style result uses explicitly optimizer-off QCSim CPU
Statevector, FP64 native `complex<double>`, 1,000 shots and one native thread.
It is separate from the failed optimizer-on calibration pilot and does not
reproduce the closed Composer/selector. The
[resolved protocol](../data/simulator/qcsim/maestro/resolved_protocol.json)
pins Python `3.10.21`, Qiskit `2.5.2`, NumPy `2.2.6`, SciPy `1.15.3`, Maestro
distribution `0.3.1`, source revision, native-module/build hashes, OpenMP and
Boost loader dependencies. The optimizer-off source patch and native build are
external. Its resolver fails when that withheld patch is absent; a unit-test
fixture does not reproduce the native patch.

Set `OMP_NUM_THREADS=1`, `OMP_THREAD_LIMIT=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1` before native import. OpenMP's
explicit hardware-count request makes `OMP_NUM_THREADS` alone insufficient.
Capture allowed CPU affinity before timing and retain it. Resolve and record
`optimize_circuit=False`, QCSim and Statevector on every attempt. The
`use_double_precision` switch is GPU-MPS/TN-only; its false value does not turn
CPU QCSim into FP32. One worker holds the timing/GPU leases, with no competing
heavy work.

Every call uses a fresh spawned process and 180-second monotonic timeout,
including each of three sessions × five repetitions per cell. Retain
engine-reported seconds, host call wall-clock and outer process wall-clock.
The selected target is `result.time_taken`, not either wall-clock and not
persistent-process warm execution. The explicit panel-target contract and actual
labels pool all 15 finite samples by median. A generic protocol field instead
describes median within/across sessions; disclose this inconsistency and preserve
the actual selected label reduction. Each repetition has a deterministic
registered simulator seed derived from exact QASM hash, context and repetition.

Calibration uses 112 synthetic cells at widths 2–9: 72 operation cells and 40
zero-state sampling cells. Operation words use noncommuting RX/RZ and interleaved
CX plus matched wrappers, repeats 256/512/1,024, with 256/1,024 for endpoint
slopes and 512 held out for linearity. Sampling uses shots 1/10/100/1,000/10,000.
Estimate operation coefficients from session slopes, reduce across sessions and
interpolate width coefficients with PCHIP at valid knots, without extrapolation
or bridging invalid knots. Sampling OLS supplies an empirical intercept and
shot slope. The component prediction is
`2**n*(C1q(n)*count1q + Ccx(n)*countCX) + b(n) + m(n)*shots`.
Synthetic calibration is independent of panel labels; no held-out runtime tunes
coefficients or parser eligibility.

All eight calibration knots pass; 143 of 150 panel hashes have labels and seven
remain unsupported. Comparators use the same successful hashes: outer-train
median, nested Ridge and a fresh CUDA source-DAG graph. Ridge uses six log1p
structural inputs, inner-fit StandardScaler, log1p target, alphas
`{0.01,0.1,1,10,100}`, four-way exact-hash inner folds selected by mean seconds
MAE and final outer-train refit. The graph uses the fixed-MPS architecture and
500-epoch settings, with newly joined QCSim labels rather than old MPS labels.
Partial-measurement/classical-width sampling limitations remain adaptation
limits; do not exclude supported circuits because their prediction error is large.
The native shared-library hash was captured after execution, limiting independent
pre-run binary identity attestation. [The evaluation manifest](../data/simulator/qcsim/maestro/evaluation/result_manifest.json)
links calibration, attempts and evaluation provenance.

## What a reader can repeat

Use [the public verification commands](reproduction.md) to check the inventory
and saved simulator scores. They do not restore omitted QPU records or regenerate
the full private report. The historical common-panel analyzer and package replay
dispatcher are not bundled. Earlier byte-identical table builds describe that
larger evidence selection, not fresh source extraction, checkpoint reconstruction
or native timing reproduction from this checkout.

A source refit needs complete hash-matching source circuits, exact feature/model
helpers, snapshot assets, declared representation, fitting environment and
frozen folds. Some runners retain imports of development helpers and historical
external paths. Their inclusion is execution provenance, not proof that all
imports and native engines are portable from this tree. The complete Ma–Li/Qonductor
archives, QPack executed angles/submissions, every historical helper byte and all
intermediate checkpoints are not distributed. Source/model pins and exclusions
are recorded by the published source index and file inventory described in
[reproduction](reproduction.md). Resolve those requirements before
claiming an independent refit. Do not infer omitted precision, loader variables,
binary identity or source payload from a current installation.

For any separately authorized new run, freeze input/runner/import/environment
hashes before execution, use a new output location and preserve the existing
labels, attempts, predictions, splits and historical manifests. Verify technical
canaries and fold-0 identities before continuing unchanged settings; poor accuracy
alone is not a failed gate. Report assigned, observed, predicted and quality-pass
counts independently. Pair methods on exactly identical successful held-out IDs
with a saved ID-set hash and retain unavailable reasons. Current QPU and simulator
intervals use 10,000 paired group/hash bootstrap replicates over saved fits; they
do not include refitting uncertainty, establish unseen-family/backend/width
generalization or authorize a cross-engine ranking.
