# Training and measurement

## Real-QPU dataset

The real-QPU benchmark contains 4,515 observations in 165 conservative groups.
[Dataset construction](data_preprocessing.md) records source filters, units,
reconstruction qualifications, missingness and the fresh fold counts.
The [machine-readable execution summary](../protocol/real_qpu_common_panel.json)
pins the active local contract and prepared inputs. This publication does not
change that running contract, its checkpoints or its raw source evidence.

## Fresh fitting and grouped validation

All methods share five outer test folds. Outer allocation seed is 42; the
four-way inner folds use seeds 43–47. Splits operate on whole conservative
circuit/workflow/model-input groups, without target-based allocation. Every
observation is tested once; no group crosses a train/test or inner partition.
All three sources are represented in every outer test and inner validation set.

Both neural methods use seeds 42, 1234 and 31415. Each cell starts afresh;
no weights, OOF predictions, fitted masks or calibrators from another training
population are reused. Graph and MLP fit on inner1–3 and select the minimum
inner0 validation MSE over the complete epoch budget, with earliest ties.
There is no full-outer-train neural refit. Regressors tune on all four inner
folds then refit outer train.

### Ma–Li-style graph and matched MLP

The raw schema is 51 global positions and 178 positions per graph node,
using pinned upstream model/feature helpers. See the
[ordered feature dictionary](data_and_feature_dictionary.md).
Column selection, means, sample standard deviations and constant-position
zeroing use gradient-fit inputs only; record each fold's actual transforms.
Nominal-index T1/T2 are declared node context, not historical physical placement.

| Setting | Value |
| --- | --- |
| Device | CUDA, one neural GPU worker |
| Seeds | 42, 1234, 31415 |
| Cells | Two methods × five folds × three seeds = 30 |
| Epochs | 500 per cell |
| Effective batch | 32 observations, node-budgeted graph microbatches |
| Node budget | 250,000 |
| Optimizer | Adam, learning rate 0.0005, weight decay 0.0001 |
| Loss | Unweighted raw-seconds MSE |
| Precision | Float32, no AMP |
| Graph cache cap | 8 GiB |
| GPU allocation policy | 12 GiB budget, 4 GiB reserve |
| CPU threads for neural worker | 8 |
| Checkpoint | Atomic every epoch, including optimizer, RNG and identity pins |
| Output reduction | Median prediction per held-out observation, only after all three seeds exist |

The observed environment is Torch 2.7.1+cu128 / CUDA 12.8 on an RTX 5070 Ti
with 16 GiB VRAM. Preprocessing and analytical context use Qiskit 2.5.2 and
qiskit-ibm-runtime 0.49.0. These are recorded environment facts, not a claim
that installing any newer package preserves the same representation.

No fixed circuit timeout stops this training run. Correctness or nonfinite
errors stop a cell; poor validation/test accuracy does not. Resume requires
matching runner, model, transform, context and environment identity.

### Qonductor-style regression families

Inputs follow the pinned upstream five-field route:
`swap` (CX/CZ/ECR count), operand-stack depth, touched width, shots and
circuit count fixed to one. No source ID, backend one-hot or source-specific
missing mask is added.

| Family | Frozen candidate budget |
| --- | ---: |
| Extra Trees | 64 |
| Random Forest | 64 |
| Gradient Boosting | 64 |
| AdaBoost | 48 |
| Histogram Gradient Boosting | 64 |
| Polynomial Regression | 3 degrees |
| **Total** | **307** |

Sampler seed is 0. Select by mean raw-seconds R² across four inner folds,
using earliest ties, then refit outer train. A family selector, if reported,
also uses inner scores only. This bounded search is an adaptation of the
upstream estimator/grid, not the full original exhaustive search.

CPU regression and analytical workers may run alongside GPU fitting under
the contract's four-worker cap and RAM reserve. This does not authorize
simulator timing during heavy training or interference with a measurement clock.

### Analytical methods

Keep raw schedules/cost/throughput values and fitted service-time outputs
separate. Fit affine/log-affine calibration on successful outer-train rows
only. Hyb nominal log-cost Ridge selects alpha from 0.1, 1, 10, 100 using
source-balanced inner MAE and is fitted anew. No snapshot is chosen using
held-out performance.

QCRE invokes the pinned original estimator through BQSKit on the unitary
view; it explicitly omits final measurement and barriers. Qiskit retains its
declared scheduled-duration semantics. Scholten uses nominal throughput and
a compiled wire-depth proxy; that proxy is not original QV-effective depth.
Missing original-paper template/kernel inputs remain unavailable. Raw duration
reuse is allowed only with matching hashes and declared semantics, never as a
shortcut to reuse a learned prediction.

## Reporting and current status

The fresh-fit real-QPU result table is complete and its saved-evidence QA passed.
Each method is assigned all 4,515 observations, including unavailable inputs.
Every score reports its successful row set, coverage, units and output clock.
Pair methods on identical successful held-out IDs; retain an ID-set digest.
Neural seed reduction, source-balanced metrics and tails precede ranking claims.

All 30 neural cells completed 500 epochs; the saved fit-only positive-sum mask
retains 40 of 51 globals in every fold. There are five completed nested-regression
folds, 105 affine/log-affine fit receipts and five Hyb Ridge fit receipts. The
finalizer checked identities, train-only transforms, seed medians, formulas,
coverage and saved metrics; it did not rerun a simulator or a neural forward pass.
See [result provenance](../provenance/real_qpu.json).

Grouped paired bootstrap uses 10,000 replicates and seed 42. The observed
paired error difference is the point estimate; bootstrap mean and percentile
interval are separate. These intervals condition on saved fits and do not
include refitting uncertainty. Source slices reuse unified fits and are not
separate source-local training runs.

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
with a saved ID-set hash and retain unavailable reasons. The QPU protocol and saved simulator
intervals use 10,000 paired group/hash bootstrap replicates over saved fits; they
do not include refitting uncertainty, establish unseen-family/backend/width
generalization or authorize a cross-engine ranking.
