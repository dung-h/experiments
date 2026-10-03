# Benchmark recovery and completion plan

Decision date: 2026-10-03.
Status: scientific choices locked; implementation and execution NOT STARTED by this decision.
Authority: the user's request to freeze the remaining strong-agent decisions and
hand the full execution/monitoring loop to a routine coding agent.

This is the active recovery queue. It supersedes older "no further experiments"
or "strong review required before full execution" wording ONLY for the bounded
tasks below. It does not authorize QPU submissions, changing canonical labels,
unrestricted experiments, a public release, or five-fold V4 control-flow training.
A passing test or a signed plan is not an executed experiment.

## Objective and current evidence

One real-QPU ledger: 8,767 observations (Ma–Li 340, Qonductor 4,482,
QPack MCP 3,945). Methods train on this common ledger and frozen assignment,
not separate source datasets. Source breakdowns explain the common evaluation.

One simulator source panel: 204 members / 191 exact QASM hashes, with a
q2–q9 predictive core of 162 members / 150 hashes and a q10–q16 frontier.
Compare estimators on identical test hashes WITHIN an engine, precision,
shots, timer boundary and quality policy. Different engines have different
labels; one source panel does not imply one shared simulator runtime label.

Existing valid Aer, QCSim optimizer-off, analytical-QPU, cuTensorNet and
Pasqal evidence is retained. Weak accuracy is not an implementation failure.
The new work addresses:
1. incomplete CUDA-Q conversion and MPS worker timeouts;
2. QPU learned inputs that did not implement the promised compiled route;
3. missing CUDA-Q dense statevector predictor comparison.

## Working tree and code ownership

Development/execution root:
`/home/server/Documents/quantum-runtime-estimator-replications`.

Existing publish candidate, a packaging tree rather than a second execution root:
`work/repo_finalization/publish_candidate`.

Some latest Family/MLP/registration scripts exist only in the candidate; older
QPU graph/tabular and dense runners exist in the development root. First
inventory both. Restore missing, hash-verified dependencies into the development
root. For overlapping files, inspect differences and pick the documented active
implementation; do not blanket copy, delete, or overwrite user changes.
Record the selected inputs/code paths and hashes in one run manifest. All new
runs use the development root. Synchronize the allowlisted package only at the
end. The two copies of this plan are maintained identically.

Keep original code behavior as the default. Add explicit protocol/input/output
parameters for new contexts; remove stale identity/count assumptions only in
the new protocol path, not globally. Never relabel new predictions with old IDs.
Do not run historical commands whose hard-coded paths point at mixed-stage
features or prior output directories.

## Decisions locked now

### A. CUDA-Q adapter repair

Repair only unsupported `rccx` conversion using the pinned Qiskit instruction's
exact definition, recursively lowered to operations the pinned CUDA-Q adapter
already supports. Do not substitute CCX: relative phases are part of the gate.

- Preserve circuit order, wire identity, global phase and terminal measurement
  map. MPS removes only terminal measurements; dense sampling keeps them.
- No optimization, routing, gate cancellation or wholesale transpilation.
- Pin original QASM hash, normalized QASM/IR hash, adapter revision, environment
  and ordered operation counts. The same normalized unitary must be used for
  FP32, FP64, MPS and dense fidelity reference.
- CPU canaries: full 8x8 RCCX operator agreement, max absolute discrepancy
  <= 1e-12; phase-align if needed but retain the phase in the lowered circuit.
  Verify composed circuits with a controlled RCCX decomposition, inverse and
  phase-sensitive superposition tests. A single |000> test is insufficient.
- For the six affected q<=9 circuits, compare original and normalized FP64
  states, norm error <= 1e-10 and infidelity <= 1e-10. Verify endianness.
  Dense sampling additionally verifies terminal mapping and total shot count.
- Tests must show a circuit without RCCX is a no-op in ordered normalized IR.
- Exactly six core hashes caused the historical 108 Family attempts to fail:
  Grover-v-chain q5/q7/q9 and QWalk-v-chain q5/q7/q9. Discover their exact
  identities from logs, not filename guesses. Audit the full 204 panel for
  additional affected frontier inputs, using no timing.

### B. MPS recovery, timeouts and quality

Retain six rungs chi=2/4/8/16/32/64, FP64, cutoff=1e-10, gesvdj,
three fresh-process sessions, first call + three warmups + five warm repeats,
dense FP64 fidelity reference after timing, fidelity threshold >=0.99.
No shot feature: the target is warm `get_state()`, not sampling.

The new worker timeout is **900 seconds for the entire hash/rung/session
process**, not for each kernel. Record separate import/build, first call,
warmups, each warm call, reference, extraction and canary stages. Flush stage
markers outside the timed sections and retain the last marker if killed.
First/warm/build/reference/extraction are never summed into the warm label.

Recovery identities are historical failures only:
72 timeouts = four hashes x six rungs x three sessions:
Grover-noancilla q7/q8 and QWalk-noancilla q8/q9;
108 RCCX adapter errors = six hashes x six rungs x three sessions.
Use a new attempt root and keep all historical terminal records.

Pilot: one session at chi=16 for all ten affected hashes, plus the first five
successful unaffected core hashes in sorted SHA order, spanning different
widths when possible. Technical gates are adapter equivalence, valid timing,
basis/norm checks, complete records and resource compliance. Low fidelity
is an outcome, NOT a gate failure. If a hard case still times out, retain its
status and continue the other hashes; run its other rungs within the budget
rather than declaring them failures without an attempt.

Resource guard: preserve max GPU used 12 GiB / free >=4 GiB. Recovery wall cap
24 hours including pilot/controls; this is a newly authorized recovery budget,
not a reset of the old campaign's cap. Report old and new elapsed time.
At the cap, mark unattempted remaining identities `not_attempted_budget`.
Do not claim 180/180 completed attempts if some were never attempted.

Reuse old successful labels only if context, semantic/timing implementation
and normalized IR are unchanged. Pilot controls get three sessions at chi=16.
For each control, compare median warm new/old; if the median of these ratios
is outside [0.8,1.25], the two measurement eras are not merged. Instead measure
the full 150-hash ladder in the new context within its remaining budget, keep
a separate table, and report partial coverage if the budget ends. This guard
screens gross drift; it does not prove timing equality.

At most two attempts per failed identity across this recovery, and only after
a documented technical repair; do not rerun unchanged timeouts to cherry-pick.
A valid low-fidelity measurement is never retried to obtain a better fidelity.

### C. Real-QPU compiled-input correction

New IDs:
`qonductor_style_compiled_unified_polynomial`,
`mali_style_compiled_unified_graph`,
`compiled_unified_metadata_mlp`.
All are adaptations/baselines, never original-paper reproduction.
Old mixed-stage scores remain valid descriptions of their old inputs and
are retained as historical/representation-sensitivity results.

Representation:
- Ma–Li: exact logical QASM -> deterministic nominal target compilation.
- Qonductor: exact submitted physical QASM unchanged; no silent re-transpile.
- QPack: the six pinned reconstructed QAOA structures, existing representative
  angles and generator revision -> deterministic nominal target compilation.
  Not exact per-iteration optimizer angles or submitted circuit recovery.

Reuse the pinned C133 backend registry/Qiskit 2.5.2/runtime 0.49.0.
All targets are same-backend nominal, NOT historical job-day calibration.
Match the existing analytical compile route: optimization_level=1,
layout_method=trivial, routing_method=sabre, translation_method=translator,
seed from the existing `run_recovery_c133_c135_v3.seed_for` implementation
and seed registry. Record its context/digest/seed derivation, actual seed,
Target JSON/class/package hashes, compiled circuit hash and operation coverage.
Cache only by full source recipe/input + target + compiler settings identity.
No runtime label participates in compilation/cache selection.

Every source gets the SAME feature/graph materializer after its declared route.
Polynomial globals: active_width, structural_depth, two_qubit_count,
swap_like_count, shots; circuit_count=1 is an assertion.
Graph/MLP globals add one_qubit_count and measurement_count.
Count active wires, not full backend capacity. Keep physical wire indices
deterministically mapped as required by the inherited graph schema.
No source/backend ID, representation tier, calibration, hash, workflow,
missingness mask, observed runtime or telemetry enters these primary models.

Dynamic row4477 remains unsupported by the declared flat DAG/schedule route.
Retain it in the 8,767 denominator; no flattening or V4 splice.
For the controlled primary comparison, train all three models on the exact
same representation-eligible IDs, and compare their identical held-out IDs.
A supported metadata-only dynamic prediction belongs in the existing separate
case table, not the main compiled-input comparison.
Other genuine parse/compatibility/resource failures remain explicit, not
filled with medians or reconstructed from observed runtime.

Keep original frozen five outer folds and grouped inner folds.
Audit exact logical/submitted source-identity leakage before execution.
A new compiled-hash collision is recorded and investigated: it is not
automatically proof that distinct source instances are the same circuit.
QPack's representative structures can recur across workflow folds. This
primary split tests unseen workflows, NOT unseen QAOA structure or exact QASM;
document collision counts and do not claim circuit-OOD for QPack.
If actual known source-identity leakage is found, block this primary stream;
continue independent simulator work. Do not invent a replacement split.

Model settings, inherited without accuracy-driven tuning:
- Polynomial: log1p five globals, log1p seconds target, degree 2/3/4 chosen
  by grouped inner source-balanced MAE, lower-degree tie break; no backend
  context. CPU with fixed threads.
- Graph: V3-large three TransformerConv + mean pool + global branch/head;
  FP32; Adam lr=0.0005, weight_decay=0.0001; 500 epochs, effective batch 32,
  node-budget microbatch 250000 with full large-graph handling, no truncation,
  no early stopping. Seeds 42/1234/31415 and rowwise median.
- Metadata MLP: existing matched seven-global-feature model, same optimizer,
  epochs, batch, target transform and seeds as graph; zero graph slot remains.
- Normalizers/vocabulary are train-fold only; finite seconds after expm1 and
  lower clip zero. Fit from scratch, no historical checkpoint transfer.
- All neural forward/backward/training/inference on CUDA, no CPU fallback.
- Parameterize existing materializer/runner/QA paths; do not change architecture
  or bypass hash assertions merely to make a new directory run.

Graph can cover the three giant Qonductor rows without truncation. If it cannot
under the recorded GPU resource cap, record a resource failure; no tail dropping,
special formula or synthetic prediction. All three models' training population
is fixed from representation preflight, not altered after seeing test errors.

### D. CUDA-Q dense statevector predictor comparison

Two independent contexts: CUDA-Q nvidia dense FP32 and FP64, `sample()`,
**32 shots**, three sessions, three warmups, five warm repeats.
Primary label: per-hash median of the three session medians of five
`t_warm_execution_s` repeats. First/build/extraction/E2E are separate
descriptive clocks. Sampling result count must be 32. No fidelity label from
MPS is substituted. FP32/FP64 label and model tables stay separate.

Envelope: all 204 members / 191 hashes. Predictor OOF uses the SAME 150 q2–q9
hashes and C44 five-fold hash assignments (37/28/26/25/34) as the completed
local simulator studies. Aliases are reduced before fit. q10–q16 remain
frontier coverage/scaling, not silently added to OOF training or width-OOD
accuracy claims.

Reuse valid old dense measurements only after verifying hashes, full context,
duplicate identities and IR no-op for repaired adapter. Recover only affected
adapter-error identities (audit exact counts across precision and sessions);
do not assert the 108 Family errors are dense counts. New dense worker timeout
900 seconds (historical dense runner already used 900). Dense recovery budget
12 hours, including controls; record unattempted budget statuses if exhausted.

Use three sessions on five unaffected sorted-hash controls per precision.
Apply the same [0.8,1.25] median warm ratio gross-drift guard independently
per precision. If failed, obtain fresh full-core timing in that context rather
than mixing eras, within its remaining budget.

Pre-execution representation: normalized CUDA-Q-convertible circuit, terminal
measurements retained. Build directed operation DAG and structural features
from exactly this IR; do not use Aer transpilation or MPS stripped features.
Seven globals use the QPU names above; shots is constant 32. Gate vocabulary
and normalizers are outer-train only.

Run the following nine predictor families on EACH context, identical eligible
training IDs and frozen test hashes:
1. outer-train median;
2. linear regression;
3. Ridge alpha=1;
4. RBF-SVR;
5. Random Forest;
6. XGBoost;
7. source-DAG graph + seven globals (local Ma–Li-style adaptation);
8. matched seven-global-feature metadata MLP;
9. work-scaled Ridge baseline.

Classical 2–6 use StandardScaler fit on outer-train on log1p nonnegative
globals; target log1p seconds, inverse expm1 clipped at zero.
Inherit existing Azizov-local classical parameters: SVR C=10, epsilon=0.01,
gamma=scale, tol=0.001; RF 300 trees, min_samples_leaf=2, other parent
parameters unchanged; XGBoost 300 trees, max_depth=3, lr=0.05, subsample=1,
colsample=1, lambda=1, alpha=0, hist/CPU; stochastic seed=1234, n_jobs<=2.
No new HPO. Linear/Ridge deterministic. Record a genuine missing dependency
as a repairable installation issue, not an unavailable paper method.

Graph and MLP use the frozen QPU architecture/model settings above, trained
from scratch for each dense context, 500 epochs, three fixed seeds.
Only the graph schema's width/caps may use the existing small-panel handling;
do not change layers/head/loss. MLP input dimensions match the seven globals.

Work-scaled Ridge: raw-seconds Ridge alpha=1, train-only StandardScaler of
[2^n, 2^n*1Q_count, 2^n*2Q_count, 2^n*multiQ_count, shots], intercept fitted,
negative prediction clipped at zero. It is an empirical internal baseline,
NOT Maestro, a published law, or a physical launch-overhead decomposition.
No coefficient interpretation without a separate identification experiment.

Primary row population per context is label/representation eligibility,
determined before fits. Every method retains an attempt for all 150 hashes;
method-specific numeric failures remain explicit. Rerun with a different
model/hyperparameter solely to improve an outer-test metric is forbidden.
Calibration-based Maestro on CUDA-Q is NOT inferred from the QCSim CPU result:
no CUDA-Q Maestro reproduction is claimed by this stream.

### E. What is not automatically rerun

- Existing noisy-Aer Azizov GNN/classical OOF, QCSim optimizer-off and
  cuTensorNet same-plan diagnostics: retain unless an identified relevant bug
  is demonstrated. Do not refresh all clocks just because new methods exist.
- Family/MPS: if recovery adds labels or changes their measurement context,
  rerun all five C44 folds for fixed-chi graph/Ridge/median/residual+ablation
  and joint runtime-quality+ablation affected by those labels. Joint's separate
  family-component holdout is rerun too. Use the same seeds/configuration and
  original 150-hash assignment, include all finite rung timings regardless of
  quality; publish quality-pass slices/selector violations separately.
  Do not splice recovered predictions into old fitted-model outputs.
  If no labels/context changed, validate and retain old fits.
- Hyb-HANAS: validate the existing numerical repair and calibration inputs;
  error=1 nominal ECR snapshot, zero survival and overflow stay visible.
  No fabricated replacement errors/durations, epsilon clamp or formula change.
- Scholten: retain nominal adaptation/missing CLOPS handling; no CLOPS_h to_v
  conversion and no throughput synthesized from T1/T2/gate durations.
- Ma–Li native 340: audit pinned upstream code, features/units, target transform
  and inverse, seed1234 fold3 training logs, train-only preprocessing,
  checkpoints and eval mode. Recompute inference from saved checkpoints if
  possible. If a concrete implementation bug is found, fix that bug only and
  rerun all affected source-native folds with original hyperparameters/seed;
  label a fresh corrected attempt. Without a proven bug, retain unstable
  result as unresolved local reimplementation, not "paper weakness".
  No separate native hyperparameter/seed search is authorized.
- Seven QCSim measurement-map unsupported circuits: verify why mapping is
  rejected. Preserve measurements; do not silently drop/reorder repeated
  measurements to recover coverage. No new QCSim measurement semantics are
  authorized here. This stream can finish with explicit unsupported cases.
- Pasqal pilot remains analog companion; no forced digital comparison.
- No new full V4 dynamic training, live QPU job, q26–30 probe or TN expansion.

## Routine execution tasks and dependencies

| ID | Task | Depends on | Deliverable |
| --- | --- | --- | --- |
| R0 | Inventory processes/leases, unify active code dependencies, verify inputs/envs | none | one preflight manifest; no training |
| R1 | RCCX shared normalization, timeout parameter, stage telemetry and tests | R0 | verified adapter/runner revision |
| R2 | Compiled QPU circuits, globals, graphs, support and leakage audit | R0 | 8,767-row representation envelope |
| R3 | Dense target reduction and feature/DAG materialization | R0; R1 for repaired IR | 150-hash targets per precision + 204-member coverage |
| R4 | MPS pilot and failed-identity recovery | R1 | merged-or-new-context ladder with immutable lineage |
| R5 | Dense failed-identity recovery and controls | R1, R3 | per-precision target revisions |
| R6 | QPU compiled polynomial five folds | R2 technical PASS | new OOF attempts/predictions |
| R7 | QPU compiled graph + matched MLP five folds/three seeds | R2 technical PASS | CUDA OOF and checkpoints |
| R8 | Dense nine-method comparison per precision | R3, R5 terminal QA | CUDA/CPU OOF, seeds and shared-row pairs |
| R9 | Retrain affected fixed/joint MPS predictors | R4 terminal QA and changed labels/context | C44 + joint family-holdout new OOF |
| R10 | Native Ma–Li, snapshot/formula, QCSim support audit | R0 | cause classified as proven bug/limit/unresolved |
| R11 | Two reader-facing domain tables, pairs/tails/coverage and documentation | terminal R4–R10 | same-row comparisons, no false paper labels |
| R12 | Clean-clone rebuild and allowlisted package update | R11 | reproducibility report and ready-to-commit inventory |

These task IDs are internal scheduling identifiers, NOT public directory names.
Inputs missing from the package can be restored from verified local/source
collections; do not substitute a nearby QASM because its width/family matches.
If an exact required input is unavailable, keep that method-row unavailable.

## Parallel schedule and monitoring loop

During preparation, assign independent file ownership:
- CPU worker A: R1 adapter/timeouts/tests;
- CPU worker B: R2 QPU representations;
- CPU worker C: R3 dense preparation + R10 audits;
- coordinator: inventory, validations, job state and compute ownership.
Do not have multiple agents edit the same runner, protocol or tables.

CPU prep caps: at most 8 worker processes total, one native thread each,
and RAM admission control leaving >=12 GiB available. Deduplicate expensive
compilation by full cache identity, not runtime or filename. If this cap is
too high for actual available RAM, reduce automatically.
While GPU training runs, concurrent CPU preparation/fitting is capped at
four native threads total and must preserve RAM headroom.

GPU timing order: R4 MPS recovery, then R5 dense recovery. Both require the
existing shared timing lock AND host-wide /tmp/qre-benchmark-gpu-lease.lock.
Exactly ONE timed worker. Stop training, heavy CPU preprocessing and CPU fits
during timing; only low-impact monitoring/hash bookkeeping may overlap.
Do not increase timing concurrency based on idle VRAM/utilization.

GPU training queue after timing: R7 QPU graph/MLP, R8 dense graph/MLP,
then R9 affected MPS fits. R6 and dense classical fits may overlap training
under CPU/RAM caps. Independent CPU prep may finish before timing starts.
An unavailable branch does not block unrelated valid branches.

For each fit stream:
1. validate immutable inputs, code/protocol hashes and exact split;
2. run CPU unit tests and a finite CUDA forward/backward canary;
3. run fold0 with ALL declared seeds; validate identity, support, transforms,
   no leakage, finite outputs and checkpoint/inference correspondence;
4. automatically continue folds1–4 on technical PASS, even if MAE/R² is poor;
5. validate each completed fold immediately, then aggregate all terminal folds.
No human sign-off between those stages. Never choose the best seed.

Job states: planned -> preparing -> validated -> running -> completed_validated;
alternatives: failed_technical, unavailable, not_attempted_budget, blocked_scientific.
One coordinator owns the queue. It monitors PID/lease plus durable progress
every 30–60 seconds, logs a short user update every stage or ~10 minutes for
long unattended runs, and estimates ETA from completed cells/epochs, not VRAM.
No generic 90/180-second command timeout on training. Simulator cells use 900.
A running process/unchanged slow graph is not automatically a crashed process.
Persist checkpoints at least every 10 epochs for new neural runners; save
model, optimizer, RNG, epoch, ordering and hashes. Resume only if the runner
actually implements validated epoch-boundary resume; otherwise preserve the
interruption and restart that fold in a new attempt. Never claim resume by
loading weights alone. Skip only independently validated complete outputs.

Allow at most two technical repairs/restarts per stage, documenting causes.
Do not lower epochs, truncate graphs, change quality thresholds, seeds,
precision, labels, split or target clock to escape a failure.
If CUDA temporarily disappears, diagnose driver/environment and pause the GPU
queue; no CPU training fallback or automatic reboot/system changes.
Continue safe CPU work. A necessary scientific redesign or known identity
leakage is a real blocker requiring the user; ordinary package/path defects
do not require another methodological approval.

## Reporting, artifact organization and completion

New outputs:
- artifacts/benchmark_v3/real_qpu/compiled_unified/
- artifacts/benchmark_v3/simulator/cudaq_adapter_recovery/
- artifacts/benchmark_v3/simulator/dense_statevector_predictors/fp32/ and fp64/
- artifacts/benchmark_v3/simulator/mps_recovery/

One frozen manifest per experiment context and per-fold checkpoint/results;
one raw append-only attempt ledger per measurement context. Preserve old
evidence separately, never erase failed attempts after a successful retry.
Scratch/canaries/profiling go under work/benchmark_recovery/, excluded from
public payload. Maintain one work/benchmark_recovery/STATUS.md and one
machine-readable state.json; no dated report/receipt file per polling cycle.

Reader outputs: ONE current Real-QPU table and ONE current Simulator table,
plus coverage, shared-test-row comparisons, source slices and technical
limitations. The QPU main view reports all method families on the common
8,767-row envelope, same-row scores and source-balanced/source-stratified
metrics; source slices do not replace unified training.
The simulator main view has separate engine/precision/clock sections,
same source panel, scored/assigned hashes and explicit quality gates.
Do not count raw/calibrated variants or seeds as distinct paper methods.

Metrics: MAE, MedAE, log1p-MAE, R², p90/p99/max absolute error, coverage,
status/reason rates, seed dispersion, source-balanced macro for QPU, paired
observed error deltas and 10,000 cluster-bootstrap confidence intervals.
Use original leakage groups for QPU and exact-QASM hashes for simulators;
bootstrap deterministic seed via seed_registry bootstrap stream and pair ID.
Keep observed_delta separate from bootstrap_mean_delta.
No fabricated intervals/prediction intervals; unimplemented coverage is explicit.
Keep failed/timeout rows in denominators; common successful intersections
must list exclusions and their target/tail distributions.

Disclose QPack representative angles, nominal snapshots, original logical vs
submitted physical identities, unsupported maps/control flow, local adaptations,
measurement censoring and single-host scope. Distinguish table regeneration
from full source retrieval/training/timing reproducibility. Archive rows remain
measured upstream labels; reconstruction never fabricates a replacement label.

Completion is each R0–R12 task completed_validated OR explicitly terminal with
cause, and every assigned observation/hash accounted for. "All successful",
"all papers reproduced", and "fully source-complete reproducible" are not
required claims and must not be invented.

Regenerate the existing maintained reader builders, docs/results.md,
docs/methodology.md and docs/reproduction.md; explain old vs new results in
one place. Update package allowlist/inventory and restore-part manifests only
after aggregation. Preserve CURRENT.json, C4 and historical release manifests.
Run full tests, validators, hash/reference checks, diff-check and a fresh
verification environment in a clean local clone; regenerate from saved raw
and predictions and compare the resulting tables. A hardware timing rerun
will vary and is not expected to be byte-identical.

No deletion of provenance-bearing artifacts, blanket git add, commit/push,
license/CITATION choice or publication by this handoff. Produce a focused
ready-to-commit inventory, retain user changes and report the author-owned
release decisions. Remove only proven generated scratch from the PUBLIC
allowlist; material deletions require exact targets and a recoverable plan.

## Prompt for the execution/monitoring agent

Read AGENTS.md and this entire plan before acting. Execute R0–R12 with the
parallel preparation schedule and exclusive timing/training queue above.
You may coordinate subagents for independent code/preparation/QA tasks, with
disjoint file ownership. You are authorized to implement the frozen adapters,
parameterize runners, install pinned missing dependencies, run technical
gates, recover failed measurements and complete the frozen fits without
asking after every fold. Do not run experiments until inputs, protocol and
technical canaries for that stream pass. Keep historical evidence immutable.
Continue independent work when a branch is terminal; do not chase accuracy.
Use the single STATUS.md/state.json for monitoring and update the user at
milestones. End with the two updated domain tables, exact unresolved causes,
reproducibility boundary and focused release inventory. Do not merely write
another plan and call the work complete.
