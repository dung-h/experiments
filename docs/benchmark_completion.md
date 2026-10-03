# Benchmark completion: scientific decisions and execution handoff

## Active recovery queue — 2026-10-03

The user's follow-up authorized the remaining scientific decisions and a
routine-agent execution loop. The active queue is R0–R12 in
[benchmark recovery plan](benchmark_recovery_plan.md): RCCX conversion,
900-second MPS worker recovery, compiled-input unified-QPU correction and
CUDA-Q dense FP32/FP64 predictor evaluation. These tasks are locked but
**not executed by writing the plan**. That plan supersedes the older no-new-run
wording below only for its explicit scope. Historical results remain intact;
the development root is the execution workspace, not this packaging tree.

The terminal table below describes the earlier bounded campaigns, not
completion of this recovery queue or the whole original paper benchmark.

The autonomous continuation below is the earlier execution handoff, frozen on
2026-10-03 and retained as history. The later M1–M6 / E1–E8 and round-1 sections preserve the context
and outcomes of earlier work. Their historical "not run" statements do not
override a completed run or the explicit authorization in this continuation.

The objective is one archived-QPU observation ledger and a common simulator
circuit panel, with methods evaluated on the same held-out observations within
each compatible runtime configuration. The main deliverables are the QPU and
simulator method tables, not a collection of source-specific case studies.

## Current terminal status — 2026-10-03

The four bounded streams below have reached a validated result or their
predeclared terminal limitation. No additional strong-agent methodological
freeze is needed before completing documentation and release checks. This is
not a claim that every campaign was fully successful: the family-MPS
measurement ledger is partial, the dynamic result is one case only, and the
Maestro run is a narrow local adaptation. No further training or timing is
authorized by this status block.

| Stream | Verified outcome | Scientific boundary |
| --- | --- | --- |
| QPU metadata-only MLP | Five-fold, three-seed OOF; 8,767 assigned, 8,766 primary paired IDs. MLP MAE 0.9693 s / R² 0.6631 versus V3-large 0.9731 s / 0.6164. Paired MAE interval includes zero. | Seven-global-feature graph ablation, not a new paper method or evidence of universal graph irrelevance. |
| QPU dynamic-control case | Three fresh predictions for `row4477`; median 6.249 s versus 11.969 s observed, absolute error 5.719 s. | One retrospective case; 6,989 general training rows and zero dynamic training examples. Not V4 five-fold accuracy; no splice. |
| Joint family runtime/quality MPS | CUDA C44 and family-holdout OOF complete. Measurement ledger: 2,700 identities, 2,520 successful, 72 timeouts, 108 adapter errors; campaign status `PARTIAL`. C44 scores use 840/900 hash-rung targets. | No runtime improvement established over the matched ablation; predicted-rung quality violations 45.7%. Local adaptation, not paper-exact Family-Aware or a reliable selector. |
| Maestro optimizer-off follow-up | 3,930 unique full identities: 3,825 successful, 105 unsupported; all 8 q=2–9 calibration knots valid; 143/150 panel labels. Same-row five-fold comparators completed; graph fit required CUDA. | CPU QCSim FP64, optimizer off, 1,000 shots. Component MAE 0.000104 s versus Ridge 0.000134 s on 143 rows only; no cross-engine ranking, paper-exact claim, or physical launch-overhead interpretation. |

The local packaging candidate has completed R1–R3: its inventory and reader
documents are synchronized, and a newly installed verification environment
rebuilt the supported tables from a clean local clone. The full regression
suite passes. R4 was prepared as a 218-path staged change set. The user has
now requested commit and push of that evidence and the recovery handoff to
`presentation-review` (2026-10-03). This authorizes a review-branch update,
not a license/citation choice or a tagged final release; S56's remaining
rights decisions and excluded external sources remain explicit. Historical
packs, `CURRENT.json` and C4 remain unchanged.

## Frozen execution contract — authorization retained for audit

This section records the authorization and design constraints that governed
the four completed streams above. It is retained for audit, not as an active
queue or a request for another strong-agent sign-off. Technical checkpoints
were automatic validation gates, not human approvals. The bounded campaigns
were authorized under the contracts below; the existence of a contract alone
was never treated as evidence that a trainer or hardware worker worked. No
timing or training is performed by writing this handoff.

| Stream | Frozen contract | Terminal deliverable |
| --- | --- | --- |
| QPU metadata MLP | `benchmark_v1/protocol/qpu_global_metadata_mlp_baseline_v1.json` | Five-fold OOF for seeds 42/1234/31415, 8,767 assigned IDs, primary paired comparison on the same 8,766 IDs as V3-large. row4477 is supplementary and excluded from every training fold. |
| QPU dynamic-control case | `benchmark_v1/protocol/qpu_control_flow_case.json` | Target-free CUDA canary, fresh three-seed fold-0 fit, persisted row4477 predictions then retrospective case error. Exactly 6,989 train rows; no five-fold V4 score or splice. |
| Family joint runtime/quality | `benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json` | Six-rung measurements, two-head CUDA trainer, matched family-agnostic ablation, C44 OOF and separately labelled family-component holdout diagnostic. |
| Maestro optimizer-off follow-up | `benchmark_v1/protocol/maestro_cpu_component_optimizer_off.json` | Fresh optimizer-off synthetic calibration and panel measurements, same-context component/median/Ridge/graph comparisons, or an explicit terminal pilot failure with retained evidence. |

### Decisions already frozen

Family's architecture, losses, classifier cross-fitting, six bond rungs,
selection threshold, seeds, folds and evaluation objectives are fully specified
in its existing contract. Implement its missing trainer literally. Check
hash-group leakage, train-only normalization and family classification,
same-seed initial shared weights in the ablation, rung matching, and CUDA
forward/backward before the first full fold. A separate human implementation
review is not required to advance when those technical checks pass.

Maestro's follow-up is a **new optimizer-off simulator context**. All 112
synthetic calibration cells and the 150-hash panel must be measured with
`optimize_circuit=False`; reuse neither optimizer-on labels nor the small
probe's coefficients. Keep the parent shots, precision, normalization, thread
policy, clock, folds, calibration formula and validity gates. The new protocol
pins its parent and the diagnostic patch. Native build identities are recorded
before timing. Parameterize existing scripts instead of cloning another
runner; default behavior for historical protocols must remain compatible.
Optimizer-off coefficients are not guaranteed to be valid at untested widths.
Retain every invalid knot and associated prediction unavailability. The
original optimizer-on result still has its original coverage and meaning.

### Execution order and parallel work (historical plan; completed)

The steps below record the pre-execution schedule. The current outcomes are
summarized above; these commands are not a new authorization to rerun them.

1. Inspect processes, output manifests, inputs and environments. Validate
   completed outputs before skipping them; the existence of a directory is
   not completion. Restore missing pinned inputs from the original workspace
   and verify hashes. Use an isolated CUDA environment for neural fitting and
   the recorded CUDA-Q bridge for MPS measurement.
2. Run QPU MLP fold 0 and its validator, then folds 1–4, without waiting for
   model accuracy review. Afterward run the dynamic case's fresh three-seed
   fold-0 fit; `fit` already requires its canary.
3. While QPU training owns the GPU, implement/test the Family trainer and
   Maestro protocol parameterization using bounded CPU work. Subagents may
   handle independent implementation, input QA and aggregation preparation.
   Assign different files to each worker. Limit concurrent implementation
   tests/preprocessing to four CPU threads total during training.
4. With training finished, reserve the shared timing lock and host GPU lease
   for Family's first-ten-hash pilot: 180 sessions across six rungs. If its
   technical, memory and projected-time gates pass, automatically complete
   the declared 2,700-session full campaign. Preserve the 12-GiB memory ceiling,
   180-second cell timeout, one-hour pilot cap and 24-hour full cap. Validate
   basis order, norms and identical IR; finite fidelity below 0.99 is an outcome.
5. Finish Family C44 fold 0 technical QA, then folds 1–4 and the separate
   family-component holdout diagnostic. Train both declared variants on CUDA
   with all three seeds and unchanged settings. Keep each hash's rungs together.
6. Run Maestro optimizer-off preflight and the parent's 300-call paired-thread
   pilot under exclusive timing ownership. On technical/scientific pilot pass,
   finish the 3,930 assigned full identities, reduce coefficients, and only
   then run its same-context CUDA graph comparator. Parent pilot rows may be
   reused only when their full optimizer-off context/seed/identity matches.
7. Aggregate all validated streams; preserve partial/failure outcomes of any
   stream that could not complete. Regenerate existing reader-facing tables
   and update this progress table once per stage, rather than creating a new
   report or dated receipt for every command.

The schedules are sequential for GPU work. CPU hashing, light QA and writing
may overlap; heavy CPU work and all training stop during simulator timing.
Idle GPU/CPU utilization alone does not justify overlapping timed workloads.

### Recovery without further intervention

- Missing packages, loader paths, ordinary implementation bugs and process
  launch errors may be diagnosed and repaired autonomously. Preserve the
  dependency versions that affect circuit/model semantics; any version change
  must be explicit in a new environment record before running that stream.
- Keep raw attempts and completed outputs immutable. Use a fresh attempt
  directory if measurement code/context changes; pin code again before timing.
  A scientifically changed experiment is outside this bounded handoff.
- The QPU MLP currently has no automatic checkpoint resume: preserve an
  interrupted attempt and restart that fold in a new attempt directory. Skip
  already validated completed folds. Never claim checkpoint resume occurred.
- Use Family's identity-checked `--resume` only for identities its runner
  permits. Do not retry/replace terminal measurement failures contrary to a
  protocol. Never reset cumulative pilot/full wall budgets to pass a cap.
- After an identified implementation/infrastructure repair, allow at most two
  new stage attempts. If the same failure persists, record the stream's exact
  terminal reason and continue independent streams. This bounds retries; it
  does not permit erasing earlier failures or selecting a successful seed.
- Poor MAE/R2, seed dispersion, invalid coefficients and low fidelity do not
  authorize tuning, changing split/epochs/thresholds, dropping hard circuits,
  or continuing a pilot whose fixed acceptance criteria failed.
- Wait for an existing compute lease to be released; do not launch competing
  timing/training. After a reboot first establish which processes actually
  survived, which outputs are valid, and what can resume under each contract.

### Reporting, provenance and finish condition

Every assigned ID must retain prediction or terminal unavailable/failure
status. Keep QPU source identity and reconstruction tier, snapshot provenance,
shots, mitigation metadata/missingness and native target semantics visible.
Keep simulator engine, optimization flag, precision, measurement map, bond,
quality threshold and clocks alongside its score. Disclose local adaptations
and reconstructed inputs rather than implying all original papers/data were
reproduced.

QPU reporting includes full-envelope coverage, same-test-row comparison,
source-stratified/source-balanced metrics and grouped paired intervals. MLP
versus V3-large uses the pinned 8,766-row intersection. The dynamic case is a
single retrospective result with seed dispersion, not population accuracy.
Simulator scores compare the same hash, engine configuration, clock and
quality policy. Family runtime error always compares the same hash and bond
rung; report predicted-rung quality violations separately. Maestro optimizer-off
is a separate configuration table, and its component coverage is never hidden
by scoring only its successful subset.

Use existing maintained aggregators or extend them with new method/context
IDs. Pin raw attempts, predictions, protocols, code, environment, folds and
seeds; generate MAE/MedAE/log error/R2/tail metrics, coverage and grouped paired
bootstrap (10,000 replicates, observed delta separate from bootstrap mean).
Retain the frozen Family family-holdout diagnostic as a separate table.

Update existing `docs/results.md`, `docs/methodology.md`, `docs/reproduction.md`,
method cards and release inventory. Keep scratch/caches/native builds/logs
under ignored `work/`; publish only reusable code/tests, necessary protocols,
raw evidence, predictions, metrics and one manifest per final run. Do not sweep
unrelated modifications. Check meaningful implementation tests, artifact
hashes, row/fold/seed completeness, and a final CPU table-rebuild/verification
in a clean checkout. Report table reproduction, input reconstruction and
hardware retraining/retiming separately. Use the previously authorized review
branch for a focused commit/push after verification; preserve existing
CURRENT/history and public-license/tag decisions outside this queue.

Finish when every stream has a validated result or an explicit terminal
failure/limitation, the two-domain reader tables reflect those outcomes, and
the scoped reproducibility/release checks are complete. Do not claim every
method succeeded just because orchestration and tests passed. The final
response gives completed work, concrete findings, coverage, terminal reasons,
artifact paths and publication status. Ordinary stage checkpoints should not
return to the user for confirmation.

## Scientific decisions

| Work | Controlling document | Routine execution consequence |
| --- | --- | --- |
| M1: QPU target comparability | `docs/methodology.md`, S82 | Preserve 8,767 observations and source label semantics; a common unit does not prove identical timing boundaries. Disclose shots, missing mitigation metadata and representative-angle QPack replay. |
| M2: Azizov inputs/model | `benchmark_v1/protocol/azizov_common_core_gnn_v1.json` and its linked feature dictionary | Implement the explicitly defined local adaptation, with paper dimensions recorded separately. Reuse existing Aer labels, common hash folds and context. |
| M3: MPS training isolation | S85, its gate and `run_mps_fixed_chi16_runtime_adaptation_v1.py` | Maestro completion is not a scientific prerequisite. Require exclusive timing lock, host-wide GPU lease and no active timing processes throughout CUDA preflight/training. |
| M4: Family-Aware | `benchmark_v1/protocol/family_residual_runtime_only_v1.json` | Evaluate a clearly labelled fixed-MPS runtime-only residual adaptation using existing labels, predicted family and a matched family-agnostic ablation. Original joint threshold/runtime method remains unavailable under S83. |
| M4: Maestro | S83 and S84 | Existing calibration failed stability. S84 is a thread-policy diagnostic, not a predictor score. The cumulative budget of attempt_003 cannot be reset. Preserve its partial outcome; E1–E6 do not enlarge that old run. The prospective round-1 component design below is a separate experiment. |
| M5: comparison | `docs/methodology.md`, `benchmark_result_table_schema_v1.json` | Use identical test IDs, frozen folds, train-only fitting, full coverage and correctly grouped uncertainty. |
| M6: reproducibility | `docs/reproduction.md` and this handoff | Distinguish table regeneration from input reconstruction and retraining. Missing packaged evidence must be restored and hash-checked before claiming a blocker is scientific. |

Family-Aware's joint method needs new measurements of the same digital
circuits at several approximation rungs, each with warm runtime and an
out-of-timing fidelity reference. That campaign remains a later experiment:
the fixed-configuration residual adaptation does not create those labels.
Maestro similarly needs a resolved context intervention, accepted synthetic
component calibration, and held-out per-candidate runtime evaluation. Simulator
selection is optional for this project's prediction objective. A calibration
failure is retained as implementation evidence; it is not a low score of the
paper method.

## Routine queue

Execution update (2026-10-02): E1–E6 are complete and validated. E7's reader
documentation links both domains without merging their clocks; artifact-local
method cards are present, while the reader-facing fidelity registry and
`CURRENT.json` are unchanged. E8's earlier clean-clone receipt covers
`461cc6e`. Final C6 cloned commit `860829a`, freshly installed the pinned
Python 3.10 verification environment, restored the three row-partitioned CSVs,
verified 1,481 hashes, rebuilt simulator and QPU tables, reproduced all 22
non-manifest scorecard outputs, and passed 155 tests. The rebuilt scorecard
manifest validates. Direct validation of the older stored v3 manifest reports
its superseded source pins for `report_finalization.json` and
`docs/presentation.md`; the historical pack was not rewritten. No model
training, timing, QPU access, or broad source-workspace cleanup occurred.

| ID | Work | Outcome |
| --- | --- | --- |
| E1 | Restore/validate recorded Aer environment and lock; verify pinned context assets | Complete. Environment, package lock, FakeSherbrooke snapshot and noise hashes match the protocol. The environment was not reinstalled or replayed. |
| E2 | Materialize Azizov source/hybrid/transpiled views and compiled replay evidence | Complete. 162 panel members / 150 exact-QASM hashes use frozen C44 folds; replay and QPY round-trip checks pass. |
| E3 | Run fixed-chi16 MPS median, Ridge and graph methods | Complete. Five folds and all three seeds; 144 finite targets, 142 quality-pass, two finite quality-fail, six unavailable. |
| E4 | Run Azizov local classical baselines and three GNN views | Complete. All five folds; GNN training/inference used CUDA. All 27 output methods cover the same 150 hashes. |
| E5 | Run fixed-MPS Family-Aware-inspired residual adaptation and family-agnostic ablation | Complete. Five folds on the same frozen MPS labels; not the original joint runtime/quality method or a family-OOD test. |
| E6 | Aggregate hash-level metrics, paired uncertainty, coverage and method cards | Complete. Separate Aer and MPS results; 18 Aer and 10 MPS paired comparisons, each with 10,000 bootstrap replicates. No cross-clock pooling or imputation. |
| E7 | Update reader-facing result/method disclosures | Content complete in `README.md`, `docs/results.md`, `docs/methodology.md` and `docs/reproduction.md`; E6 method cards remain pinned in its manifest. Historical packs and `CURRENT.json` were not rewritten. Registry promotion remains a review action, not silently applied. |
| E8 | Verify rebuild/reproducibility and package boundary | Final C6 commit `860829a` passed fresh dependency installation, inventory/artifact checks, simulator/QPU rebuilds, a byte-identical 22-file scorecard comparison, and 155 tests. Direct validation of the historical stored manifest flags its two superseded source hashes; rebuilding creates a current manifest that validates. This is table reproduction, not source-complete retraining, hardware rerun, or rights clearance. |

Implement missing runners as ordinary maintained scripts in
`benchmark_v1/scripts/`, with meaningful tests under `benchmark_v1/tests/`.
Keep intermediate files in `work/`; publish final manifests, feature schema,
predictions, metrics and environment only. Do not create a receipt/report for
every small task. Keep progress in this document's table and the run manifest.

## Commands and scheduling

The commands below document the completed E1–E6 runs. Final reporting uses
their saved outputs. Any future authorized rerun must use a new output
directory and its own environment/measurement authorization.

Run commands from the candidate repository root. The existing local CUDA
interpreter is
`/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python`;
verify the frozen environment before using it. The path is a local convenience,
not a portable dependency declaration. The verification environment is
`/home/server/Documents/.venv-qonductor/bin/python`.

The E3 output directory below now exists and is part of the review package:

```bash
/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py --action preflight --output-dir artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof
/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py --action fit-five-fold-oof --output-dir artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof
```

E2 and E4 now have maintained entry points under `benchmark_v1/scripts/`;
their inputs, source hashes and frozen folds are recorded in the artifacts.
E4's fitting overlay and exact runtime lock currently live under ignored
`work/venvs/`, so public reproduction must recreate them from the checked-in
requirements and environment recipe rather than reuse this machine's absolute
interpreter or overlay paths. Do not substitute an older paper-script command
whose targets, CPU policy or split differ.

Use separate subagents for E1, E2 and aggregation preparation. Keep one GPU
training queue for E3, E4 and E5. Fixed-thread CPU baselines can run concurrently
with neural training when no simulator timing is active; use four CPU threads
total by default, including any estimator's declared two-thread fit. All CUDA
runners hold both `work/locks/maestro_qcsim_v2_cpu_timing.lock` and
`/tmp/qre-benchmark-gpu-lease.lock` with nonblocking flock throughout their
CUDA work; use the S85 runner's tested `exclusive_compute_lock` helper.
Timing experiments reserve the corresponding shared lock/lease. Do not weaken a timing
gate to make training available; training checks isolation independently.

Technical failures stop a run: changed target/context, hash/fold mismatch,
leakage, non-finite predictions, absent CUDA or wrong configuration. Low MAE,
high MAE, negative R2, classifier accuracy and seed ranking do not control
continuation. A scientific-contract change returns to a strong reviewer;
ordinary missing packages, path restoration and implementation defects are
routine fixes within this queue.

## Comparison and reporting requirements

Retain all assigned observations. Report target availability and prediction
coverage separately, including the six missing MPS target hashes. Runtime-only
MPS scoring uses 144 finite labels, including two quality failures; report the
142 quality-pass subset separately for the constrained interpretation.

For new simulator comparisons, reduce to one label/prediction per exact QASM
hash and use paired resampling of those hashes. Azizov uses the median of the
existing Aer alias-level `observed_seconds` as its hash target; MPS uses its
existing S85 session/member-median reduction. Compute primary metrics against
these declared hash targets, not by averaging alias errors. Keep member-level
C44-compatible summaries separate. Reaggregate historical predictions from
their saved per-member records before a new paired hash-level comparison;
do not compare an old member-weighted score with a new hash-level score.
Use 10,000 replicates and a
95% percentile interval. Derive the seed from the existing seed registry:
`benchmark_v1/registry/seed_registry.json`, SHA256 of
`root_seed|bootstrap_stream|artifact_id|comparison_label|0`, first
four bytes big-endian. Record the exact seed and quantile implementation.
Report the observed point estimate separately from the bootstrap mean. These
intervals condition on the fitted OOF models; they do not replace per-seed
training dispersion or prove performance on a new backend/family/hardware.

QPU metrics and bootstrap grouping follow `docs/methodology.md`: retain source
and workflow dependence. Different method fidelity classes can be compared
when targets, test IDs and prediction context match; keep the class and input
information beside each number. Raw schedules and calibrated service estimates
must have distinct method IDs and output clocks.

## Completion evidence

E1 progress (2026-10-02): restored the environment JSON and pip freeze at the
protocol paths. SHA-256: environment
`cf4c721c88aed65f6ed39345925eeb790bb58ae790332909c31eff890e47c5ec`; lock
`195dd1a5f87204994a84a32a64f191802d73f295949f01ac71f17385891410d6`. The
recorded versions match the protocol: Qiskit distribution 0.44.2,
qiskit-terra 0.25.2.1, module Qiskit 0.25.2, and Aer 0.12.2; the recorded
FakeSherbrooke snapshot and noise-model hashes also match. Evidence was
restored without installing or replaying the environment.

E3 progress (2026-10-02): fixed-chi16 MPS preflight and five-fold OOF both
passed. Fold sizes were 37/28/26/25/34, covering all 150 assigned hashes;
144 had finite runtime labels (142 quality-pass, two quality-failed) and six
remained unavailable. The technical fold-0 checkpoint passed and folds 1–4
continued automatically. On the 144 finite labels, MAE was 1.7635 s for the
training-fold median, 1.2286 s for Ridge, and 0.3024 s for the median of the
three graph seeds; each seed prediction is retained. On the 142 quality-pass
subset, graph-seed median MAE was 0.3037 s and Ridge MAE 1.2354 s. The paired
hash bootstrap file reports the observed delta separately from its resampled
mean and 95% interval. This is a local fixed-MPS adaptation on this panel; it
does not establish Family-Aware reproduction or family-OOD performance.
The immutable OOF CSV SHA-256 is
`9459502a997fc35d22e4b2b1e2e653b368c6c446c67bbcca2e3966874c95a88d`; the
quality-pass derivation SHA-256 is
`c51ec1e0df3155a23e0e366961041daf8d721793b734e332da1bc01a65f6a008`.
Rebuild the quality-pass summary with
`/home/server/Documents/.venv-qonductor/bin/python benchmark_v1/scripts/aggregate_mps_quality_subset_v1.py --run-dir artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof`.

E4 progress (2026-10-02): Azizov-style local adaptation completed all five
frozen folds over the 150-hash Aer core. The panel has 162 alias-level members;
all aliases of each exact QASM hash share the C44 fold. The three GNN views
use the explicit local dimensions 51/64/51, not the paper's reported
41/54/41; all three neural seeds were fit on CUDA. Together with five
classical estimators per view, the artifact records 27 output methods (24
base model/view/seed methods plus three three-seed medians), each with 150
hash predictions. It is a one-FakeSherbrooke/Opt1 local adaptation, not the
paper's 1,402-circuit/two-backend/four-optimization-level result. The corrected
runner SHA-256 is `66baf8c9477059dc7abf1b156f4ba386575d2797da0bfd7914ebf94c00112ab6`;
the output and environment/lock evidence are under
[`azizov_common_core_adaptation`](../artifacts/benchmark_v3/simulator/azizov_common_core_adaptation/).

E5 progress (2026-10-02): the fixed-MPS residual adaptation and its matched
family-agnostic ablation completed all five folds. On 144 finite targets (142
quality-pass), MAE was 1.0009 s and 1.0813 s respectively; the paired observed
difference was −0.0804 s with 95% hash-bootstrap interval [−0.2852, 0.0804],
which includes zero. This does not establish an incremental family benefit.
The immutable run receipt still says `five_fold_fit_technical_qa_complete_aggregation_pending`;
it predates E6. E6 aggregates the completed OOF predictions without rewriting
that historical execution receipt.

E6 progress (2026-10-02): the reader-oriented aggregate at
[`predictive_runtime_aggregate_v1`](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/)
contains separate Aer warm-execution and CUDA-Q MPS FP64 bond-16 warm-state
tables, coverage rows, method cards and paired bootstrap results. It has 27
Aer methods on 150/150 hashes and five MPS methods on 150 assigned hashes
(144 finite labels; 142 quality-pass, two quality-failed, six unavailable).
All 28 paired comparisons use 10,000 exact-hash bootstrap replicates; the two
domains are never pooled. Aggregate manifest SHA-256:
`701007ea8d38899fcc3bcd5c51a11e236b71b0423faad67e7ca75130bd6d9a96`; source
inventory SHA-256:
`2e81047fda9c2fe37967763e440a5d330b45f5ba60a07c5a973bbb3d3f2085c2`.

Validation on 2026-10-02: E2/E4/E6 validators and focused tests passed; the
full `benchmark_v1/tests` suite passed (155 tests) after E1–E6. A read-only E6
audit checked all eight files against manifest hashes/byte counts, 32 method
cards, 27 Aer
methods with 150/150 hash coverage, five MPS methods with the declared target
denominators, and all 28 paired bootstrap rows (10,000 replicates each). The
historical scorecard and `CURRENT.json` were not changed. This validates the
local evidence and supported aggregation; it is not a reproduction of the
original papers' complete training or execution environments.

Each executed method supplies input/code/protocol/environment hashes, frozen
fold assignment, per-seed OOF predictions and terminal status for every assigned
ID, technical QA, and common-row metrics. The reader receives a dataset
construction table, a QPU method table, a simulator configuration table and a
simulator method table. Include unavailable methods with concrete reasons.
Do not state that every paper was reproduced or all six simulator families
were evaluated when a row remains unfinished.

## Historical round-1 experiment design (2026-10-02)

This section records the pre-execution design and static-check state. Its
statements that timing/training had not run were true on 2026-10-02 and are
superseded by the 2026-10-03 terminal results at the top of this document and
the dated execution follow-through below. The design constraints remain audit
history; they are not current run status or permission to rerun.

This section is the single handoff for the next four experiments. It does not
change the completed E1–E6 results, authorize live QPU jobs, expand the Aer
density-matrix frontier, or promote a new score into CURRENT. The design pass
performs static input checks and unit tests only: no CUDA canary, training or
new runtime measurement has been performed. New results remain pending.

| Experiment | Frozen contract | Implementation boundary |
| --- | --- | --- |
| QPU matched-input baseline | [`qpu_global_metadata_mlp_baseline_v1.json`](../benchmark_v1/protocol/qpu_global_metadata_mlp_baseline_v1.json) | Preflight, CUDA fit, checkpoints and CPU fold validator implemented; fitting not run. |
| Maestro component predictor | [`maestro_cpu_component_benchmark.json`](../benchmark_v1/protocol/maestro_cpu_component_benchmark.json) | Static planner, process-isolated QCSim CPU worker, calibration reducer, and OOF evaluator are implemented and unit-tested. No QCSim timing or learned-model training has been run. |
| Joint runtime/quality MPS predictor | [`family_aware_joint_runtime_quality_mps_v1.json`](../benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json) | Preparation and measurement runner implemented, with mocked technical tests; real CUDA-Q pilot not run. Folded joint-model trainer still needs implementation against the frozen architecture/loss/selection policy. |
| One dynamic-control QPU case | [`qpu_control_flow_case.json`](../benchmark_v1/protocol/qpu_control_flow_case.json) | Pinned preflight, target-free CUDA canary and fresh fold-0 fit implemented; CUDA/PyG execution not tested. Not a five-fold V4 leaderboard. |

### Why these experiments, and what they can establish

**QPU baseline.** The existing polynomial has five coarse features while the
graph adaptation has seven global features. Their difference is not an isolated
test of graph information. The new baseline keeps the graph's exact seven
global inputs, transform, training support, head, optimizer, epochs and seeds,
but removes graph input. Train on the same eligible rows as V3-large in every
fold, excluding row4477 even when it would otherwise be in outer-train. Retain
all 8,767 assigned observations; compare predictions on the same 8,766 shared
test IDs. Its fold-0 metadata-only prediction for row4477 is supplementary.
This is a matched-input graph-removal comparator, not an equal-parameter model
or faithful reproduction of Ma–Li or Qonductor. Classifier/context IDs and
availability masks are forbidden inputs.

**Maestro.** Retain the paper's component-complexity idea, not its closed
Composer or simulator selector. Use the pinned public QCSim CPU statevector
engine, explicit FP64, 1,000 shots and a prospective one-native-thread policy.
Calibrate 112 synthetic cells covering q2–q9, separate pooled 1Q work,
wrapper-matched CX work and shot cost. Repeat 512 checks linearity rather than
fitting it; PCHIP cannot bridge an invalid width knot or extrapolate. Sampling
intercept is empirical and must not be called physical launch overhead.
Keep reported engine time, host wall time and process wall time separate.
This is process-isolated reported execution, not an established persistent-
process warm clock. The new 10-cell paired-thread pilot replaces neither the
old failed pilot nor attempt_003. Old budgets/results remain unchanged.

All 150 common exact QASM hashes need new QCSim labels. Old Aer/CUDA-Q targets
and scores cannot stand in for that engine. Evaluate the component predictor,
outer-train median, nested Ridge and source-DAG adaptation on these identical
labels and folds; the graph comparator needs new CUDA fitting. Low predictive
accuracy is reportable and is not a continuation gate. Unresolved/unstable
calibration coefficients remain explicit unavailable outcomes. This is a
local component adaptation, not the paper's complete prediction engine.

**Family-Aware.** Measure the same 150 hashes at chi 2/4/8/16/32/64, FP64:
900 hash/configuration cells, three separate-process sessions per cell, first
execution separate from warm `get_state`, and fidelity against a dense reference
outside timing. State production is not sampling. Preserve all terminal
measurement/quality outcomes; do not impute missing measurements or assume
monotonic fidelity. For q<=9 the maximum possible Schmidt rank is <=16;
chi 32/64 measure saturation/settings overhead, not asymptotic high-bond scaling.

The contract freezes the two-head model, training-only predicted-family
cross-fitting, matched family-agnostic ablation, losses, CUDA seeds and selection
threshold. Score runtime at the **same hash and same rung** as its measured
label. Report per-rung error, predicted-selection same-rung error/quality
violations, and oracle-minimum-rung error separately. Predicting chi=2 and
comparing its time with an observed chi=16 time would not be runtime-prediction
error. A finite low-fidelity state is a result; invalid norms, wrong basis order
or non-finite state data are technical failures. This local CUDA-Q digital MPS
experiment is not the paper's QuantumRings implementation.

**Dynamic control.** row4477 already has an archived observed label
(11.968528270721436 seconds). The missing piece is compatible representation,
not missing QPU ground truth. Keep it in frozen fold 0, fit fresh V4-layout
weights using 6,989 eligible outer-training observations, and report only this
one retrospective case. A target-free canary must pass before fitting; no test
target is consumed until seed predictions are persisted. Gate vocabulary and
normalization are training-only. The signed S79 layout is applied through an
explicit conversion of the immutable prototype, including per-wire merges and
an empty false branch. Neither the historical prototype's authorization flag
nor the old five-fold V4 prohibition is rewritten.

There are no dynamic-control examples in this training subset. New condition
and branch channels therefore have no informative dynamic training support.
Report all three seeds, median and individual errors, not a population accuracy
claim or n=1 confidence interval. Do not splice the prediction into V3-large
or describe it as a blind validation: the difficult case was identified during
earlier analysis. It is a feasibility/structural-extrapolation case study.

### Routine implementation and execution queue

| Work | Can be done in parallel | Acceptance before advancing |
| --- | --- | --- |
| Restore pinned inputs and environments | Metadata checks/source restoration for all four designs | Exact hashes, fold counts and leakage checks; actual interpreter/library availability. |
| Complete remaining experiment implementations | Family joint trainer only; no GPU/timing | Maestro worker/reducer/evaluator has deterministic fixture, timeout/resume/schema tests. Family joint trainer remains outstanding; no formulas/architecture retuned. |
| QPU baseline fold 0 | One CUDA training queue | `validate_fold` must pass, then folds 1–4 unchanged. No accuracy gate. |
| Dynamic case canary/fit | After training queue is available | Pinned layout/input checks, finite target-free canary; fresh three-seed fold-0 fit. Full five-fold V4 remains out of scope. |
| Family MPS pilot10 | Exclusive simulator timing window | Positive/negative basis controls, normalized finite states, 10 hashes x 6 rungs x 3 sessions recorded; resource/projected-time gate in protocol. Finite quality failures do not fail the technical gate. |
| Maestro paired-thread pilot | Exclusive CPU simulator timing window | Primary thread arm passes context/stability/identification gates. Default arm does not select the primary policy. |
| Full calibration/measurements, then learned comparisons | Metadata/aggregation may overlap; heavy timing and training may not | Complete attempts/statuses, unchanged context; neural comparisons CUDA only. |
| Final two-domain tables | After relevant outputs validate | Common successful test IDs, coverage/quality/clock disclosures and grouped paired uncertainty. No merging incompatible engines or clocks. |

Current local input restoration is not optional cleanup. C134 sidecar,
V3-large graph pickles and the signed prototype/layout sources are available
in the original workspace but are not all in the publish candidate. Use the
explicit `--input-root`/`--features` options for local preparation; restore and
verify those same pinned files before claiming clean-clone training works.
An absolute path on this machine is not a public reproduction recipe.

Run the following **static** checks from the candidate repository root:

```bash
python benchmark_v1/scripts/run_qpu_global_metadata_mlp_baseline_v1.py --stage preflight --features /home/server/Documents/quantum-runtime-estimator-replications/artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c134_feature_sidecar_v3.csv
python benchmark_v1/scripts/prepare_maestro_cpu_component.py --input-root /home/server/Documents/quantum-runtime-estimator-replications --source-dir /home/server/Documents/Quantum-Execution-Time-Prediction/data/quantum_circuits
python benchmark_v1/scripts/run_family_aware_joint_mps_ladder_v1.py --action prepare --output work/family_joint_design_preflight
python benchmark_v1/scripts/run_qpu_control_flow_case.py --action preflight --input-root /home/server/Documents/quantum-runtime-estimator-replications
python -m pytest -q benchmark_v1/tests
```

Use the verification interpreter described above for these commands. The
Family prepare output path must be new. CUDA execution uses a verified CUDA
environment, not the CPU verification interpreter. The baseline contract has
the complete fold/validator shell sequence; the case runner has `--action
canary|fit` and requires a new `--output-dir`. `fit` performs its own canary.
The Family runner exposes `--action measure --scope pilot10|full`; full requires
its passing pilot report and identical context. Restore/check the exact CUDA-Q
bridge interpreter and Qiskit parser paths before calling measurement. Maestro
now has a static preparer and an executable, process-isolated timing runner.
The runner was tested with mocks only. Do not invoke `pilot10` or `full`
without explicit execution authorization and resolution of the measurement-map
coverage question recorded below.

All timing/training holds the tested shared timing lock plus host-wide GPU
lease. Do not exploit idle CPU/RAM by overlapping workloads that contaminate
timing. Preparers, read-only QA and light table work can be parallelized.
Do not infer total completion time until the pilots measure actual throughput.

Maintain only these protocols, reusable scripts/tests and this handoff as
source. Scratch preflights/logs stay in ignored `work/`; successful final runs
publish their raw attempts, predictions, configuration/environment and one
manifest each. Do not generate dated per-task receipts or another scorecard
for each development check. Existing raw results and CURRENT stay unchanged.

Round-1 verification (2026-10-02): all four static preparers/preflights passed.
QPU baseline checks 8,767 assigned / 8,766 paired IDs with zero group overlap;
the case checks 6,989 train / 1,778 withheld observations and complete signed
feature-layout parity. Maestro prepares 112 calibration cells and 150 source
hashes without running the engine. Family prepares 900 configurations / 2,700
session attempts without importing CUDA-Q. Full suite: **183 tests and 4
subtests passed**; `git diff --check` passed. Mocked Family tests cover
successful/timeout/resource-limit workers, pilot-to-full promotion, tampered
ledgers, wrong sessions and cumulative resume accounting. These tests do not
prove the real GPU/engine workers work; hardware canaries remain mandatory.
The initially misplaced three-file Family static preflight was moved, without
deletion, to ignored `work/design_preflight/family_aware_joint_initial/`.
No historical raw artifact, CURRENT, git commit or remote was changed.

Y1 host-environment audit: canonical/split/seed registry, C134, graph row
manifest, control-flow prototype/S79 pins and the 150-hash/162-alias common
panel all match their declared hashes. The source workspace contains all
6,218 graph pickle files (about 162 MB total). They and the C134/control-flow
inputs are not all materialized in the publish candidate; current preflights
use explicit source-workspace paths. Do not claim clean-clone reproduction.
The supported local CUDA model environment is the external Python 3.10.21
`.venv-mali-gpu` (Torch 2.7.1+cu128, PyG 2.6.1); the candidate's verification
environment has CPU-only Torch. CUDA-Q 0.15.1 plus its Qiskit 2.5.2 parser are
available through the externally pinned bridge manifest, and the host has an
RTX 5070 Ti / driver 595.91.07 / 16,303 MiB. No CUDA smoke test was run.

Maestro's current host supports a **native-module import** with the following
environment, without executing a circuit:

```bash
PYTHONPATH=/home/server/Documents/maestro/build \
LD_LIBRARY_PATH=/home/server/Documents/maestro/build/boost_1_89_0/lib \
/home/server/Documents/.venv-qonductor/bin/python -c 'import maestro; print(maestro.__file__)'
```

The native extension SHA-256 is
`c32ddee8d9ee2758b35d486f574dc1307b622f736642767bde1c5bb30518c21d`; it was
built from Maestro revision `2ef7000395fc2a173ab1b4221ec9a235da0392c6`. The
Boost JSON library must be found at the pinned external path; without
`LD_LIBRARY_PATH`, import fails. QCSim FP64 is supported by the bound
`Circuit<double>` / `std::complex<double>` implementation; its
`use_double_precision` flag applies to GPU-MPS/TN and is not a CPU precision
switch. These local binary paths and external QASM files are not a clean-clone
installation recipe. The exact runtime identity/pins are in the Maestro
protocol. Y1 is ready for current-host work, while clean-clone input and
environment packaging remains separate release work.

The source audit also found that QCSim passes an explicit `num_threads` value
derived from hardware concurrency into OpenMP; `OMP_NUM_THREADS=1` alone is
not a hard cap. The primary worker arm now sets `OMP_THREAD_LIMIT=1` before
native import, and the protocol pins the loaded `libgomp` runtime hash. This is
static environment control, not an observed timing result.

Y2 implementation verification (2026-10-02): the Maestro runner/evaluator
focused suite passed **29 tests**; the full benchmark_v1 suite passed **209
tests and 4 subtests**. `py_compile` and `git diff --check` passed. A static
normalization check using the runner's exact `qasm_for` path found 112 of 150
unique panel QASMs satisfy the currently frozen full-register-terminal-
measurement rule; 30 have a smaller number of measurement operations than
qubits: 23 are terminal partial measurements, 7 have nonterminal measurements,
and 8 QFT circuits measure every qubit but declare extra unused classical bits.
All 38 are retained as explicit unsupported/unavailable under the current rule.
Static inspection of pinned Maestro source indicates the terminal-partial and
extra-classical-bit cases are not QCSim engine limitations; the current full-
register restriction is a conservative protocol choice. Nonterminal cases
remain unavailable because the current equivalence check does not validate
dynamic measurement semantics. The static check does not execute Maestro/QCSim
or produce runtime labels. The coverage impact
and whether to revise the measurement-map rule must be decided before pilot:
including terminal partial measurements also requires assessing whether the
full-measurement sampling coefficient is appropriate when fewer qubits are
measured. No simulation timing or model training was run.

Strong review supersession: the maintained [terminal-measurement policy](../benchmark_v1/decisions/MAESTRO_TERMINAL_MEASUREMENT_POLICY.md)
replaces the prospective full-register restriction. Terminal partial maps and
wider classical output are admissible; seven repeated-measurement circuits remain
unavailable. Correction to the preceding historical Y2 wording: these seven
measure the same qubits twice into distinct registers, without intervening
quantum gates; they are not mid-circuit quantum evolution cases.
The unchanged zero-state sampling calibration is explicitly a
limited predictor, not a reason to remove these test circuits. The runner
checks exact maps/register widths and preserves source measurement order.
Regenerate the static plan with the new pins before pilot; the policy contains
the complete routine-agent handoff. Historical pilot failures are unchanged.

Maestro terminal-measurement follow-through (2026-10-02): after applying that
policy, the fresh static plan/preflight passed at 143 admissible hashes and 7
repeated-measurement hashes. The 300-call pilot passed all stability/context
gates. The full phase completed all 3,930 identities (3,825 successful rows,
105 explicit unsupported rows); aggregation reports 143/150 panel targets.
Grouped Ridge OOF is MAE 155.20 µs / R² 0.9231 on the 143 observed labels;
the CUDA source-DAG graph adaptation is 642.58 µs / R² −0.0094 on the same
rows. The Maestro component adaptation has only 13 predictions because only
the n=2 operation coefficient was identified; its 25.34 µs MAE must not be
ranked against full-coverage methods. CUDA graph evaluation ran with seeds
42/1234/31415. Full tests after implementation changes: 217 passed, 4
subtests passed. No commit or push was made.

The complete packaged receipt, pilot/full ledgers, target table, OOF outputs,
measurement-group errors and checksums are under
`artifacts/benchmark_v3/simulator/maestro_cpu_component_benchmark_v1/README.md`.

Maestro CX optimizer-control diagnostic (2026-10-03): the frozen 36-cell
on/off probe completed 540/540 execution calls; a separate 270-call native
optimizer-profile ledger and 54/54 state-equivalence checks passed. Disabling
the optimizer changed the CX-minus-wrapper endpoint-slope contrast from
negative to positive at widths 4 and 9. The result supports an
optimizer-dependent whole-call contrast, not negative/intrinsic CX cost. No
Maestro predictor score, panel label, coefficient or historical artifact was
replaced. The adjudication is in
`benchmark_v1/decisions/MAESTRO_CX_CALIBRATION_REVIEW.md`; data, hashes and a
read-only validator are in
`artifacts/benchmark_v3/simulator/maestro_cx_optimizer_probe_v1/README.md`.
The native diagnostic binary and external source checkout are not vendored, so
the saved measurements validate, but clean-clone rebuild-and-retime has not
been demonstrated.
