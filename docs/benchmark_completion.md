# Benchmark completion: scientific decisions and execution handoff

This is the active M1–M6 / E1–E8 handoff, dated 2026-10-02. It supplements
S82 and existing method cards. Historical measurements, predictions, reports
and CURRENT remain evidence for their original runs. E1–E6 execution is now
complete; this update records those outcomes and the reproducibility boundary.
No new timing or training was performed during this documentation/validation
pass.

The objective is one archived-QPU observation ledger and a common simulator
circuit panel, with methods evaluated on the same held-out observations within
each compatible runtime configuration. The main deliverables are the QPU and
simulator method tables, not a collection of source-specific case studies.

## Scientific decisions

| Work | Controlling document | Routine execution consequence |
| --- | --- | --- |
| M1: QPU target comparability | `docs/methodology.md`, S82 | Preserve 8,767 observations and source label semantics; a common unit does not prove identical timing boundaries. Disclose shots, missing mitigation metadata and representative-angle QPack replay. |
| M2: Azizov inputs/model | `benchmark_v1/protocol/azizov_common_core_gnn_v1.json` and its linked feature dictionary | Implement the explicitly defined local adaptation, with paper dimensions recorded separately. Reuse existing Aer labels, common hash folds and context. |
| M3: MPS training isolation | S85, its gate and `run_mps_fixed_chi16_runtime_adaptation_v1.py` | Maestro completion is not a scientific prerequisite. Require exclusive timing lock, host-wide GPU lease and no active timing processes throughout CUDA preflight/training. |
| M4: Family-Aware | `benchmark_v1/protocol/family_residual_runtime_only_v1.json` | Evaluate a clearly labelled fixed-MPS runtime-only residual adaptation using existing labels, predicted family and a matched family-agnostic ablation. Original joint threshold/runtime method remains unavailable under S83. |
| M4: Maestro | S83 and S84 | Existing calibration failed stability. S84 is a thread-policy diagnostic, not a predictor score. The cumulative budget of attempt_003 cannot be reset. Preserve its partial outcome; no calibration/panel enlargement is authorised by this handoff. |
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
documentation is updated and links both domains without merging their clocks;
artifact-local method cards are present, while the reader-facing fidelity
registry and `CURRENT.json` are deliberately unchanged pending review. E8's
supported aggregate rebuild and tests pass in this workspace and in a fresh
local clone of the review-package candidate. The clone materialized the three
row-partitioned CSVs, verified 1,469 inventory hashes, rebuilt E6
byte-identically, passed 155 tests, and remained clean afterwards. It is a
table-rebuild validation only: it does not freshly install dependencies,
retrain models, or repeat simulator/QPU measurements. The release inventory
records the retained evidence and omitted local intermediates; no broad source
workspace cleanup was performed.

| ID | Work | Outcome |
| --- | --- | --- |
| E1 | Restore/validate recorded Aer environment and lock; verify pinned context assets | Complete. Environment, package lock, FakeSherbrooke snapshot and noise hashes match the protocol. The environment was not reinstalled or replayed. |
| E2 | Materialize Azizov source/hybrid/transpiled views and compiled replay evidence | Complete. 162 panel members / 150 exact-QASM hashes use frozen C44 folds; replay and QPY round-trip checks pass. |
| E3 | Run fixed-chi16 MPS median, Ridge and graph methods | Complete. Five folds and all three seeds; 144 finite targets, 142 quality-pass, two finite quality-fail, six unavailable. |
| E4 | Run Azizov local classical baselines and three GNN views | Complete. All five folds; GNN training/inference used CUDA. All 27 output methods cover the same 150 hashes. |
| E5 | Run fixed-MPS Family-Aware-inspired residual adaptation and family-agnostic ablation | Complete. Five folds on the same frozen MPS labels; not the original joint runtime/quality method or a family-OOD test. |
| E6 | Aggregate hash-level metrics, paired uncertainty, coverage and method cards | Complete. Separate Aer and MPS results; 18 Aer and 10 MPS paired comparisons, each with 10,000 bootstrap replicates. No cross-clock pooling or imputation. |
| E7 | Update reader-facing result/method disclosures | Content complete in `README.md`, `docs/results.md`, `docs/methodology.md` and `docs/reproduction.md`; E6 method cards remain pinned in its manifest. Historical packs and `CURRENT.json` were not rewritten. Registry promotion remains a review action, not silently applied. |
| E8 | Verify rebuild/reproducibility and package boundary | In-workspace and fresh-local-clone aggregate rebuild, artifact hashes and full test suite pass. The clone validates the packaged table-rebuild path, not a fresh dependency install, retraining, hardware rerun, or redistribution of unavailable external inputs. |

Implement missing runners as ordinary maintained scripts in
`benchmark_v1/scripts/`, with meaningful tests under `benchmark_v1/tests/`.
Keep intermediate files in `work/`; publish final manifests, feature schema,
predictions, metrics and environment only. Do not create a receipt/report for
every small task. Keep progress in this document's table and the run manifest.

## Commands and scheduling

Run commands from the candidate repository root. The existing local CUDA
interpreter is
`/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python`;
verify the frozen environment before using it. The path is a local convenience,
not a portable dependency declaration. The verification environment is
`/home/server/Documents/.venv-qonductor/bin/python`.

E3 has an existing runner; the new output directory below must not exist:

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
