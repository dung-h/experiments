# Methodology

We compare runtime estimators using recorded QPU execution times and local
simulator measurements. QPU and simulator results are separate. Each
comparison identifies its test rows, the part of execution being timed,
circuit representation and train/test split. The
[method registry](../benchmark_v1/registry/method_fidelity_registry_v2.json)
records how each implementation differs from its source paper;
[`CURRENT.json`](../benchmark_v1/registry/CURRENT.json) identifies the
versions used by the report.

The reader-facing construction and comparison rules are pinned in the
[unified benchmark contract](../benchmark_v1/decisions/S82_UNIFIED_BENCHMARK_CONTRACT_20261002.md).
It separates the archived-QPU ledger from simulator configuration cells and
states the table schemas and completion gates. It does not authorize new
experiments.

## Data and reconstruction

The real-QPU ledger contains 8,767 archived observations. Their labels come
from the source artifacts; reconstructed circuits never create replacement
QPU ground truth. The common label name is
`archived_observed_service_execution_time`, in seconds. It denotes the
provider's recorded execution/service boundary rather than pulse duration or
a newly measured local wall clock. Collection protocols differ by source.

| Source | Runtime label and original input | What we reconstruct | Verified identity and remaining limits |
| --- | --- | --- | --- |
| Ma–Li, 340 rows | Mean of three archived IBM `Result.time_taken` values at 1,024 shots; exact logical QASM | Learned V3 uses the original logical circuit; analytical adapters separately compile to a nominal same-backend Target | Logical input is exact. Analytical compiled route and snapshot are not demonstrated to match the historical job |
| Qonductor, 4,482 rows | IBM one-circuit job label, recorded shots, exact submitted physical QASM | Structural features from submitted QASM; separately declared analytical compatibility transformations | Direct DB/ZIP/digest association verified on all 4,482. Submitted physical QASM does not establish original logical QASM |
| QPack MCP, 3,945 rows | Single measured QAOA circuit per optimizer evaluation; source milliseconds divided by 1,000 | Learned V3 uses six pinned angle-insensitive logical `(size,P)` structures; analytical replay separately compiles representative circuits to nominal Targets | Replay matches recorded structural configurations. Exact optimizer angles, submitted routing and QASM are absent; matching structure does not prove executed-instance identity |

Reconstruction supplies model inputs, not new ground-truth execution times.
The runtime labels remain those collected by the source authors. Ma–Li
provides logical QASM, and Qonductor provides submitted physical QASM.
Qonductor's reconstructed logical recipes cover only a subset. For QPack,
we can rebuild the recorded QAOA structure with example angles, but cannot
recover the optimized angles, routing or exact submitted circuit. Tables
therefore distinguish exact inputs, archive-resolved reconstructions and
structure-qualified reconstructions.

### Included data and missing source inputs

The checkout contains canonical row identities and archived labels, frozen
splits, derived feature/graph inputs, predictions, attempts and aggregation
evidence needed to rebuild the published tables. It does **not** bundle the
complete original Ma–Li logical-QASM corpus or the Qonductor
`circuits.zip`/database export. Their upstream repositories and pinned
revisions are recorded in [`upstream/upstream.lock.json`](../upstream/upstream.lock.json),
but source retrieval and redistribution are separate from the table-rebuild
workflow; the rights/provenance hold is recorded in
[S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md).

Consequently, a clean clone can verify the packaged evidence and regenerate
reported metrics from frozen rows, but it cannot independently re-extract all
circuit features or retrain the QPU models from the original source QASM. The
exact-input statements in the table above describe inputs used by the local
runs; they do not imply that every original source circuit is distributed in
this checkout. For QPack, only recorded structural configuration and the
explicitly qualified representative-angle replay are available; historical
optimized angles, submitted QASM and the full execution payload were not
recovered. No reconstructed circuit is substituted for a missing source
instance or runtime label.

QPack's 46 optimizer workflows control grouping and workflow-balanced
reporting. The label is one circuit's runtime, not the duration of a workflow.
IC/RH multi-circuit VQE labels are excluded from this ledger. Source label
semantics are documented in [S24](../benchmark_v1/S24_DATASET_SEMANTICS_AND_COMMON_PANEL_DECISION_20260927.md).

The unified dataset retains all 8,767 archived rows. Results are also broken
down by source because collection protocols and circuit representations
differ. A score on this mixture is not a test of transfer to a new source.

Qonductor also has a logical-evidence partition: 230 archive-resolved recipes,
14 graphstate structural-only identities and 4,238 without a logical archive
mapping. These groups overlap the 4,482 physical-QASM rows; they add no new
observations. The 230 recipes use MQT Bench 1.0.3 after a physical-QASM archive
hash join. They are separately labelled reconstructions, not byte-exact
original logical circuits. Unified V3 uses submitted physical QASM for
Qonductor, not these recipes. See the
[representation authority](../benchmark_v1/protocol/representation_authority_v2.json)
and [archive addendum](../benchmark_v1/S39_QONDUCTOR_ARCHIVE_RECOVERY_ADDENDUM_20260928.md).

The [QPack replay audit](../artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/REPORT.md)
checks six structures covering 3,945 rows and 26 backend configurations,
pinned to LibKet/QPack revision `9beaf65e951e01181b1e324cbb1b227af10eef96`.
This validates the declared reconstruction, not recovery of missing angles.
The [V3 circuit contract](../benchmark_v1/decisions/circuit_representation_v3.md)
records the intended input route. The 2026-10-01 code/artifact audit found a
deviation from that contract: learned feature/graph materialization never
performed its promised Ma–Li/QPack target compilation. The recorded inputs are
`logical_pre_transpile` for Ma–Li and reconstructed logical structure for
QPack; the model consumes those circuits directly. Qonductor uses submitted
physical circuits. The scores therefore describe an adaptation with mixed
logical and physical inputs, not the consistently compiled inputs promised
by S67. A compiled-input comparison would require a new experiment. This
deviation is recorded in the
[report contract](../benchmark_v1/execution/manifests/report_finalization.json).

QPack replay uses fixed nonzero representative angles (`rz(0.3)`, `rx(0.2)`)
to instantiate the structure. Learned V3 sees parameter presence, not these
numeric values. Analytical compilation does see them, so its schedule is a
representative-angle nominal replay, not the actual per-iteration schedule;
angle-dependent cancellation or synthesis is not verified against the
unarchived executed circuit.

Five Qonductor legacy `sum_*` alias checks fail, while 4,477 pass. These are
feature-semantic discrepancies, not job/QASM mismatches: [S69](../benchmark_v1/decisions/S69_QONDUCTOR_FEATURE_AUTHORITY_AND_C122_ADJUDICATION_V1_20260929.md)
separately establishes 4,482/4,482 direct associations. Historical
archive-metadata polynomial results remain an adaptation; unified V3 uses
the corrected QASM structural route.

## Hardware context and runtime boundaries

Snapshot-dependent analytical methods use pinned same-backend FakeBackend
assets under the [snapshot policy](../benchmark_v1/decisions/snapshot_policy_v3.md).
The current twelve snapshot records are nominal tier 3 contexts. They have
versions, dates, asset hashes and operation coverage; they are not claimed to
be job-day calibration. Missing duration/error inputs remain unavailable.
CLOPS_h is never converted into CLOPS_v or inferred from T1/T2.

Scheduled single-shot duration, shot-scaled schedule, nominal CLOPS cost and
Hyb-HANAS effective cost retain their own output clocks. Their raw errors
against service-time labels are diagnostic. Outer-train-only affine or
log-affine calibration produces a separate learned adaptation. It is not the
native API or paper output.

Simulator evidence is local to the recorded RTX 5070 Ti host, approximately
45 GiB usable RAM and 16 GiB-class VRAM, with pinned library/driver and run
configuration. Compile, network build, path search, first execution, warm
execution, result extraction and end-to-end boundaries remain separate.
Shots, precision, workspace, threads and approximation/quality policy are
part of the target context. Same QASM alone does not make different engines
execute the same workload.

## Split, fitting and comparison

The unified methods share the frozen [outer assignment](../artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv).
QASM identities and QPack workflows are grouped to prevent repeated circuits
or optimizer iterations crossing folds. An outer test fold is held aside;
selection, preprocessing, early stopping and calibration use outer training
only, with grouped inner validation where applicable. Polynomial degree is
selected in grouped inner folds. Graph V3 instead uses a fixed 500-epoch
schedule: no HPO, early stopping or test-driven checkpoint selection. Each observation
receives an out-of-fold prediction or an explicit terminal status.

These are group-held-out folds, not automatically unseen-backend,
unseen-family or future-time tests. QPack keeps each of its 46 workflows
together, but its six reconstructed structures recur across workflows and
outer folds. Thus its current score measures held-out workflows with possibly
seen structural templates; it does not demonstrate unseen-QAOA-structure
generalization or recovery of the missing per-iteration angles.

The [V3 learned contract](../benchmark_v1/decisions/learned_method_contract_v3.md)
defines the unified polynomial and graph methods. Polynomial uses five
`log1p` features: active width, structural depth, two-qubit count, swap-like
count and shots; `circuit_count=1` is checked then dropped as a constant.
Graph adds one-qubit count and measurement count to those five features.
The graph adaptation combines a directed graph
branch with a separate seven-feature structural/shot MLP; source identity,
backend one-hot and representation tier are excluded from its primary input.
Neural runs use CUDA and seeds 42, 1234 and 31415, reduced by rowwise median.
The separate Ma–Li source-native 340 run used seed 1234; it is not a
three-seed result and has an unstable fold.

Report coverage on the full assigned population, then compare prediction
errors only on identical successful held-out rows. Preserve failures and
unavailable states in the denominator. Source-stratified and balanced
summaries expose the mixture of collection protocols; pooled metrics do not
establish cross-source transfer. MAE, MedAE, log error, R² and tail errors use
each result's frozen metric contract. Resampling groups, seed and repetition
count are read from that contract rather than assumed universal. Pairs with
different success coverage remain descriptive: common-row errors are not a
primary full-envelope superiority claim. Existing
reader-pack `primary_same_clock` labels for those pairs need correction in
new derived outputs. The pooled cluster-bootstrap intervals are not
source-specific or source-balanced intervals. The current aggregate covers 24
methods and 210,408 attempt rows, with 72 source-local pair comparisons and
equal-workflow QPack diagnostics across 46 workflows. It does not compute pooled cross-source error metrics;
source-balanced macro summaries, where present, are diagnostic and not
rankable. See the [aggregate manifest](../artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/manifest.json).

### Hyb-HANAS bounded numerical repair

The bounded repair is `PASS`: it evaluates log-survival stably and fits the
calibrations on outer-training rows. This is a numerical/provenance pass, not
a predictive-quality claim. The source metrics show enormous raw effective-cost
errors (for example, Qonductor raw one-circuit MAE about `1.10e281` seconds,
with R² unscorable). All Hyb variants predict 0/340 Ma–Li rows because the
compiled-cost values overflow or have zero survival. Outer-train log-affine
calibration is still weak: the all-source one-circuit adaptation has MAE
about 1.88 seconds and R² about −1.36 on its 8,420 predicted rows. Raw Hyb
clocks remain cross-clock diagnostics, not observed service-time estimates.
See the [repair manifest](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_repair_manifest.json)
and [source metrics](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_source_metrics.csv);
the older artifacts remain unchanged.

The snapshot audit traced all 148 zero-survival Ma–Li cases to
Kyoto. Its nominal FakeKyoto snapshot reports error=1 for all 144 ECR gates,
despite marking them operational. Those inputs cannot establish the
historical QPU's actual reliability. The other 192 Ma–Li cases, on Osaka,
overflow the effective-cost calculation. These outcomes describe this
formula/calibration adaptation; they do not establish failure of the paper's
original workload. See the bounded repair
[manifest](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_repair_manifest.json)
and source-stratified [metrics](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_source_metrics.csv).

## Simulator panel and method availability

The common panel has 204 source members, 191 distinct QASM hashes, 22
families and widths 2–16. Declared duplicate aliases remain visible. The core
has 162 members at q≤9; the frontier has 42 at q=10–16. The
[panel manifest](../artifacts/benchmark_v1/sim_common_q16_manifest_20260927/manifest.json)
identifies the files; no circuit was invented to fill a rectangular grid.

| Approach | Current evaluated scope | Implementation limit |
| --- | --- | --- |
| Ma–Li-style local graph adaptation | OOF predictions on 162 Aer core members / 150 QASM hashes | Three-layer graph plus seven-feature branch adapted to frozen local Aer warm labels; not original Ma–Li or Azizov GNN; no timing was run |
| Qonductor polynomial | Unified QPU adaptation and separately scoped source-local results | Grouped evaluation/common features change source estimator protocol |
| Azizov-related Aer evidence | Historical logical ridge and compiled GBR OOF evidence on 162 core members; the local graph result is a separate adaptation | Ridge/GBR are not Azizov's GNN; the graph adaptation is not a faithful Azizov implementation; author code/table unavailable; frontier accuracy excluded |
| Family-aware residual | Public-artifact reconstruction elsewhere | Common local-panel approximation/quality labels absent |
| Maestro | Existing component calibration plus parser/prediction preflight on 204 panel members | Preflight only: no candidate timing or score; paper-exact Composer unavailable |
| Pasqal EMU-MPS | Completed 18-cell analog pilot | Four quality failures, unpromoted coefficients; separate pulse/layout workload rather than digital QASM panel |
| cuTensorNet | Native estimate paired to matching selected-plan warm scalar contraction | This estimates contraction, not whole simulator end-to-end runtime |
| Aer / CUDA-Q dense / CUDA-Q MPS | Local timing and failure/quality evidence | Simulator measurements are not runtime-estimator methods; MPS always carries its quality gate |

The [simulator pack](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/HOW_TO_READ.md)
shows exact compatible cells and pending methods. The local Aer graph
adaptation's five-fold OOF result on the 162-row core has MAE `0.387654` s,
median absolute error `0.036025` s,
log1p MAE `0.091407`, and R² `0.155835`; the 42-row q10–16 frontier is not
used for fit or accuracy. This local Ma–Li-style graph adaptation does not
complete the six-method panel comparison or establish an Azizov GNN result.

Maestro's existing component model was used only for parser/prediction
preflight: 56 of 408 candidate-member cells are predicted, 244 are out of
grid, and 108 are unavailable; 35 panel members fail parser/width checks.
Simulator timing was not run and there is no candidate score. Its
component calibration remains separate from the missing matched held-out
runtime evaluation. The grid begins at q=8; q2–q7 cannot silently use
extrapolated coefficients. Maestro auto-selection is not required for
candidate runtime prediction.

Maestro is not a scored common-panel predictor. Its ten-cell calibration
pilot made 240 API calls (150 timed); nine statevector cells failed the frozen
stability gate and one MPS cell passed. The run stopped at that gate: there is
no accepted calibration/predictor score and no 408-cell panel timing result.
This is an implementation/calibration limitation, not a runtime-accuracy
claim. The terminal state is retained in the
[pilot run manifest](../artifacts/benchmark_v3/simulator/maestro_candidate_runtime_v3/calibration/run_manifest.json)
and the [report contract](../benchmark_v1/execution/manifests/report_finalization.json).

The [Azizov paper](https://arxiv.org/html/2609.12980#S4.SS2) does describe
the two-branch architecture: three TransformerConv layers, mean pooling and
an MLP. Its source/hybrid/transpiled views use 41/54/41 global features.
Missing author code does not make local reconstruction impossible. The local
Aer graph adaptation reuses the graph architecture for a different target and
feature contract; it
does not implement the Azizov source/hybrid/transpiled feature views or
provide Azizov's common-panel GNN evaluation. Ridge and GBR results must not
be presented as that GNN evaluation.

## Interpretation and residual limitations

Claims refer to the implementation actually evaluated. The
[fidelity overlay](../benchmark_v1/registry/method_fidelity_registry_v2.json)
distinguishes source-native reimplementation, independent reimplementation,
adaptation, native API output and local measurement. Weak results may reflect
method limitations under this protocol, but reconstruction and missing
historical context prevent direct attribution to the paper's original setup.

V3-large covers three giant circuits that V3 could not represent within its
resource cap, but their prediction error remains large. The only confirmed
`if_else` row, `row4477`, is unavailable for the flat graph and analytical
schedule routes. V4 is contract-only; a single dynamic observation cannot
support a population-level dynamic-control accuracy claim. Approximate
simulator feasibility is conditional on fidelity/quality, and this single
host does not establish portability to other hardware.

Current evidence and historical decisions retain their original bytes.
Packaging/reproduction work does not authorize new experiments, trained V4
results, or promotion of pending methods. Public release remains subject to
[rights/citation gates](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md).
