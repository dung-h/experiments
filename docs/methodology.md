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

For readers, the archived-QPU results remain in the
[two-domain scorecard v3](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md).
The new simulator predictor results are in the separate E6
[Aer/MPS aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/).
The older simulator reader pack is a measured-configuration inventory, not a
predictor leaderboard. These outputs are intentionally not merged because
their targets, engines and runtime clocks differ.

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
| Qonductor, 4,482 rows | Archived IBM `Result.time_taken` for a one-circuit job; recorded shots range from 1 to 20,000; exact submitted physical QASM | Structural features from submitted QASM; separately declared analytical compatibility transformations | Direct DB/ZIP/digest association verified on all 4,482. Submitted physical QASM does not establish original logical QASM |
| QPack MCP, 3,945 rows | IBM `Result.time_taken` for one measured QAOA circuit per optimizer evaluation at 4,096 shots; source milliseconds divided by 1,000 | Learned V3 uses six pinned angle-insensitive logical `(size,P)` structures; analytical replay separately compiles representative circuits to nominal Targets | Replay matches recorded structural configurations. Exact optimizer angles, submitted routing and QASM are absent; matching structure does not prove executed-instance identity |

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

The current review-package candidate has passed a fresh-local-clone check:
it can verify packaged evidence and regenerate the documented frozen-row
metrics, including E1–E6's aggregate. It cannot independently re-extract all
circuit features or retrain QPU models from the original source QASM. The
exact-input statements in the table above describe inputs used by the local
runs; they do not imply that every original source circuit is distributed in
this checkout. For QPack,
only recorded structural configuration and the explicitly qualified
representative-angle replay are available; historical optimized angles,
submitted QASM and the full execution payload were not recovered. No
reconstructed circuit is substituted for a missing source instance or runtime
label.

QPack's 46 optimizer workflows control grouping and workflow-balanced
reporting. The label is one circuit's runtime, not the duration of a workflow.
IC/RH multi-circuit VQE labels are excluded from this ledger. Source label
semantics are documented in [S24](../benchmark_v1/S24_DATASET_SEMANTICS_AND_COMMON_PANEL_DECISION_20260927.md).

The unified dataset retains all 8,767 archived rows. Results are also broken
down by source because collection protocols and circuit representations
differ. A score on this mixture is not a test of transfer to a new source.

All three source routes use the IBM `time_taken` primitive, as checked in
[S24](../benchmark_v1/S24_DATASET_SEMANTICS_AND_COMMON_PANEL_DECISION_20260927.md).
That shared field supports a unified prediction task; it does not prove that
the provider included exactly the same internal overhead in every historical
job. Queue time, optimizer time and total workflow time are not substituted
for the target. The canonical extract has no source-wide verified mitigation
or resilience setting: it is recorded as unknown, not assumed absent. Only
Qonductor has timestamps on every retained row; Ma–Li and QPack lack canonical
job timestamps. Nominal snapshots cannot establish job-day properties for
those sources. Recorded shots remain part of the prediction input and are
not used to rescale the archived labels to a fictitious common shot count.

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
native API or paper output. A calibrated analytical adaptation can be
compared with graph and polynomial predictions on the same observed target,
outer split and test rows. Different fidelity classes or input views do not
by themselves prohibit a comparison: the table must display those differences
and attribute the result to each complete implemented pipeline. A score
comparison does not isolate the effect of model architecture when pipelines
receive different information.

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

Report coverage on all 8,767 assigned observations and terminal counts by
source. Compute predictive metrics only on finite successful predictions;
an unavailable row has no numeric error and is never assigned zero error.
For every method pair, join by canonical observation ID, verify target and
split hashes, and freeze the common successful row set before computing
either error. Equal counts without equal IDs are insufficient. The comparison
is valid on that intersection even if coverage differs; it cannot establish
which method is better on the excluded rows. Existing reader-pack
`primary_same_clock` labels on coverage-asymmetric pairs must be qualified in
new derived outputs. Full coverage and conditional accuracy are reported
together rather than combined through an invented failure penalty.

New result tables use these explicitly named summaries of the same unified
OOF run; source slices do not retrain separate source models:

- **Source results:** error and coverage for Ma–Li, Qonductor and QPack.
- **Source-balanced MAE:** arithmetic mean of the three source MAEs on the
  declared successful population. For a pair, both methods use the same IDs
  within each source. If any source has no paired rows, the three-source
  summary is unavailable rather than silently averaging fewer sources.
- **QPack workflow-balanced MAE:** first average absolute errors within each
  workflow, then give every represented workflow equal weight. Report the
  represented/assigned workflow count and coverage for each workflow. A
  missing workflow is not a zero-error workflow. This is a companion measure;
  the ordinary QPack source MAE remains observation-weighted.
- **Pooled MAE:** all successful observations have equal weight. This describes
  the dataset mixture and is secondary; it is not an unseen-source test.

Report MAE, MedAE, `mean(abs(log1p(predicted_seconds) -
log1p(target_seconds)))`, p90/p99/max absolute error and R² in seconds.
Predictions must follow their variant's frozen nonnegative-output policy;
aggregation must not silently clip them or replace infinities. R² is
undefined when target variance is zero. A source-balanced mean of source R²
values, if shown, is labelled macro R² rather than pooled R². Preserve all
neural seeds, the prescribed rowwise median prediction and per-seed metrics;
three seeds describe observed training variability, not a precise estimate
of all initialization uncertainty.

For new paired summaries use 1,000 bootstrap replicates and percentile 95%
intervals, recording the registered `20260925-bootstrap` stream, derived
seed and group column in the output manifest. Sample entire frozen leakage
groups with replacement within each source; every sampled group brings all
its paired rows, including multiplicity. The frozen
[split manifest](../artifacts/benchmark_v2/real_qpu/split_manifest_v2.json)
records zero proven cross-source lineage edges, so the current sources can
be resampled separately. Verify that fact before aggregation. Discovery of
a proven cross-source component requires a reviewed joint-cluster protocol;
it cannot be ignored or split into independent source draws. QPack workflows
are its clusters.
Use the same draw for both methods and compute
`observed_mae_difference = MAE(candidate) - MAE(reference)` directly from
the original paired rows. Negative means smaller candidate error. The mean
of resampled differences is `bootstrap_mean_mae_difference`, not the
observed difference. Source-balanced intervals recompute the equal-source
mean in every replicate; workflow-balanced intervals resample workflows
and recompute the equal-workflow mean. Include cluster counts: a narrow
interval from many repeated observations does not establish many independent
circuits. These intervals condition on fitted OOF predictions and measure
sampling variability; they do not replace the per-seed results or full
retraining uncertainty. Select comparisons in advance and report all of them,
not only significant pairs.

The historical aggregate covers 24 method variants and 210,408 attempt rows,
with 72 source-local comparisons and equal-workflow QPack diagnostics across
46 workflows. Its source-balanced macro is marked `diagnostic_not_rankable`,
and it does not compute pooled cross-source metrics. Its existing intervals
and scores remain unchanged. The definitions above govern new derived
summaries; they are not claims that those summaries have already been
computed. See the [aggregate manifest](../artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/manifest.json)
and [shared split contract](../benchmark_v1/protocol/unified_split_contract_v2.json).

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

The 204-member common simulator inventory (191 exact-QASM hashes) is the
coverage and local-measurement panel; it is not a single predictor denominator.
The predictive results below use the common 150-hash q≤9 Aer core and the
same 150-hash fixed-MPS cell. The 42 q10–16 members remain frontier/coverage
evidence, not predictor test rows. See the
[completion handoff](benchmark_completion.md) for runner and environment
details. E4 and E5 are complete five-fold local adaptations, not paper-exact
reproductions. The joint Family-Aware threshold/runtime method still lacks the
required approximation-ladder labels. Maestro remains a separate terminal
calibration pilot; its status is not a prerequisite for these completed
predictors.

The common panel has 204 source members, 191 distinct QASM hashes, 22
families and widths 2–16. Declared duplicate aliases remain visible. The core
has 162 members at q≤9; the frontier has 42 at q=10–16. The
[panel manifest](../artifacts/benchmark_v1/sim_common_q16_manifest_20260927/manifest.json)
identifies the files; no circuit was invented to fill a rectangular grid.

| Approach | Current evaluated scope | Implementation limit |
| --- | --- | --- |
| Ma–Li-style local Aer graph adaptation | Existing OOF predictions on 162 Aer core members / 150 QASM hashes | Three-layer graph plus seven-feature branch adapted to frozen local Aer warm labels; not original Ma–Li or Azizov GNN |
| Qonductor polynomial | Unified QPU adaptation and separately scoped source-local results | Grouped evaluation/common features change source estimator protocol |
| Azizov-style common-core adaptation | Three source/hybrid/transpiled GNN views and five classical estimators per view, all five folds on the same 150 Aer hashes | Explicit local 51/64/51 feature contract; one FakeSherbrooke/Opt1 cell, not the paper's 1,402-circuit, two-backend, four-optimization-level experiment |
| Family-Aware-inspired residual | Completed fixed-MPS runtime-only residual model and family-agnostic ablation on 150 hashes | Not the paper's joint runtime-quality method; no approximation ladder or family-OOD claim |
| Maestro | Existing component calibration plus parser/prediction preflight on 204 panel members | Preflight only: no candidate timing or score; paper-exact Composer unavailable |
| Pasqal EMU-MPS | Completed 18-cell analog pilot | Four quality failures, unpromoted coefficients; separate pulse/layout workload rather than digital QASM panel |
| cuTensorNet | Native estimate paired to matching selected-plan warm scalar contraction | This estimates contraction, not whole simulator end-to-end runtime |
| Aer / CUDA-Q dense / CUDA-Q MPS | Local timing and failure/quality evidence | Simulator measurements are not runtime-estimator methods; MPS always carries its quality gate |

### Local predictor results on shared simulator targets

These are two separate evaluations: Aer's existing noisy warm-execution
labels and the fixed CUDA-Q MPS FP64 bond-16 warm-state labels. A prediction's
`method_output_clock` is named separately from its measured
`evaluation_target_clock` in the [E6 aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json).
The table reports MAE in seconds and R² against each cell's own target; rows
from the two engines are not rankable against each other.

| Target cell | Method | Scored hashes | MAE (s) | R² |
| --- | --- | ---: | ---: | ---: |
| Aer noisy warm execution | Azizov-style source-view GNN, 3-seed median | 150/150 | 0.4497 | 0.0927 |
| Aer noisy warm execution | Azizov-style hybrid-view GNN, 3-seed median | 150/150 | 0.4064 | 0.0914 |
| Aer noisy warm execution | Azizov-style transpiled-view GNN, 3-seed median | 150/150 | 0.3008 | 0.5206 |
| Aer noisy warm execution | XGBoost, transpiled view | 150/150 | 0.2970 | 0.2889 |
| CUDA-Q MPS FP64, bond-16 | Training-fold median | 144/150 | 1.7635 | −0.0383 |
| CUDA-Q MPS FP64, bond-16 | Ridge, six structural features | 144/150 | 1.2286 | 0.3787 |
| CUDA-Q MPS FP64, bond-16 | Source-DAG graph, 3-seed median | 144/150 | 0.3024 | 0.9375 |
| CUDA-Q MPS FP64, bond-16 | Family-Aware-inspired residual, 3-seed median | 144/150 | 1.0009 | 0.5562 |
| CUDA-Q MPS FP64, bond-16 | Matched family-agnostic ablation, 3-seed median | 144/150 | 1.0813 | 0.4823 |

For the Aer core, the paired source-view minus transpiled-view GNN MAE was
`+0.1489 s` (95% exact-hash bootstrap interval `[+0.0284,+0.3039]`), so the
transpiled view had lower error on this panel. The transpiled GNN versus
transpiled XGBoost difference was `+0.0037 s` (`[−0.1726,+0.1324]`): no
resolved advantage between those two. This is evidence about the tested
feature views on one backend and optimization level, not a universal benefit
of transpilation or of GNNs.

For the MPS cell, the family residual's observed MAE improvement over the
family-agnostic ablation was `−0.0804 s`, with interval
`[−0.2852,+0.0804]` including zero; an incremental family benefit is not
established. Its error was higher than the graph median by `+0.6984 s`
(`[+0.1878,+1.4256]`). The two quality-failed finite targets remain in the
144-label runtime-only metrics; on the 142 quality-pass subset, graph MAE is
`0.3037 s`, family residual `1.0081 s`, family-agnostic `1.0925 s`, and Ridge
`1.2354 s`. Six targets are unavailable and never imputed. All intervals use
10,000 paired exact-QASM-hash bootstrap replicates and condition on the fitted
OOF predictions; they do not establish transfer to unseen families, engines,
backends or hardware.

Machine-readable metrics, per-hash coverage, paired intervals, method cards
and pinned source hashes are in the
[predictive aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/).
It records 27 Aer methods and five MPS methods. The older
[simulator reader pack](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/HOW_TO_READ.md)
remains an immutable local-measurement/configuration pack; it does not contain
these predictor scores and does not combine their clocks.

The separate local Aer graph baseline has a historical five-fold OOF result
on the 162-member core (150 exact hashes): MAE `0.387654` s, median absolute
error `0.036025` s, log1p MAE `0.091407`, and R² `0.155835`. It is a different
model/input contract from E4's Azizov-style views and should not be merged or
renamed as an Azizov result.

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

The [Azizov paper](https://arxiv.org/html/2609.12980#S4.SS2) describes the
two-branch architecture (three TransformerConv layers, mean pooling and an
MLP), with source/hybrid/transpiled global-feature widths 41/54/41. The local
common-core adaptation follows those three view roles but uses the explicit
local 51/64/51 feature contract and frozen q≤9 common panel; it is not the
paper's full benchmark. The earlier logical Ridge/compiled GBR results remain
separate historical baselines and must not be relabeled as Azizov GNN results.

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
