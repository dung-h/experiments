# Results

The [QPU and measurement tables](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md)
contain 24 QPU result variants over 8,767 observations, comparisons by source
and QPack workflow, and historical simulator measurements. The current
[E6 predictor tables](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json)
separately report the new Aer and fixed-MPS predictor evaluations.
Variants include raw and calibrated estimates; they are not 24 distinct
paper methods.

For a single compact method-by-dataset overview across both tracks, see the
[reader method table](../artifacts/benchmark_v3/results/reader_method_dataset_table.csv).
It preserves the evaluation target clock, method output clock, assigned/scored
counts and fidelity/status label; analytical QPU rows are source-stratified
diagnostics, not pooled runtime scores.

QPU models are evaluated against recorded execution/service time. Simulator
comparisons specify the engine and the part of execution being timed. The
tables retain failures and missing predictions in their coverage counts.
See [methodology](methodology.md) for circuit reconstruction, splits and
differences from the original methods.

## Archived real-QPU comparisons

The current real-QPU aggregate at
[unified_method_comparisons_v2](../artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/manifest.json)
joins frozen predictions and analytical attempts only: it contains 24 methods,
210,408 attempt rows and 72 source-local paired comparisons. It also emits
QPack equal-workflow diagnostics for all 46 workflows. Pooled cross-source
error metrics were not computed; source-balanced macro summaries are
diagnostic, not a cross-source leaderboard. No timing, model fitting or
training was performed for this aggregation.

The Hyb-HANAS bounded numerical repair passed its numerical checks, but that is not a
performance pass. Raw one-circuit and shot-linear effective-cost errors are
enormous (all-source MAE `5.84e280` s and `2.34e284` s, respectively) and
their R² values are unscorable. The source metrics show 0/340 finite
predictions on Ma–Li for every Hyb variant. The outer-train log-affine
one-circuit adaptation remains weak (MAE `1.881` s, R² `−1.360` on 8,420
predicted rows). These raw/effective-cost clocks are cross-clock diagnostics,
not archived service-time predictions. See the [repair manifest](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_repair_manifest.json)
and [source metrics](../artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_source_metrics.csv).

Interpretation now also includes the snapshot audit: all 148 Ma–Li Kyoto
zero-survival cases use a nominal snapshot whose 144 ECR gates all report
error=1. The other 192 Ma–Li cases are Osaka effective-cost overflows. These
inputs and cross-clock assumptions limit attribution to Hyb-HANAS itself.
The current scorecard retains these scores with the input-state limitations
described in the [methodology](methodology.md#hardware-context-and-runtime-boundaries).

## Unified learned models

The learned scores in this section are retained earlier outputs, not the newer
aggregate. Their source-stratified slices are descriptive. The pooled rows
below are historical diagnostics only and are not a pooled source ranking.

The registry classifies these models as adaptations, not exact paper
reproductions.

| Reader label | Fidelity class | Claim boundary |
| --- | --- | --- |
| Qonductor-style unified polynomial adaptation | `adaptation` | Five common structural/shot inputs; grouped OOF on the unified ledger, not the 4,482-row source-local estimator |
| Ma--Li-style hybrid graph + metadata adaptation (V3) | `adaptation` | Seven global features and harmonized DAG encoding replace the source feature/context contract |
| Ma--Li-style hybrid graph + metadata adaptation (V3-large) | `adaptation` | Resource extension of V3, not Ma–Li original |

The [methodology audit](methodology.md#data-and-reconstruction) also records
that the executed learned inputs for Ma–Li/QPack were logical, not the
target-transpiled inputs promised by the earlier protocol. The scores below describe those
executed inputs; they do not certify compliance with the compiled-input plan.

All 8,767 observations are assigned test folds. The comparisons below use
shared test rows within each source. Models were trained on the unified
dataset with the same frozen split; breaking down results by source does
not mean training separate models.

| Shared test population | n | V3-large MAE (s) | Polynomial MAE (s) | V3-large / polynomial R² |
| --- | ---: | ---: | ---: | ---: |
| Ma–Li | 340 | 0.596772 | 0.646791 | 0.752822 / 0.756901 |
| Qonductor | 4,481 | 1.027606 | 1.394624 | 0.626512 / 0.401766 |
| QPack MCP | 3,945 | 0.943547 | 1.221289 | −0.094887 / −0.631629 |

These [source-stratified paired values](../artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1/source_stratified.csv)
show lower V3-large MAE within each source, but both methods have negative
R² on QPack: neither improves squared error over that test population's mean.
Lower MAE alone does not establish a strong model of QPack runtime variation.

Giving each source equal weight yields a mean MAE of 0.855975 s for
V3-large and 1.087568 s for polynomial. This prevents the larger sources
from dominating the summary, but does not replace the individual-source
results.

Full successful-population pooled scores are below. Different row sets mean
this table is not itself a paired ranking; even common-row pooled metrics
remain diagnostic rather than evidence of uniform transfer across sources.

| Evaluated adaptation | Predicted / assigned | MAE (seconds) | R² |
| --- | ---: | ---: | ---: |
| Qonductor-style unified polynomial V3 | 8,767 / 8,767 | 1.287598 | 0.394293 |
| Ma–Li-style hybrid graph + metadata V3 | 8,763 / 8,767 | 0.935988 | 0.761906 |
| Ma–Li-style hybrid graph + metadata V3-large | 8,766 / 8,767 | 0.973066 | 0.616421 |

On the same 8,763 test observations, V3 has MAE 0.935988 s versus
1.256830 s for polynomial. That comparison excludes four observations V3
could not predict; it does not establish performance on all 8,767 rows.
Nor does it isolate the benefit of graph structure: the hybrid model also
has two additional scalar inputs. The
[reader intersection tables](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/qpu/unified_learned_pairs.csv)
and [derived comparison report](../artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1/REPORT.md)
also contain V3-large/polynomial pairing on 8,766 rows, source slices and
cluster-bootstrap evidence. Always quote the row population with a score.

V3-large recovers prediction coverage on Qonductor rows 39/40/41. Their
average absolute error is approximately 76.3 seconds: supporting a large
graph does not imply accurate tail extrapolation. On V3's shared 8,763 rows,
V3-large MAE is approximately 0.947277 s, slightly worse than V3. The dynamic
`row4477` remains unavailable, retained in coverage and never assigned zero
error. [Tail diagnostics](../artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1/tail_diagnostics.csv)
preserve these cases.

## Analytical QPU diagnostics

| Raw route | Predicted / 8,767 | Main unavailable condition |
| --- | ---: | --- |
| Qiskit scheduled duration / QCRE critical path, single-shot and shot-scaled | 8,766 | Dynamic-control duration semantics on row4477 |
| Hyb-HANAS one-circuit effective-cost and shot-linear adaptation | 8,420 / 8,767 after bounded repair | 347 unavailable: 198 float64 overflow, 148 zero survival and row4477 control-flow unsupported |
| Scholten nominal single-circuit adaptation | 5,769 | Missing admissible nominal CLOPS_v-like context; CLOPS_h-only Ma–Li rows; row4477 |

Qiskit and QCRE single-shot predictions agree to floating-point precision on
the frozen Target. Their MAE against service time is about 5.552107 s. This
is scheduled-duration agreement between implementations, rather than two
independent predictive wins. Shot scaling and outer-train calibration have
distinct method IDs and diagnostic scores. Raw Hyb effective-cost scale can
be extremely large; it is retained without presenting it as observed service
seconds. The [archived method coverage table](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/qpu/archived_method_coverage.csv)
retains separate raw and calibrated clocks.
The repaired variants and same-row/source-local comparisons are in the
current aggregate above. QPack equal-workflow diagnostics are present for all 46
workflows; neither these nor source-local pairs produce a pooled
cross-source error score.

The source-native Ma–Li run and source-local Qonductor runs remain separate
from unified evaluation. The reader pack's
[source-native scope table](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/qpu/source_native_scope.csv)
records that the Ma–Li run used only seed 1234 and had an unstable fold with
MAE about 7,190.6 s. It is not a three-seed median.

## Simulator evidence and remaining gaps

The [simulator reader pack](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/README.md)
has separate availability, per-configuration, native-pair and blocked tables.
Counts below are cells/observations under those configurations, not necessarily
unique circuits; precision and first/warm clocks remain separate. A separate
[predictive aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/)
now reports held-out predictor results for Aer and fixed-MPS targets. It does
not replace or pool the measurement/configuration pack.

| Evidence | Current coverage | What can be claimed |
| --- | --- | --- |
| Local noisy Aer warm measurement | All 204 panel members | Local runtime labels/coverage under frozen Aer configuration |
| Azizov-related tabular predictors (historical) | 162 core members | Logical ridge warm MAE ≈0.314433 s; compiled GBR ≈0.514132 s; neither is Azizov's GNN |
| Historical local Aer graph baseline | 162 core rows / 150 QASM hashes | OOF MAE 0.387654 s; MedAE 0.036025 s; log1p MAE 0.091407; R² 0.155835; a separate older model, not an Azizov GNN |
| Maestro prediction preflight | 408 candidate-member cells over the 204-member panel | 56 predicted, 244 out of grid, 108 unavailable; preflight only, no simulator timing or score |
| Historical Maestro calibration | 330/330 synthetic cells; 4,950 timed and 2,970 untimed calls | Diagnostic-only; not the terminal ten-cell pilot and not a panel predictor accuracy result |
| CUDA-Q dense | Per precision/clock: 612 attempts, 594 successful, 18 adapter errors | Descriptive local first/warm sampling runtime; no cross-engine estimator rank |
| CUDA-Q MPS | Per clock: 612 attempts, 573 successful, 21 quality failures, 18 adapter errors | Quality-qualified timing at configured FP64 bond/cutoff policy |
| cuTensorNet | Per clock: 612 attempts, 603 successful, 9 timeouts | Native estimate-versus-actual diagnostic for the same selected contraction plan |

These counts come from the
[availability matrix](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/availability_matrix.csv)
and [per-configuration metrics](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/per_configuration_metrics.csv).
The local graph adaptation's raw OOF numbers are in its [core metrics](../artifacts/benchmark_v1/aer_mali_graph_adaptation_v1_20261001/core_metrics.csv)
and are independently summarized in the [v3 reader table](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/aer_mali_graph_c3a_diagnostic.csv);
The preflight-only status is summarized in the
[current Maestro pilot table](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/maestro_pilot_summary.csv)
and terminal [run manifest](../artifacts/benchmark_v3/simulator/maestro_candidate_runtime_v3/calibration/run_manifest.json).
Because this supplementary graph diagnostic has no method ID in fidelity registry v2, the
reader table leaves the registry method/class fields blank and explicitly
keeps it supplementary; it is not an extra registered paper-method score.
First-call overhead, precision, approximation and plan/workspace can alter
runtime substantially. Scalar contraction, dense sampling and noisy Aer
execution are different work targets even with a common source QASM.

The Maestro calibration pilot stopped after ten cells and 240 API calls
(150 timed). Nine statevector cells failed the timing-stability checks;
one MPS cell passed. Independent median/MAD calculations confirmed the
failures. Repeated SV n16/1Q/R256 measurements ranged from 0.0149 s to
7.3553 s; the cause is unresolved. No calibration was accepted, and no
panel prediction score was produced. See the
[run manifest](../artifacts/benchmark_v3/simulator/maestro_candidate_runtime_v3/calibration/run_manifest.json)
and [blocked-method table](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/blocked_unavailable.csv).

The broader paper-wide simulator comparison remains incomplete: this local
panel does not reproduce Azizov's two-backend/four-optimization-level scope,
the Family-Aware paper's joint quality/runtime target, or Maestro's
paper-exact Composer estimator. q10–16 remain outside predictor fitting and
accuracy. The historical Azizov Ridge/GBR rows remain tabular independent
reimplementations. Maestro's pre-existing 220-cell component calibration /
3,300 warm observations is distinct from its parser/prediction preflight.
Simulator timing was not run during that preflight and no candidate score was
produced; 35 panel members failed parser/width checks. Pasqal's analog pilot
completed 18/18 cells, with four quality failures and an unpromoted fit; it
remains a separate workload. See the
[blocked table](../artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/blocked_unavailable.csv).

## Local simulator predictors

The local held-out predictor results are in the
[E6 aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/).
It contains one score table per target, exact-hash coverage, 28 paired
comparisons (18 Aer, 10 MPS), method cards, and source hashes. Each interval is
based on 10,000 paired exact-QASM-hash bootstrap replicates. No row-level
predictions or errors are pooled across engines.
These are pointwise exploratory intervals over saved OOF fits; the 28
comparisons have no multiplicity adjustment and do not include retraining
uncertainty. See the [scientific review](scientific_review.md) for the current
method matrix and report claims.

| Target and evaluated method | Scored / assigned hashes | MAE (s) | R² | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Aer noisy warm execution — Azizov-style transpiled-view GNN, 3-seed median | 150 / 150 | 0.3008 | 0.5206 | Best listed GNN view on this Aer cell; adaptation, not paper-exact |
| Aer noisy warm execution — XGBoost, transpiled view | 150 / 150 | 0.2970 | 0.2889 | Similar MAE to the GNN; paired difference unresolved |
| Aer noisy warm execution — Azizov-style hybrid-view GNN, 3-seed median | 150 / 150 | 0.4064 | 0.0914 | Local adaptation |
| Aer noisy warm execution — Azizov-style source-view GNN, 3-seed median | 150 / 150 | 0.4497 | 0.0927 | Local adaptation |
| CUDA-Q MPS FP64, bond-16 warm state — source-DAG graph, 3-seed median | 144 / 150 | 0.3024 | 0.9375 | Fixed MPS cell; includes two finite quality failures |
| CUDA-Q MPS FP64, bond-16 warm state — Family-Aware-inspired residual | 144 / 150 | 1.0009 | 0.5562 | Runtime-only fixed-cell adaptation, not the joint paper method |
| CUDA-Q MPS FP64, bond-16 warm state — family-agnostic ablation | 144 / 150 | 1.0813 | 0.4823 | Matched ablation |

On Aer, the transpiled-view GNN MAE exceeded XGBoost's by `+0.0037 s`
(95% interval `[−0.1726,+0.1324]`), so this comparison does not establish a
winner. The source-view GNN had `+0.1489 s` higher MAE than the transpiled
view (`[+0.0284,+0.3039]`) on this panel. That is evidence for these views on
one FakeSherbrooke/Opt1 cell, not proof that transpilation or GNNs generally
improve simulator prediction.

For the MPS cell, the residual model's observed MAE difference versus the
family-agnostic ablation was `−0.0804 s` (`[−0.2852,+0.0804]`); the interval
crosses zero, so an incremental family benefit is not established. Its MAE
was `+0.6984 s` higher than the source-DAG graph model (`[+0.1878,+1.4256]`).
There are 150 assigned hashes, 144 finite labels, 142 quality-pass labels,
two finite quality-failed labels and six unavailable targets. On the
quality-pass subset, graph MAE is 0.3037 s and residual MAE is 1.0081 s.
These are separate engine/target results, not a single simulator leaderboard.

## Conclusions and remaining work

The evaluated hybrid model has lower MAE than polynomial on shared QPU
test rows, but large-circuit errors remain substantial and neither model
explains QPack variation well. Scheduled duration also leaves a substantial
gap to recorded service time. For simulators, runtime depends on the engine,
timing boundary and output-quality requirement.

Predictor evaluation is complete for the bounded local Aer and fixed-MPS
panels described above, but the broader paper-wide simulator comparison is
not complete. Maestro has no accepted prediction score, the original
Family-Aware target is unavailable, q10–16 are not in predictor accuracy, and
V4 has not been trained. These limits remain despite passing implementation
checks. [Reproduction](reproduction.md) explains which tables can be rebuilt
and which experiments have not been reproduced.

The unified rebuild recomputes learned-QPU metrics from frozen predictions,
archived analytical-QPU metrics from frozen attempt shards, and local
simulator measurement summaries from pinned raw files. The v2 reader-table
builder separately recomputes the local graph summary from its raw OOF predictions. No
simulator timing is repeated by either path. Held-out common-panel evaluation
for the remaining blocked simulator predictors is pending, so the unified
rebuild still reports `PARTIAL`.
