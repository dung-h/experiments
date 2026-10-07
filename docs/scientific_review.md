# Scientific review

Review date: 2026-10-07. Three independent read-only checks examined the selected
export at `755e97892ecacb0d676d031663567efa11e008d4`: QPU numerical evidence,
feature/fitting contracts, and simulator evidence. Numerical checks used separate
code without importing the report builder. No model was fitted, inference run,
simulator timed or QPU contacted. The corrections below change maintained prose
and derived reader metadata, not saved predictions, labels or frozen splits.

This review concerns the saved scientific evidence in the source export.
Earlier package replay used a larger local evidence selection. This public
subset excludes third-party QPU observations and does not reproduce those
private-input calculations from a fresh clone. Older compiled-input comparisons remain
in the development archive, not the current logical-input comparison.

## Dataset and comparison decisions

The canonical ledger retains 8,767 archived observations: Ma–Li 340, Qonductor
4,482 and QPack 3,945. The current common panel contains 7,350 of those rows,
selected by logical-input availability. Neither number denotes distinct circuits.

| Source | Current rows | Input qualification |
| --- | ---: | --- |
| Ma–Li | 340 | Exact source logical QASM; archived target is the mean of three execution-time records at 1,024 shots |
| Qonductor | 3,065 | 230 archive-supported logical-recipe adaptations and 2,835 unverified candidate-recipe sensitivity rows |
| QPack | 3,945 | Six reconstructed QAOA structures with representative angles; original optimized angles, submitted QASM and routing are absent |

The excluded 1,417 rows are all Qonductor observations. Their physical-circuit
evidence remains in the older compiled panel. Logical-input selection changes
the evaluated distribution, so these two panels do not isolate a representation
effect. Archive-supported recipes are not proven exact historical logical
instances: an ideal zero-input distribution check does not establish arbitrary-
input unitary equivalence.

Reconstruction supplies inputs, not new runtime labels. QPack milliseconds are
converted to seconds; source job/service boundaries remain heterogeneous.
Source slices are diagnostics of the same unified fits, not separate training
runs. The [QPU guide](real_qpu_benchmark.md) and
[feature dictionary](data_and_feature_dictionary.md) explain the joins and tiers.

The simulator core has 162 source members / 150 exact-QASM hashes; its frontier
adds 42 members, giving 204 members / 191 hashes at q2–q16. Current predictor
panels use q2–q9. Aer execution, dense 32-shot sampling, MPS state extraction and
TN contraction remain different targets. Hardware, precision, shots, quality
threshold, target revision and clock are part of comparison identity.

## Fitting and method identity

The QPU panel uses 206 transitive groups: Ma–Li 148, Qonductor 52 and QPack 6.
Five outer test folds are shared across methods. Group membership, including
equality links through excluded bridge rows, is respected. Four grouped inner
folds partition each outer-training population.

- Graph and matched MLP fit on inner folds 1–3 and select the epoch using
  inner fold 0 over the recorded 500-epoch run. They do not refit on validation
  afterward. Seeds are 42, 1234 and 31415; predictions are per-row medians of
  all three. Recorded training uses CUDA FP32 on an RTX 5070 Ti.
- Both start from a 51-position raw schema and consume the 40 transformed
  globals retained by the fit-only positive-column-sum mask in every fold.
  The graph additionally consumes 178 node positions; 167 vary in the fit
  population, while 11 constant positions are
  zeroed. Nominal-snapshot T1/T2 are attached by logical wire index. Shots,
  source IDs, explicit backend IDs and numerical gate errors/durations are not
  node/global inputs. These T1/T2 values are not job-day calibration.
- Polynomial uses five compiled/submitted inputs: the upstream `swap` field
  counts CX/CZ/ECR, followed by operand-stack depth, touched width, shots and
  circuit count=1. Degrees 2/3/4 are selected by mean inner R²; all folds choose
  degree 2 and refit on full outer-train. No log transform, scaling or constant-
  field pruning is applied; finite negative predictions are retained.
- Hyb-style Ridge predicts `log1p(service_time)` from cost/shot inputs, with
  scaling inside inner-fit and alpha selected by source-balanced inner MAE.
  Its `expm1` output is calibrated service time, not raw cost. Affine
  schedule/throughput calibration also fits only outer-train.

This is a shared-row, shared-outer-test benchmark with declared method-specific
inputs and selection procedures. Polynomial and analytical inputs follow a
compiled/submitted route, unlike the logical-recipe neural route. Graph versus
matched MLP tests the full node/graph branch, including T1/T2 context; it does
not isolate topology alone. These unified implementations are adaptations,
not faithful paper reproductions. Native/source-local scores remain separate.

The execution record notes inspection of aggregate fold-1 fit/validation/test
target summaries before all predictions were complete. No responsive protocol
change is reported, and the implementation excludes outer-test labels from
fitting, preprocessing and selection. Nevertheless, this is a retrospective,
exploratory benchmark, not a fully analyst-blind prospective preregistration.

## Independently verified QPU findings

All 30 neural cells, 44,100 seed predictions, 470 graph files, fold memberships,
fit-only transforms and input signatures reconcile. All 37 summary rows, 968
analysis metric rows, 62 pair rows and 120 tail records reproduce. Fourteen
10,000-replicate grouped bootstraps also match independently.

| Unified adaptation | Scored / assigned | MAE (s) | R² | Source-balanced MAE (s) |
| --- | ---: | ---: | ---: | ---: |
| Logical graph + global features | 7,350 / 7,350 | 1.2186 | 0.4291 | 1.1351 |
| Matched global MLP | 7,350 / 7,350 | 1.7322 | 0.1873 | 1.5703 |
| Five-feature polynomial | 7,350 / 7,350 | 1.4006 | −0.5867 | 1.1928 |

Graph minus MLP MAE is −0.5136 s, with 95% grouped interval
[−0.9074,−0.1748]. This supports a benefit of the full graph/node-context
branch on this panel. Graph minus polynomial is −0.1819 s,
[−0.4883,+0.1968]; that comparison is unresolved. Polynomial has lower
source-slice MAE on Ma–Li and Qonductor, while graph has lower MAE on QPack.
There is no universal winner across sources or error criteria.

Polynomial p99/max absolute errors are 4.0035/187.0537 s, versus graph
7.0822/76.0941 s. Its negative R² reflects squared-error sensitivity to tails;
it is not, by itself, evidence of an implementation error.

The 4,515-row sensitivity subset excludes only the 2,835 candidate-recipe rows
and reuses the same full-panel fits. Graph/MLP/polynomial MAEs there are
0.7158/1.3756/1.1636 s. It is not an all-exact-input panel or a new training run.
All held-out exact transformed global, graph and combined signatures are absent
from fit and fit-plus-validation populations. Unique signatures do not establish
unseen families, backends, topology or quantum semantics.

Nominal Hyb Ridge predicts 7,202/7,350 rows; its composite variants predict
7,350. Nominal Scholten affine predicts 5,199. Their successful-subset MAEs
must not be ranked against full-panel scores without a common-row comparison.
All 7,350 labels are observed even when a method cannot predict some of them.

## Independently verified simulator findings

All 401 populated metric fields in the 49 current scored reader rows match
their evidence. Engines are label sources, not additional predictor methods.

The public standard-library checker now recomputes the 49 predictor rows from
retained simulator labels and OOF predictions without importing report-builder
metrics. This check corrected the QCSim controls' emitted-prediction and attempt
counts to 150; their scored counts remain 143 and no error score changed.
It does not repeat the private QPU audit or reproduce bootstrap intervals.

- Dense FP32 and FP64 each compare nine predictors on 150 hashes. Random
  Forest has the lowest MAE, 0.000821/0.000813 s; work-scaled Ridge has the
  highest R², 0.8728/0.8649. The empirical feature model does not identify a
  physical launch-overhead intercept.
- Aer transpiled GNN MAE is 0.3008 s versus XGBoost 0.2970 s; their paired
  interval includes zero. Source versus transpiled GNN differs in this local
  context, not proof of a general GNN advantage.
- Recovered fixed-MPS graph MAE is 0.2961 s on 150 finite labels and 0.2908 s
  on 147 quality-pass labels. The older E6 target has 144 finite and 142
  quality-pass labels and is a separate revision.
- Joint MPS assigns 900 hash×rung rows but measures 840; 60 remain unavailable.
  Its 2,700 technical attempts comprise 2,520 successful, 72 timeout and 108
  adapter-error records. Family-conditioned versus agnostic runtime MAE is
  0.31163 versus 0.30752 s; delta +0.00411 s has interval
  [−0.02660,+0.03923], unresolved. Brier scores are 0.08159/0.08631, but the
  predicted-rung quality violation rate is 64/140=45.7%, so the quality head
  is not a dependable selector. Superseded 900-scored-row entries remain
  historical, not current accuracy evidence.
- Optimizer-off QCSim compares four predictors on 143/150 supported hashes;
  component MAE is 0.000104 s. The target is the pooled median of 15 finite
  engine-reported samples per hash, with a fresh process for each call—not
  persistent-process warm execution or process wall-clock. The explicit
  panel-target contract and labels agree on this reduction; a generic protocol
  field instead describes medians within/across sessions. This inconsistency
  is disclosed, not used to silently relabel data. It neither rescues the
  failed optimizer-on pilot nor reproduces the closed Composer/selector.
- cuTensorNet has 603 successful / 612 attempted same-plan sessions. Median
  estimate/warm ratios are 3.7340 for core and 4.1324 for frontier. These
  contraction estimates do not predict full simulator wall-clock.
- Pasqal has 18 analog cells, 17 quality-pass and one quality-fail. Four failed
  stage/repetition records belong to `quench|4x4|chi16`, whose fidelity is
  0.95147 against a 0.99 candidate threshold. The reference threshold is
  0.9999. Its fitted formula remains unpromoted; there is no generic digital-
  panel accuracy score.

See [results](results.md) and the two reader tables for coverage, tails and
paired comparisons. Intervals condition on saved fits, omit refitting uncertainty
and are exploratory/pointwise, not multiplicity-adjusted. Six QPack groups,
small simulator circuit populations, three neural seeds, reconstruction
assumptions and nominal calibration constrain generalization.

## Corrections and reproduction boundary

No numerical or fitting defect found here requires rerunning training or
measurement. Reader corrections distinguish direct learned service-time
predictions from calibrated analytical outputs, identify actual method inputs,
keep observed-label counts separate from prediction coverage, classify calibrated
Qiskit as an adaptation, restore Maestro's process-isolated clock and count
Pasqal failures per cell rather than timing record. Saved experiment records
and historical reports remain unchanged in the local evidence archive.

The historical source-export payload at `cdc5dee71991d8995da093a5a59d12154306b898` passed a
fresh clone, Python 3.10.21 requirements installation, 315 tests and four subtests,
restored partitions/evidence, inventory and two byte-identical table builds.
Those test counts describe the historical source export, not this publication's
test selection. The current [reproduction guide](reproduction.md) describes
the smaller public checkout, file-level verification and saved simulator-score checks.

Earlier local saved-prediction replay was verified; this public subset cannot
recompute QPU results without the excluded row-level inputs. Fresh source
extraction, model refits and simulator retiming were not performed for this
publication pass. Historical exact code versions and complete upstream
archives are excluded or unavailable, as disclosed in [reproduction](reproduction.md).
The fitting audit inspected available source helpers against frozen hashes;
the publication includes inspected preprocessing and fitting helper bytes,
but does not independently re-execute extraction or model fitting.

This public selection excludes third-party QPU labels and their row-level
derived evidence. It retains aggregate findings and locally measured simulator
records. See [data availability](distribution.md) for that boundary. No new
training or measurement is needed to publish this report selection; publication
does not establish fresh end-to-end reproduction.
