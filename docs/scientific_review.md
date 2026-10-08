# Scientific review

## Real-QPU dataset checks

The public real-QPU population is 4,515 observations: Ma–Li 340, Qonductor 230
and QPack 3,945. Selection is based on source linkage and qualified inputs,
not on prediction errors. See the [source-filtering report](data_preprocessing.md).

The prepared input manifest confirms the source/tier counts, 165 groups,
positive finite observed labels, representation/snapshot joins and new five-fold
outer/four-fold inner assignments. All sources occur in each outer test and
inner validation set; no conservative group crosses a partition.

Qonductor's original IBM source audit covers 4,482 one-circuit jobs. Submitted
physical-QASM joins pass for all of them, but only 230 have an admissible
archive-supported logical recipe for this benchmark. The 32 distinct accepted
physical circuits pass zero-initial-state measured-output checks. This does
not establish exact historical logical DAGs or full-unitary equivalence.
QPack retains 3,945 source-backed MCP structural reconstructions without
recovering per-iteration optimized angles or historical physical submissions.

These checks support a declared reconstruction-qualified benchmark, not a
claim of 4,515 exact recovered historical circuits. QPack contributes most
rows and only six structural groups. Nominal snapshots and heterogeneous
source execution boundaries remain limitations.

## QPU scores are not yet a finding

Fresh Graph/MLP/regression fits and fold-local analytical calibrations are
running under the prepared 4,515-observation contract. Complete OOF scores,
required seed reduction, failure coverage, source slices and paired intervals
must pass final QA before publication. No previous-population error scores,
masks or intervals are relabeled as this run's results.

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

## Reproduction boundary

The public verifier checks the current file inventory and documentation;
the numerical checker recomputes saved simulator scores. Neither reconstructs
the excluded third-party QPU observations or validates a fresh QPU fit from
a Git clone. Retained source helpers and their original paths explain how the
study was run; complete source reconstruction and model refitting require
external inputs and the frozen environments.

This publication changes the real-QPU presentation and dataset documentation,
not local archived source evidence or active run contracts. Simulator labels,
predictions and scores are unchanged. See [reproduction scope](reproduction.md)
and [data availability](distribution.md).
