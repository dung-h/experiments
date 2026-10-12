> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Benchmark interpretation addendum

Scientific Freeze v2 remains the experimental authority. This package reuses
the completed 4,515-observation benchmark and 21/21 RQ4 grid; it adds no training,
inference, runtime measurement, QCEC or execution authorization. Historical
reports and producing code are unchanged. Decision: **B — reporting fix only**.

## Direct answers

1. **Why41 versus 51?** The public extractor has 44 gate counters +2 size +5
   structural positions. Its positive-sum refinement removes unsupported
   positions. The pinned simulator corpus supports 34 gate counters, giving 41.
   The paper describes this effective schema, not a fixed raw 51-vector input.

2. **Which ten?** Zero-based `(4,id), (5,u0), (33,csx), (34,cu), (38,rc3x),
   (39,c3x), (40,c3sqrtx), (41,c4x), (42,xx_plus_yy), (43,ecr)` have zero
   support in the reconstructed original simulator population. All 1,510 QASM
   and four timing CSVs were checked against the pinned upstream Git blobs.

3. **Why40 now?** `rccx` at position 37 additionally has zero support in the 340
   released real rows and all 4,515 qualified rows. The upstream real notebook's
   stored 12-column example and constructor default 41 do not establish the
   authors' executed real tensor. Every parent gradient-fit transform
   retains 40 varying inputs. RQ4 widths are fit-dependent: final 11/22/40 and
   selection 11/17/22/40. The 257-transform table records each exact fit.

4. **Do extra positions contribute?** Not the eleven zero fields in this
   cohort. rccx is active in 12 original simulator rows, but isolated benefit
   has not been tested. Other fields lost only at OOD test are a distinct
   support limitation, particularly18 globals in Random.

5. **Keep Graph/MLP results?** Yes, as the declared architecture/core
   adaptations. Their raw schema/support mechanism follows upstream; train-only
   per-feature standardization, grouped training, nominal calibration and
   reconstructed inputs differ from exact paper execution. Those differences
   are disclosed, not repaired by retraining to match a dimension label.

6. **Hard-code41?** No. It would retain all-zero rccx in the parent and override
   population-specific refinement without restoring full original fidelity.

7. **Which contribution results are usable?** Fixed-model interventions for
   reliance; retrained coarse topology/static/critical controls for incremental
   value in their historical populations; QWalk and strict transfer for limited
   negative-transfer diagnostics. Paper component tables are external evidence,
   not locally reproduced scores. No topology-only causal benefit on 4515 is established.

8. **Which dimensions are answered?** The 120-row matrix covers 15 method/context
   scopes ×8 dimensions. Local implementation fidelity, native clock meaning,
   saved in-distribution errors and many matched baseline contrasts are answered.
   Contribution, broad generalization/scaling and prospective reliability are
   usually partial. Original Scholten accuracy, systematic QPU shot transfer,
   several cross-context tests and identified Pasqal prediction remain unanswered.
   ANSWERED means a bounded local question, not universal validity.

9. **What remains genuinely open?** Full-model topology attribution, incremental
   benefit of individual global/calibration fields, shot extrapolation, causal
   hardware-generation transfer, robust prediction intervals, broader engine
   transfer and a faithful original-Scholten kernel experiment. See unresolved
   questions below; failure results are answers, not missing experiments.

10. **Amendment needed now?** No corrective experiment is justified by the
    feature-count discrepancy or these interpretation findings. No amendment
    or rerun is proposed for the frozen scope. A future claim about shots,
    topology or causal hardware effects would require a separately designed
    study; existing results cannot support those stronger claims.

## Read the evidence

| Deliverable | File |
| --- | --- |
| Exact feature/schema/source audit | `mali_feature_schema_audit.md` |
| Raw51 ledger, original support and notebook/default discrepancy | `global_feature_ledger.csv`, `original_corpus_support.json`, `upstream_pipeline_evidence.json` |
| All fit widths and replay check | `effective_input_widths.csv`, `feature_validation.json` |
| Zero-field concentration and held-out support | `masked_feature_concentration.csv`, `generalization_field_support.csv` |
| Feature reliance/value/topology synthesis | `feature_contribution.md` |
| Track A fidelity/target alignment | `track_a_fidelity.csv` |
| Both tracks, eight-dimensional answers | `benchmark_question_matrix.csv` |
| Generalization decomposition | `generalization.md`, `generalization_decomposition.csv`, `shot_diversity.csv` |
| Matched prediction skill | `qpu_predictive_skill.csv`, `qpu_source_balanced_skill.csv`, `generalization_predictive_skill.csv`, `analytical_predictive_skill.csv`, `simulator_predictive_skill.csv` |
| Clock, baseline and context interpretation | `runtime_estimator_interpretation.md` |
| Input/output hashes and focused verification | `manifest.json`, `validation.json` |

Tables contain point metrics from saved predictions. Existing source-linked
paired bootstrap intervals remain authoritative; no intervals for new skill
ratios are invented. Source-balanced skill is a ratio of macro MAEs, not an
average R². QPU/source-context advantages, eligible subsets and overlapping
RQ4 cases are kept explicit. Simulator clocks remain separate.

## Reporting changes adopted by this addendum

- Replace ambiguous “51 global inputs” with raw 51/post-mask40 for the parent,
  and exact fit-dependent widths for RQ4. The node 178 schema is separate.
- Label Ma–Li architecture/core adaptation; report fit-only versus upstream
  artifact-wide/scalar normalization rather than implying exact preprocessing.
- Label the current generic-depth route **Scholten-inspired generalized proxy**;
  original Scholten remains not evaluable. Do not rename ordinary depth D_eff.
- Keep QCRE/Qiskit one-shot, deterministic shot-scaled and learned service-time
  bridges distinct. Hyb reliability/resource cost is not literal observed runtime.
- Present positive median skill with classical/source-aware comparisons and
  uncertainty; preserve pooled/source-balanced/equal-group ranking reversals.
- State QNN/Random catastrophic negative skill, Kyoto collisions and limited
  size support. Do not call21-case completion universal generalization.
- Keep historical topology reliance separate from retrained topology benefit;
  do not infer topology causality from Graph versus global MLP.
- Keep Aer, dense sample, MPS state, QCSim engine time, scalar TN contraction
  and analog TDVP as separate contexts; no universal simulator leaderboard.

These are interpretation corrections, not new producing authority. The
original scientific reports remain pinned and readable; the addendum supplies
the more precise current feature/fidelity explanation without rewriting history.

## Unresolved questions and scope

| Question | Status / consequence |
| --- | --- |
| Does native DAG topology add value beyond matched node/global/calibration information on 4515? | NOT ANSWERED; saved coarse controls do not resolve a gain |
| Which individual global or calibration fields add retrained value? | PARTIAL; fixed-model reliance and descriptor-block evidence are not full individual ablations |
| Can estimators extrapolate unseen QPU shots? | NOT ANSWERED; six Qonductor shot values are observational/confounded; separate CUDA-Q sweep is not QPU transfer |
| Is hardware generation an isolated transferable effect? | NOT ANSWERED; source/workload/clock/shot and nominal-index calibration confound it |
| Are intervals prospective prediction guarantees? | NOT ANSWERED; paired conditional error bootstrap is not calibrated predictive coverage |
| Can Scholten original D_eff be evaluated faithfully on generic circuits? | Not on the present verified inputs; current proxy does not answer it |
| Does local family conditioning make quality-constrained selection safe? | ANSWERED negatively for the saved selector;64 failures/140 plus10 unassessed |
| Are analog cost coefficients identified or TN internals established? | NOT ANSWERED; keep companion/native-plan scope |
| Would strict original-paper software/data reproduction change scores? | Not established; new reproduction scope would not invalidate honest adaptation results |
| Was the authors' real input40, stored notebook12 or default-model41? | Not established from the public processed artifacts; released-corpus support40 and current fit40 are verified separately |

The missing answers restrict claims. They do not require corrective reruns to
retain the current bounded benchmark. No future experiment is authorized here.
Third-party row-level QPU evidence and local paths are not automatically an
approved release payload; publication rights/metadata/export remain separate.

## Reproduction of this analysis

Run from the authoritative research workspace with its pinned inputs and Python
environment. This is saved-evidence replay, not a self-contained public QPU refit.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/server/Documents/.venv-qonductor/bin/python -B \
  benchmark_v1/scripts/validate_benchmark_interpretation.py
```

The builder `interpret_frozen_benchmark.py` renders apply_patch text and refuses
to overwrite divergent derived outputs. `original-support` is deterministic
static preprocessing verification of pinned QASM, not a simulator run. The
validator replays derived tables and checks pinned upstream/source hashes;
it does not repeat static parsing, fit models or call simulation engines.
