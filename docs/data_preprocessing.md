# Data construction and preprocessing

The QPU benchmark predicts archived execution/service time. Its canonical
ledger contains 8,767 observations, while the current logical-input comparison
contains 7,350 eligible observations. Circuit reconstruction supplies a model
input; it never supplies a replacement runtime label. This page describes the
source joins, selection, representation assumptions and actual handling of
missing inputs. The [feature dictionary](data_and_feature_dictionary.md) gives
the ordered feature schema, and the [QPU guide](real_qpu_benchmark.md) gives the
current comparison and its limits.

## Source records, identities and labels

The canonical ledger (not bundled)
preserves `source_id`, `source_row_index`, `canonical_row_id`, `source_row_id`,
native and converted targets, archived metadata, circuit/workflow identities,
and the serialized source row with its SHA-256. Its
manifest (not bundled)
records source-extract and split hashes. The source extracts and their complete
upstream circuit archives are external inputs, not additional files bundled
by that manifest.

Construction joins each source extract to its source split in original row
order: lengths must match, and `source_row_index` must equal the consecutive
source row numbers. The canonical ID is `source_id|row<index>`. Consequently,
the index is an identity within the pinned extract, not a new globally stable
database key. Preserve the extract hash, source row ID and circuit digest
when tracing it back to the source. The canonical target column is
`target_seconds`, and its clock is
`archived_observed_service_execution_time`.

| Source | Canonical observations | Label conversion and source join | Circuit evidence |
| --- | ---: | --- | --- |
| Ma–Li | 340 | The extract's `time_taken` is the mean of three archived IBM `Result.time_taken` values for one circuit at 1,024 shots. It is already in seconds. Source circuit name/backend and source split identify the logical QASM member; its bytes must match `qasm_bytes_sha256`. | Exact logical, pre-transpilation QASM was used by the run. Complete original QASM is not distributed here. |
| Qonductor | 4,482 | `taken_time_seconds` remains in seconds. The direct association follows the SQLite job/circuit relationship to the submitted ZIP member, then checks its SHA-256 against the canonical row. Each retained job contains one circuit. | Exact submitted physical QASM was verified for every source row; this does not recover original logical QASM. |
| QPack MCP | 3,945 | `circuit_execution_duration_ms / 1000` converts one measured QAOA circuit's IBM `Result.time_taken` to seconds. Source row and workflow identity link each optimizer evaluation to its configuration. | Recorded problem/size/depth/QAOA layer configuration supports six structural recipes; optimized angles, exact submitted QASM and routing are absent. |

The [source registry](../protocol/data_sources.json) and
[canonical materializer](../methods/real_qpu/materialize_canonical_corpus.py)
record extraction boundaries. QPack's 46 optimizer workflows are groups of repeated
evaluations; their total duration is not the label. IC/RH VQE evaluations,
which aggregate multiple measurement circuits, are outside this one-circuit
ledger. Queue, job-wrapper, optimizer and total-workflow durations are not
substituted for circuit execution time. A common IBM field name supports a
shared prediction target but does not establish identical provider-internal
overhead across sources or historical collection periods.

The direct Qonductor job/circuit provenance audit (not bundled)
records 4,482 successful DB/ZIP/digest associations and zero failures. The
database has 4,613 one-circuit jobs across providers; the additional 131 are
outside the frozen IBM extract. Five failed legacy `sum_*` feature-alias checks
are a separate issue: they do not invalidate those five job-to-QASM joins.
The canonical columns retain the archive values; a method that requires
parsed circuit features must use its declared extractor instead of repairing
those columns by assumption.

## Logical reconstruction and current cohort selection

The current cohort is selected by `status=available` in the pinned logical
feature ledger, without selecting by target value or prediction error. The
feature manifest (not bundled)
accounts for all 8,767 assignments: 7,350 available and 1,417 unavailable.
The panel manifest (not bundled)
pins the selected IDs and split construction.

| Current tier | Rows | Assumption and permitted interpretation |
| --- | ---: | --- |
| `source_has_logical_input` | 340 | Ma–Li source logical QASM is loaded and its exact archived byte hash checked. |
| `archive_supported_recipe_adaptation` | 230 | Qonductor submitted physical QASM hash matches a member of the separate benchmark archive. Its filename identifies family and logical width; MQT Bench 1.0.3 reconstructs that recipe. This is a supported recipe adaptation, not the original pre-transpilation bytes. |
| `candidate_recipe_sensitivity_only` | 2,835 | A pinned generator and reviewed measurement/register rules yield a candidate logical recipe. Its historical instance identity is unverified. |
| `qpack_reconstruction_qualified` | 3,945 | Six recorded MaxCut QAOA structures are rebuilt with representative angles. The input is qualified structural reconstruction, not an exact optimized or routed instance. |
| **Current panel** | **7,350** | **340 + 230 + 2,835 + 3,945; all labels remain from the archive.** |

The remaining 1,417 observations are Qonductor rows whose logical input is
unavailable. They remain in the canonical ledger and older physical/compiled
evidence. They do not enter the current logical-input cohort. The selection
changes the workload distribution, so comparing the older and current panels
does not isolate an effect of circuit representation.

Qonductor reconstruction begins by parsing the submitted QASM and checking
measurement destinations, register allocation, measured wires, terminal
measurement status, reset and dynamic control. Reviewed families can propose
width from one fully written measurement register; generator-specific rules
account for an unmeasured wire where appropriate. QFT uses the written
register rather than total classical allocation. Non-bijective measurements,
partially written registers, unreviewed multiple written registers,
unsupported measurement lifecycle and archive-width conflicts do not become
unique logical identities. Backend capacity and archived aggregate width
are not inverse-transpilation rules.

The [archive reconstruction protocol](../protocol/qonductor_logical_recovery.json)
records a narrower archive-evidence partition: 230 reproducible recipe rows,
14 graphstate identities with structural-only evidence, and 4,238 rows without
that archive join. This partition and the later candidate eligibility tiers
describe the same 4,482 observations under different evidence questions; their
counts must not be added. The current candidate tier requires a generated,
hash-pinned recipe and successful feature materialization. A candidate-width
hypothesis or family name alone is insufficient. Ideal zero-input output
checks do not prove arbitrary-input unitary equivalence or historical-instance
identity.

QPack reconstruction uses the pinned LibKet/QPack structure and recorded size
and layer count. Representative nonzero rotations are `rz(0.3)` and `rx(0.2)`.
The neural schema records operation identities and structural features rather
than numerical angles. Nominal analytical compilation sees the values, so cancellation,
synthesis and schedule are representative-angle results. They cannot be
claimed as each optimizer iteration's actual compiled schedule. The
[logical-input review](../protocol/logical_input_review.json)
retains the reconstruction qualification. The detailed original replay audit
and optimized-angle payload are not packaged; a provenance reference to them
does not create a runnable source extraction path.

The 4,515-row sensitivity subset removes only the 2,835 candidate-recipe rows.
It reuses predictions fitted on the full 7,350-row cohort. Its remaining
230 archive-supported and 3,945 QPack rows are still reconstructions, and it
is neither an all-exact-input cohort nor a separate refitting experiment.

## Packaged joins and representation checks

The packaged panel (not bundled) joins to
targets (not bundled), the
neural feature ledger (not bundled),
input index (not bundled),
outer split (not bundled) and
inner split (not bundled)
by `canonical_observation_id`; the neural ledger calls the same key
`canonical_row_id`. The panel stores source, reconstruction tier, logical
instruction digest, graph path/hash, nominal snapshot/property hash and group.
The target sidecar adds archived backend, shots and seconds. These fields
have different roles: a graph digest is an input identity, while the canonical
ID identifies an observation that may share that input with other observations.

The retained [neural runner](../methods/real_qpu/train_mali.py)
checks duplicate IDs and exact 7,350-to-7,350 joins, verifies source and ledger
globals, requires finite inputs and positive finite targets/shots, checks graph
file hashes and paths, and validates the 178-position node shape and directed
edge indices. The input receipt (not bundled)
records 470 distinct graph files for 7,350 observations, including a largest
graph of 25,080 nodes and 38,322 edges. Fewer graph files than observations
does not mean that labels have been averaged or observations deduplicated.

The neural route uses logical recipes before transpilation. The polynomial
and timing routes instead use pinned compiled/submitted inputs: Ma–Li and
QPack are nominally target-compiled, while Qonductor uses its exact submitted
physical circuit. Thus a shared cohort and split do not give all methods
identical input information. The older compiled-input and historical mixed-
stage results retain their own identities and are not silently relabelled as
the current logical-input experiment.

## Width, families, timestamps and shots

The [dataset profile](../results/real_qpu/analysis/dataset_profile.csv)
reports current logical-feature widths and depths, rather than treating
source metadata fields as interchangeable with those quantities.

| Source in the current panel | Rows / transitive groups | Logical allocated width: min / median / max | Logical depth: min / median / max | Recorded shots: min / median / max |
| --- | --- | --- | --- | --- |
| Ma–Li | 340 / 148 | 9 / 79 / 127 | 205 / 294 / 6,068 | 1,024 / 1,024 / 1,024 |
| Qonductor | 3,065 / 52 | 3 / 6 / 103 | 5 / 20 / 1,508 | 1 / 4,000 / 20,000 |
| QPack | 3,945 / 6 | 2 / 4 / 7 | 14 / 41 / 131 | 4,096 / 4,096 / 4,096 |

`num_qubits` in the neural schema is allocated quantum width, not active width,
physical backend capacity or quantum-plus-classical allocation. Polynomial
`num_qubits` instead counts touched qubits under the upstream extractor.
The canonical Qonductor `width_qubits` copies legacy `sum_qubits`: its range
is 7–136,337 even within the current selected rows. It must not be interpreted
as logical width or hardware capacity. This archived metadata remains for
traceability; the current logical-width evidence comes from the parsed recipe
feature ledger. The five legacy-alias failures make the distinction explicit.

The following family counts are obtained by joining current panel IDs to the
canonical `family` field. They describe source names and do not certify semantic
equivalence across generators.

| Source | Current family distribution, observations |
| --- | --- |
| Ma–Li | qft 46; qftentangled 47; qnn 26; qpeexact 35; qpeinexact 36; qwalk-noancilla 2; random 30; realamprandom 40; su2random 38; twolocalrandom 40 |
| Qonductor | ae 295; dj 251; ghz 227; grover-noancilla 209; qft 198; qftentangled 204; qpeexact 196; qpeinexact 219; qwalk-noancilla 230; random 229; realamprandom 242; su2random 187; twolocalrandom 203; wstate 175 |
| QPack | MCP 3,945; width counts are 2:860, 3:876, 4:888, 5:823, 6:267 and 7:231 |

The full Qonductor ledger additionally contains graphstate 228,
portfolioqaoa 211, portfoliovqe 222, qaoa 195, qnn 213, vqe 197 and 151
observations under 29 `circuit-*` source names. Those counts sum to the 1,417
rows outside the current panel. Selection therefore removes whole source
family categories as well as changing the width distribution.

All 4,482 canonical Qonductor rows have `timestamp_utc`, spanning
2023-07-16T15:07:55.627934+00:00 through
2023-10-21T11:22:17.130022+00:00. Its 3,065 selected rows span
2023-08-13T14:30:07.382619+00:00 through the same last timestamp.
Ma–Li and QPack have no canonical job timestamps. These are archive timestamps,
not a guarantee of a specific job-start boundary, and the grouped split is
not a future-time test.

No source-wide verified error-mitigation or resilience configuration is
available in the canonical extract. Its status is unknown; absence of a field
does not establish that mitigation was disabled. Shots remain the recorded
per-observation value. Labels are not divided by shots or rescaled to a
fabricated common shot count. The neural graph/global schema does not append
shots; polynomial, throughput and cost methods consume them where declared.

## Group identity and fitting scope

The grouping relation is a transitive union of exact archived QASM hashes,
QPack workflow hashes and parameter-invariant circuit-input digests. The latter
hash circuit quantum/classical allocation, instruction names, ordered operands
and conditions while omitting numerical parameters. It groups repeated declared
structure; it does not prove quantum equivalence. Transitive component IDs are
formed over the full canonical corpus and retained when selecting the current
panel, including equality links through excluded bridge rows.

The current panel has 206 groups: Ma–Li 148, Qonductor 52 and QPack 6. The
46 original QPack workflows merge into six groups through structural identity.
They remain 3,945 observations, but they do not provide 3,945 independent
structures or 46 independent current split groups. Tier group counts can
overlap, so the archive-supported 11 groups and candidate 49 groups are not
60 independent Qonductor groups.

The [split freezer](../methods/real_qpu/prepare_panel.py) assigns
whole components to five outer folds using normalized source-count balance,
largest relative groups first, and deterministic SHA-256 seed/group/fold tie
breaks. Four inner folds partition each outer-training set. The source labels
guide balance; runtime labels and prediction errors do not guide selection or
assignment. These are group-held-out tests, not guaranteed unseen-family,
unseen-backend or future-period tests.

Graph and matched MLP fit gradients and transforms on inner folds 1–3, select
the epoch using inner fold 0, and do not refit on that validation set. Their
reader output is the per-observation median of seeds 42, 1234 and 31415.
Polynomial and Ridge select their hyperparameters on grouped inner folds and
refit on eligible outer-train. Schedule/throughput affine calibration uses
successful outer-train inputs only. The execution disclosure records prior
inspection of aggregate target summaries; the benchmark is retrospective and
exploratory even though the reviewed fitting paths exclude outer-test labels
from transforms, training, selection and calibration.

## Zeros, unavailable inputs and failures

Several numerically similar cases have different meanings. They are retained
separately in the ledgers and coverage accounting.

| Case | Meaning and implemented treatment |
| --- | --- |
| Absent named gate | A valid parsed circuit contains zero occurrences of that named operation. Its gate-count feature is zero. This is measured structural absence. |
| Encoder padding | Inactive one-hot positions and unused operand slots are zero; sparse wire sentinel 255 means no encoded operand. One-qubit and named higher-arity operations preserve the upstream encoder's operand/T1/T2 padding behavior. Padding is not a zero calibration measurement. |
| Constant fit feature | Fit-only normalization sets low-variance positions to zero. It does not replace missing raw values. |
| Missing nominal calibration | A required property/hash/unit/T1/T2/duration/error is absent or invalid. The consuming method is unavailable; zero is not a substitute. A virtual gate duration is zero only when the frozen target actually supplies zero. |
| Missing runtime target | There is no observed quantity to score. Neither reconstruction, a predicted value nor a nominal schedule creates a label. All 8,767 canonical targets here are finite and positive, including all current 7,350 labels. |
| Prediction failure or overflow | Inputs/label may exist, but the computation produces no valid prediction. The attempt records failure/overflow and remains in assigned coverage. Accuracy uses the declared successful subset. |

The neural preprocessing implementation is in the retained
[full-feature runner](../methods/real_qpu/mali_model.py).
Starting from 51 raw globals, it retains columns whose gradient-fit population
sum is positive, then computes means and sample standard deviations on that
same population. A standard deviation above `1e-6` enables scaling; other
retained fields become zero. All five current folds retain 40 global positions.
Node statistics are node-weighted sample moments over fit graphs only. All
178 positions remain in the graph, with 167 varying and 11 constant positions
zeroed. These are schema and transform rules, not mean/zero imputation of
missing observations. The MLP uses the same transformed global vector.

Logical graph materialization requires positive valid T1/T2 for required
logical wire indices in a hash-pinned nominal snapshot, converts supplied units
to microseconds, and maps logical wire `i` to nominal qubit `i`. It rejects
unknown gates, unsupported classical operations, nonterminal measurement and
dynamic control instead of flattening them into a static graph. These T1/T2
values are real snapshot properties but are not attested job-day calibration.
The graph/global schema contains neither numerical gate errors/durations nor
explicit source/backend IDs. Snapshot choice can still carry backend context
through the supplied T1/T2 values.

Polynomial uses `[swap, depth, num_qubits, shots, circuit_count]` in the saved
polynomial manifest (not bundled).
Here `swap` counts CX/CZ/ECR, depth follows the upstream operand stack,
`num_qubits` is touched width, and circuit count is one. It applies polynomial
expansion without logging, scaling or constant-column pruning, selects degree
2/3/4 by mean inner raw-seconds R² with lower-degree tie break, and refits on
outer-train. The common-panel runner requires complete eligible features for
all 7,350 rows; it does not complete an aggregate by dropping missing feature
rows. All folds selected degree 2. Finite negative predictions remain in
raw-seconds scoring; the separate log metric floors predictions at zero.

QCRE/Qiskit schedule projection reuses hash-pinned raw outputs on current IDs
and fits new affine or log-affine calibration on successful outer-train rows.
Unavailable raw inputs remain unavailable after calibration. Calibrated
predictions have a zero floor, while raw schedule outputs retain their schedule
clock. Calibration does not fill rows that lack durations or a supported
circuit route. In older full-corpus static routes, Qonductor `row4477` has
dynamic control and explicit unavailability; metadata-only treatments and the
separate retrospective control-flow case have their own contracts. That row
does not become a member of this logical cohort.

Hyb cost construction uses operation-weighted mean error in one-qubit,
two-qubit and readout buckets, minimum positive operand T2, and serial used
gate/readout duration. The [analytical extension protocol](../protocol/analytical_extension.json)
defines the log-domain cost. A nonempty bucket whose mean error equals one
has true zero survival: nominal log cost is unavailable for 148 current
Ma–Li rows. An individual error-one operation does not automatically imply
zero bucket survival if the bucket mean is below one. Probabilities are not
epsilon-clipped to make the cost finite.

Raw exponential Hyb cost records float64 overflow rather than substituting a
capped cost: nominal variants have 198 overflow rows plus 148 unavailable;
composite variants have 346 overflow rows. A finite log cost can still be
used by log-cost Ridge without exponentiating it first. Its features are
`asinh(log_cost)` and `log1p(shots)`; its target is `log1p(service_seconds)`.
Scaling is fitted inside each inner training partition, alpha is selected by
mean source-balanced inner MAE, and final scaling/Ridge use eligible
outer-train. Inverse `expm1` predictions are floored at zero; nonfinite inverse
outputs and overflow receive explicit statuses. The gate-time control requires
valid positive gate time and shots, and does not inherit noise/T2/zero-survival
exclusions. The Kyoto composite is a separately declared nominal sensitivity
using CSV fields plus pinned older asset fields; it is not selected by test
error. The saved Hyb manifest (not bundled)
records 7,202 nominal log-cost predictions and 7,350 predictions for both
composite Ridge variants.

Scholten-style throughput uses recorded shots, declared compiled quantum-wire
depth and sourced nominal throughput. It does not infer CLOPS from T1/T2 or
convert between throughput definitions. The 2,151 current rows with no sourced
nominal throughput remain unavailable. Affine calibration therefore predicts
5,199 rows, not the complete cohort. A successful-subset error cannot be
ranked against full-panel errors without pairing identical successful IDs.

The saved [unavailability table](../results/real_qpu/summary/unavailable_reasons.csv),
[coverage table](../results/real_qpu/summary/coverage.csv) and
per-method attempts distinguish input unavailability, invalid probability,
zero survival, overflow and prediction failure. The reporting code checks the
assigned envelope and seed completeness; a missing seed blocks the final
neural median instead of silently taking the median of fewer seeds. Large
errors and poor R² remain outcomes, not reasons to remove rows.

## Reproduction and provenance boundary

The earlier local evidence package replayed saved labels, IDs, splits, derived
inputs, attempts and predictions. This public checkout retains the processing
description, aggregate QPU results and local simulator records, but excludes
the per-observation QPU evidence. Fresh source extraction additionally requires
complete upstream archives, pinned parser/generator code and calibration
assets. The [upstream lock](../provenance/upstream.json), input manifests and
[source registry](../protocol/data_sources.json)
identify these dependencies. Preprocessing and fitting source helpers are
included under semantic method paths for inspection; complete upstream archives
and some full materialized compiled inputs remain external. Including source
does not establish fresh-clone extraction or refitting.
See [reproduction](reproduction.md) for the public checkout's verification scope.

For the preprocessing audit, the inspected development helpers
`prepare_logical_benchmark.py`, `mali_full_features.py`,
`run_common_panel_qonductor.py`, `run_common_panel_hyb.py` and
`run_common_panel_analytical.py` matched their SHA-256 entries in the retained
feature and method manifests. Their preserved source bytes, the canonical
materializer, candidate-width/recovery code, native feature adapters and
analytical component builders are included under semantic method paths.
Historical import names and paths remain provenance inside those source bytes;
the publication mapping identifies their public locations. They expose the
implemented mechanics and prerequisite checks, rather than promise executable
extraction without the external inputs. Construction source behavior is supported
by the preserved row fields, decisions and manifests; extraction was not
independently re-executed for this publication.

Simulator labels use a separate circuit-hash/configuration identity and their
own measurement boundaries. Unmeasured/unsupported configurations do not
acquire labels from a runtime prediction. Finite MPS timing and quality pass
are separate fields; a completed approximation is not evidence of fidelity.
Those rules are described in [methodology](methodology.md) and the simulator
results, rather than pooled with QPU service-time observations.
