# Construction of the real-QPU dataset

The real-QPU benchmark contains **4,515 archived runtime observations**:
340 from Ma–Li, 230 from Qonductor and 3,945 from QPack. This is the only
real-QPU evaluation population in this publication. Every method is assigned
these same observations and the same outer test folds.

An observation is a recorded circuit execution in a particular context, not
necessarily a distinct circuit. The dataset has **165 conservative
circuit/workflow groups**. Runtime labels are taken from the original sources;
reconstructing an input never generates, estimates or replaces a label.

## Source selection

| Source | Source records examined for this scope | Inclusion rule | Retained observations | Groups |
| --- | ---: | --- | ---: | ---: |
| Ma–Li | 340 real-QPU observations | Recorded execution time linked to source logical QASM | 340 | 148 |
| Qonductor | 4,482 IBM one-circuit jobs | Exact submitted QASM association plus an archive-supported, admissible logical recipe | 230 | 11 |
| QPack | 3,945 MCP circuit-execution events in six source records | Single-circuit QAOA structural reconstruction from recorded MCP settings | 3,945 | 6 |
| **Benchmark** | — | **Union of the three retained sets** | **4,515** | **165** |

The QPack number is the MCP extraction, not the size of every dataset in the
QPack repository. Its other experiments are outside the source screen. Counts
in the source-audit tables below describe filtering the original archives, not
additional benchmark populations.

The source pins are in [upstream metadata](../provenance/upstream.json).
The machine-readable selection, evidence digests and split counts are in
[dataset metadata](../protocol/real_qpu_dataset.json). The
[aggregate profile](../results/real_qpu/dataset_profile.csv) contains no row-level
runtime labels.

## Shared inclusion criteria

A record must have an attributable source, a finite positive observed runtime,
known units, backend and shots, an admissible circuit-level boundary, and one
of these three input tiers:

| Tier | What is established | What is not established |
| --- | --- | --- |
| `source_has_logical_input` | Ma–Li source logical QASM linked to the source observation | Exact physical compilation and job-day calibration |
| `archive_supported_recipe_adaptation` | Qonductor submitted physical QASM linked to the job and matching the pinned benchmark archive; qualified family/width recipe reconstruction | Byte-identical original pre-compilation logical QASM or full-unitary equivalence |
| `qpack_reconstruction_qualified` | QPack MCP structure, problem size and QAOA depth supported by source records | Optimized angles and submitted physical circuit for each historical iteration |

These are qualifications for the declared structural benchmark, not three
equivalent kinds of exact historical circuit recovery. Family names, matching
widths or similar runtime alone do not establish a circuit-to-label link.
Inputs with only unverified recipe candidates, structural hints or unresolved
logical parameters are excluded. Prediction errors are not a selection criterion.

Selection is retrospective. Provenance-based filtering reduces uncertainty
but also changes the sampled distribution; it is not analyst-blind
preregistration or evidence that excluded circuits are unimportant.

## Ma–Li: 340 observations

Source: [Quantum-Execution-Time-Prediction](https://github.com/mooselab/Quantum-Execution-Time-Prediction),
revision `32c392a6ece276f1ff046d4e30052d0571ff6dc6`.

1. Read the real-QPU records and their referenced source QASM.
2. Retain the logical circuit reference rather than substituting a newly
   generated same-family circuit.
3. Use the mean of the three archived IBM `Result.time_taken` values as
   `target_seconds`. Requested shots are 1,024.
4. Preserve backend and source identity. There are 192 Osaka and 148 Kyoto
   observations. Job timestamps are not available in this extract.
5. Hash circuit content and model-visible structure before assigning groups.
   Repeated observations are retained, but the same group never crosses a split.

All 340 records meet this source-input screen. The source audit found 192
referenced QASM files with 170 distinct byte contents. These file counts are
not observation counts. The conservative split grouping produces 148 groups;
it also joins parameter variants whose declared structural inputs omit angles.

The ten retained source families are QPE-inexact (36), random (30),
SU2-random (38), QNN (26), QFT-entangled (47), QWalk-noancilla (2),
TwoLocal-random (40), QPE-exact (35), RealAmplitudes-random (40) and QFT (46).

## Qonductor: original IBM jobs to 230 observations

Source: [Qonductor-SC25](https://github.com/manosgior/Qonductor-SC25),
revision `5d1ac8a90cd574a23e7544e1044681641354ff67`.

### Job and physical-circuit association

The source database audit identifies 4,613 one-circuit jobs across providers.
The IBM extraction contains 4,482; the 131 other-provider jobs are outside this
IBM snapshot-based screen.

For each IBM job, follow the database job/circuit foreign-key association,
require both recorded circuit counts to equal one, resolve the member in
`circuits.zip`, and hash the submitted QASM. The direct association audit
passes for all 4,482 IBM records. This establishes the **submitted physical
circuit**, not its original logical circuit.

The benchmark archive `benchmarks.zip` also contains compiled physical QASM.
Its filename and content match can support an originating benchmark recipe.
It must not be described as an archive of recovered logical source QASM.

### Logical-recipe qualification

The logical-recovery ledger partitions the 4,482 IBM records as follows:

| Ledger classification | Observations | Decision |
| --- | ---: | --- |
| Archive-resolved recipe | 230 | Retain as an archive-supported recipe adaptation |
| Candidate, not verified | 3,681 | Exclude: insufficient archive-supported recipe evidence |
| Structural-only graphstate | 14 | Exclude: exact archive match does not establish the random logical instance |
| Unresolved | 557 | Exclude: logical width or recipe remains unresolved |
| **Source total** | **4,482** | **230 retained; 4,252 excluded** |

The 230 retained observations comprise **GHZ 113, W-state 57, DJ 48,
QFT-entangled 7 and QPE-exact 5**. They are not the QWalk family. There are
32 distinct submitted physical QASM hashes and 11 logical structural groups.

The qualification procedure is:

1. Parse the submitted circuit, measurement destinations, classical registers
   and used quantum wires. Separate allocated backend width from logical width.
2. Require a filename/content-hash match to the pinned benchmark archive.
   Use its recipe metadata together with family-specific generator semantics.
3. Check that the logical-width/recipe interpretation is admissible. Backend
   capacity, a summed classical-register width or family name alone is not enough.
4. Generate the declared logical recipe with **MQTBench 1.0.3**, using the
   independent Qiskit level and recorded family/width settings.
5. Validate the declared terminal-measurement mapping and supported static
   semantics. Check ideal measured outputs from the zero initial state for
   the 32 distinct submitted circuits against their accepted recipes.
6. Retain each original job label and its physical QASM association; attach the
   reconstructed logical recipe as a separate, qualified method input.

The 32 ideal-output checks pass. Their scope is zero-initial-state measured
behavior, not all-input unitary equivalence, identity of the historical DAG,
historical routing or noise behavior. The author archive does not supply the
missing original logical bytes. This limitation remains even for retained rows.

All 230 retained records have timestamps, known shots and one-circuit job
identity. Widths are 3–7 logical qubits; shots range from 1,000 to 20,000.
The backend slice is Perth 25, Nairobi 19, Jakarta 41, Lima 41, Quito 42,
Belem 24, Manila 33 and Lagos 5.

The target is the archived provider-observed one-circuit job time in seconds,
not a gate/pulse schedule. There is no explicit per-observation
error-mitigation field in the frozen extract. Missing mitigation metadata
does not mean mitigation was absent.

### Inspectable preprocessing code

The published [recovery rules](../methods/real_qpu/qonductor_recovery_rules.py),
[archive recovery](../methods/real_qpu/build_qonductor_logical_recovery.py) and
[recipe validation](../methods/real_qpu/validate_qonductor_archive_recipes.py)
show the recorded logic. They require external source archives and helpers;
their original workspace paths are not a portable download command.

## QPack: 3,945 MCP evaluations

Source: [LibKet/QPack](https://gitlab.com/libket/qpack),
revision `9beaf65e951e01181b1e324cbb1b227af10eef96`.

1. Select the MCP/MaxCut QAOA source records whose circuit-execution events
   fit this single-circuit boundary. Six records contain 46 optimizer workflows
   and 3,945 per-iteration events.
2. Take `Circuit execution durations [ms]` /
   `circuit_execution_duration_ms` and divide by 1,000. Do not substitute
   queue time, whole-job elapsed time, optimizer duration or whole-workflow time.
3. Preserve the problem size, QAOA depth, backend, 4,096 shots and workflow
   identity. Job IDs and per-event timestamps are not recovered.
4. Reconstruct the circuit structure from the pinned MCP recipe and recorded
   settings. Use representative nonzero angles (`rz(0.3)`, `rx(0.2)`).
   Per-iteration optimized angles and submitted physical QASM are unavailable.
5. Group all shared model-visible structures across workflows and backends.
   The 46 workflows reduce to six conservative structural groups.

All 3,945 selected MCP events are retained. Their problem-size counts are
2: 860, 3: 876, 4: 888, 5: 823, 6: 267 and 7: 231. Other QPack experiments,
including VQE evaluations requiring several measurement circuits, are excluded
from this circuit-level screen; no total for every source task is inferred.

The Ma–Li-style graph/global schema omits numerical angles, so this is a
declared angle-insensitive structural adaptation. It is not recovery of exact
historical circuit instances. Nominal transpilation can still be angle-sensitive;
the representative compiled view is an approximation for physical-feature and
analytical methods. No runtime is simulated to replace these historical labels.

## Joining and normalizing the sources

The canonical row identity combines source ID and source-row identity within
the pinned extract. Preserve the source reference and raw-record digest.
A circuit hash is not a job ID: several observed runtimes can legitimately
share a circuit or workflow structure.

Three circuit identities serve different purposes:

- **Source logical identity:** source QASM, or the separately qualified recipe.
- **Submitted physical identity:** actual archived submitted QASM, available
  for Qonductor. A reconstructed circuit must never receive this status.
- **Nominal compiled identity:** a deterministic target compilation used by
  physical-feature and analytical methods; it is not the historical submission.

Each representation has its own digest and lifecycle label. Digest equality
proves equality of the encoded bytes under that definition, not equality of
two different lifecycle stages. Joining a recipe to an observation does not
create another runtime observation.

All target units become seconds. Shots, backend and source remain separate
context fields. Backend aliases normalize `ibmq_*` and `ibm_*` for snapshot
lookup without erasing the archived name.

Ma–Li and QPack lack job timestamps. Same-backend FakeBackend snapshots are
therefore **nominal context**, not proved job-day calibration. Logical wire
indices used for T1/T2 are not recovered historical physical placements.
Source execution boundaries are similar enough for the declared archived
service/execution-time adaptation, but are not identical pure hardware clocks.
Source-stratified reporting is required to expose that heterogeneity.

## Retained data profile

The following summaries are computed on retained observations, not generated
circuit runtimes. Width is the allocated source/reconstructed register width;
it is not necessarily active algorithm width.

| Source | Allocated width min / median / max | Logical/source depth min / median / max | Shots min / median / max | Observed seconds min / median / max |
| --- | --- | --- | --- | --- |
| Ma–Li | 9 / 79 / 127 | 205 / 294 / 6,068 | 1,024 / 1,024 / 1,024 | 3.107 / 8.637 / 13.696 |
| Qonductor | 3 / 5 / 7 | 5 / 8 / 15 | 1,000 / 1,000 / 20,000 | 1.058 / 3.096 / 10.882 |
| QPack | 2 / 4 / 7 | 14 / 41 / 131 | 4,096 / 4,096 / 4,096 | 3.505 / 6.590 / 9.465 |

The combined dataset contains ten archived backend names. Backend counts,
source-specific counts and unrounded profile values are in the metadata and
aggregate CSV. Large differences in allocated width/depth do not by themselves
establish differences in hardware execution time: compiled views, shots and
provider timing boundaries matter. Retained Qonductor inputs are predominantly
small, archive-resolved recipes; the excluded source records are not represented
by this benchmark. QPack's many repeated evaluations must not be interpreted
as many independent structures.

## Missing information and exclusions

| Item | Handling |
| --- | --- |
| Missing runtime, unknown units, nonfinite/nonpositive label | Reject as an observed target; never synthesize a runtime |
| Unverified logical recipe | Exclude from this dataset |
| QPack historical angles/routing | Retain qualified structural reconstruction; disclose approximation |
| Missing timestamp/job-day calibration | Keep missing; use declared nominal snapshot context |
| Missing mitigation metadata | Keep unknown; do not silently assign “none” |
| Missing gate duration/error/T1/T2 required by a method | Mark that method unavailable with a reason; do not drop the dataset row |
| Missing historical throughput or effective-depth inputs | Separate nominal adaptation from unavailable original-paper route |
| Valid zero operation count | Keep zero; it is not a missing feature |
| Failed computation or invalid prediction | Retain assigned denominator; record failure, not zero runtime |

See the [feature dictionary](data_and_feature_dictionary.md) for every ordered
neural input and the [QPU protocol](real_qpu_benchmark.md) for regression and
analytical inputs.

## Splits and independence

The 4,515 observations receive **fresh five-fold outer assignments**, with
four inner folds within each outer train set. Seed 42 fixes outer allocation;
inner seeds are 43–47. Allocation is deterministic, group-based and
source-balanced, without runtime targets.

Conservative groups join shared circuit/workflow and parameter-invariant
model-input structures. Exclusion does not split an existing group into new
independent samples. No group crosses outer train/test or inner partitions.
All three sources appear in every outer test and inner validation set.

| Outer test fold | Ma–Li | Qonductor | QPack | Total |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 68 | 47 | 823 | 938 |
| 1 | 68 | 46 | 888 | 1,002 |
| 2 | 68 | 39 | 498 | 605 |
| 3 | 68 | 36 | 860 | 964 |
| 4 | 68 | 62 | 876 | 1,006 |
| **Total** | **340** | **230** | **3,945** | **4,515** |

Unequal fold sizes follow whole-group allocation. They are not failed balancing
or permission to split circuits across train and test. These folds test
held-out groups within the combined source distribution; they are not a
backend-held-out, family-held-out or temporal evaluation.

QPack supplies 87.4% of observations but only six groups. Report pooled and
source-balanced errors, source slices, tails and group-resampled paired
uncertainty. Neither 4,515 rows nor 165 groups implies 4,515 independent circuits.

## Current result and reproduction status

Dataset preparation, split checks and final saved-evidence QA pass. Fresh
learned fits and fresh fold-local analytical calibrations use this population;
the [complete QPU table](../results/real_qpu/method_comparison.md) contains
32 variants. No scores from another training population are relabeled as
results on this dataset. [Detailed source profiles](../results/real_qpu/dataset_profile.json)
include width, depth, gate count, shots and archived runtime distributions.

The publication includes processing code, selection rules, source pins,
aggregate counts and evidence digests. Original third-party observations,
row-level derivatives and source circuits are not redistributed. A Git clone
can verify the published inventory and saved simulator evidence; it cannot
recompute omitted QPU labels or complete fresh fits without those external
inputs. See [reproduction scope](reproduction.md).

The simulator dataset remains a separate locally measured track. Its engines,
resource limits and clock/quality definitions are in
[training and measurement](training_and_measurement.md#digital-simulator-measurement).
