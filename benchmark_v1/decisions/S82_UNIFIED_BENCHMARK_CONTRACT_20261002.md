# S82 — Unified benchmark contract and completion gates

**Decision date:** 2026-10-02  
**Status:** signed for reporting and for routine materialization; it does not
authorize new timing, model training, live-QPU submission, or a change to
`CURRENT.json`.

## 1. Objective and unit of comparison

The benchmark has two distinct domains.

1. **Archived real QPU.** Predict the provider-recorded, one-circuit
   execution/service observation on the canonical ledger of 8,767 rows.  The
   evaluation target clock is
   `archived_observed_service_execution_time`, in seconds.
2. **Local simulator.** Predict a declared local runtime cell, where a cell
   fixes QASM identity, engine, version, hardware, configuration, output API,
   timing boundary, and any quality policy.  A shared QASM panel does not make
   Aer, CUDA-Q sampling, MPS state production, and tensor contraction the same
   runtime target.

QPack is a source of archived observations, not a competing estimator.
Maestro, Ma–Li, Azizov, and the other named works are method families.  The
benchmark must not substitute a source-specific case study for its method
comparison.

## 2. Canonical archived-QPU ledger

The ledger preserves source identity; it is not 8,767 independently recovered
logical circuits.  Each row has a stable canonical ID, source-row payload/hash,
shots, backend, target value in seconds, grouping key, and terminal method
status.  Reconstruction may supply an input to a method; it never replaces the
archived QPU label.

| Source | Retained rows | Observation and label | Circuit evidence used | Reconstruction status and exclusions |
| --- | ---: | --- | --- | --- |
| Ma–Li real QPU | 340 | Mean of three archived IBM `Result.time_taken` values, 1,024 shots | Exact logical QASM | Logical input is exact. Recompiling to a current nominal Target is analytical replay, not proof of historical routing/calibration. |
| Qonductor one-circuit IBM | 4,482 | IBM one-circuit job time, with recorded shots | Exact submitted physical QASM, DB/ZIP/digest association | All retained rows have `circuit_count = db_circuit_count = 1`. Submitted physical QASM does not recover the original logical circuit. Batch rows and count disagreements remain out of this ledger. |
| QPack MCP | 3,945 | One direct-QAOA-circuit optimizer evaluation; milliseconds divided by 1,000 | Six pinned angle-insensitive logical structures, keyed by recorded size and depth parameter | Optimized angles, submitted routing, and submitted QASM are absent. The 46 workflows remain grouping units. IC/RH multi-circuit VQE/workflow labels are excluded. |

The current extract has no source-wide, provenance-complete error-mitigation or
resilience field.  Reports must say **not observed/verified in the canonical
contract**, rather than claim mitigation is absent or infer it from a backend.

The historically executed unified learned inputs have mixed lifecycle stages:
Ma–Li logical QASM, QPack reconstruction-qualified logical structure, and
Qonductor submitted physical QASM.  These runs remain valid **mixed-stage
adaptations**.  They do not establish a uniformly target-transpiled
representation benchmark.  This decision does not authorize a rerun.  A
future representation study needs a new protocol that proves target/compiler
compatibility for every source before it is called a common compiled panel.

## 3. Fair-comparison rules

1. Every method receives the canonical observation ledger and frozen outer
   partition when its input contract permits it. A row produces a prediction
   or an explicit terminal status.
2. Trainable methods use outer-train data only for feature fitting, degree
   selection, normalization, calibration, early stopping, and model choice.
   The graph seed reduction is the predeclared rowwise median, never the best
   test seed.
3. Coverage uses every assigned row. Error-to-error comparisons use exactly
   the same successful held-out IDs and state that row set beside the number.
4. Source slices are breakdowns of one unified OOF run, not separate training
   runs. They are required because source collection protocol and circuit
   lifecycle differ. A pooled mixture score is descriptive, not evidence of
   transfer to a new source.
5. A raw scheduled, throughput, or effective-cost output remains on its own
   method-output clock. It may be compared to the archived service label as a
   cross-clock diagnostic, but cannot enter the same-clock primary ranking.
   Any outer-train calibration is a separately named adaptation.
6. QCRE and Qiskit values that are numerically equivalent under one frozen
   Target are cross-checks, not independent winners. `CLOPS_h` is never
   converted to `CLOPS_v`; T1/T2 or gate durations are not substitutes.

## 4. Method eligibility and current status

### Archived real QPU

| Method family | Unified route | Output clock | Reader status |
| --- | --- | --- | --- |
| Ma–Li | V3/V3-large hybrid DAG + global-feature adaptation | archived observed service seconds | evaluated adaptation; source-native 340 is separate |
| Qonductor | Five-feature grouped polynomial adaptation | archived observed service seconds | evaluated adaptation; source-local 4,482 is separate |
| Scholten | Nominal single-circuit CLOPS_v-like adaptation | nominal throughput seconds; optional calibrated service seconds | partial coverage, analytical diagnostic |
| QCRE | Snapshot critical path and shot-scaled schedule | scheduled seconds; optional calibrated service seconds | cross-clock analytical diagnostic |
| Qiskit `estimate_duration` | Frozen-snapshot native API | scheduled seconds; optional calibrated service seconds | native schedule diagnostic |
| Hyb-HANAS | Identifiable one-circuit effective-cost component | effective-cost seconds; optional calibrated service seconds | partial coverage, analytical adaptation |

### Local simulator

| Method family | Compatible cell | Required result before a score can be primary | Current status |
| --- | --- | --- | --- |
| Ma–Li-style graph | Noisy-Aer warm execution, core q2–q9 | Exact-QASM OOF score with frozen context | supplementary local adaptation only |
| Azizov | Same Noisy-Aer cell | Source, hybrid, and transpiled GNN results on identical hash-grouped folds | classical Ridge/GBR adaptations only |
| Family-aware residual | Approximation ladder with runtime plus fidelity/quality labels | Strict-circuit OOF and family holdout; prediction-time family policy | unavailable; labels do not exist |
| Maestro | Candidate-specific warm execution/quality matrix | Accepted stable calibration and held-out candidate prediction score | terminal pilot gate failed; no score |
| cuTensorNet | Same scalar selected plan and warm contraction | Estimate/actual paired ratio with matching plan/workspace/precision | native paired diagnostic only |
| Pasqal emu-MPS | Separate analog pulse/layout panel | Quality-qualified analog fit on its own panel | analog pilot, not digital-QASM comparison |

## 5. Required reader-facing tables

The table schemas are pinned in
`benchmark_v1/protocol/benchmark_result_table_schema_v1.json`.

The report must contain, in this order:

1. an archived-QPU **dataset construction table**;
2. an archived-QPU **method results table**, separated into same-clock learned
   results and cross-clock analytical diagnostics;
3. a local-simulator **panel/configuration table**;
4. a local-simulator **method results table**, where unsupported or unfinished
   methods say `unavailable` or `not evaluated`, not zero or blank success;
5. a method-card appendix with fidelity class, representation, clock, code and
   data pins, and claim boundary.

No final prose may claim that the six simulator method families have been
fully benchmarked.  The report may claim that the frozen data construction,
evaluated real-QPU adaptations, local measurement evidence, and their limits
are reproducibly documented.

## 6. Completion gates for future experiment work

`benchmark_v1/protocol/simulator_completion_gates_v1.json` is controlling for
unstarted simulator method work.  Routine workers may materialize manifests,
validate source/hash joins, and prepare commands. They must not start a GPU
job until the relevant strong-method gate is signed.

The existing Maestro pilot is terminal at `pilot_gate_failed`. It cannot be
resumed or enlarged. A new intervention design and a new decision are required
before a new Maestro timing run.

