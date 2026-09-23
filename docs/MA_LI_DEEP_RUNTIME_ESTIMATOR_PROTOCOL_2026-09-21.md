# Ma--Li deep runtime-estimator protocol

Date: 2026-09-21  
Status: active experimental protocol  
Target: Ma--Li recorded IBM Osaka/Kyoto `result.time_taken`, averaged upstream
1,024-shot runs; queue time excluded

## Why this protocol exists

The original local reproduction showed that a simulator-pretrained Ma--Li graph
model catastrophically overpredicts one Osaka QWalk row, while a matched
from-scratch model does not. A saved-model intervention audit further showed
that the tail failure is calibration-conditioned and that graph connectivity is
used, but it did not prove that attention, DAG direction, or source-level graph
topology improves a clean unseen-circuit estimator.

The purpose of this protocol is to answer the narrower and scientifically
useful question:

> Under a frozen current-FakeBackend reconstruction, does physical compiled
> structure or a DAG model improve prediction of source-specific IBM observed
> runtime beyond a gate-aware static critical path and compiled tabular model?

It is not a universal-QPU or vendor-generalization claim.

## Evidence motivating the design

### Existing Ma--Li representation limits

The public model uses a logical pre-transpile DAG, three sparse
`TransformerConv` layers, global mean pooling, 40/41 global features and a
large MLP head. It has about 842k parameters for 340 real-QPU rows.

- Three one-hop layers cannot explicitly propagate critical-path state across
  QWalk's logical depth of 6,068.
- Mean pooling is not a weighted makespan or scheduling operation.
- The source graph has no physical layout, routing history, native duration,
  schedule, or retained measurement nodes.
- Public code does not populate single-qubit op nodes with qubit/T1/T2 fields
  as its paper description suggests.
- The source graph associates snapshot values with logical QASM qubits; it does
  not preserve an execution-time physical layout.

The intervention audit is preserved in
[`graph_intervention_audit_20260921`](../artifacts/mali/graph_intervention_audit_20260921/REPORT.md).
It showed that neutralizing standardized T1/T2 removes the catastrophic
pretrained QWalk tail but does not generally lower MAE, while deleting all DAG
edges destroys both saved models. Thus T1/T2 is a focused transfer-risk
question, and graph topology must be tested by retraining controls rather than
test-time perturbation alone.

### Data and split limits

The 340 labels are not 340 independent circuits:

| Item | Value |
|---|---:|
| Raw label rows | 340 |
| Exact logical-QASM hashes | 170 |
| Hashes observed on both backends | 130 |
| Distinct `(QASM hash, backend)` cells | 300 |
| Duplicate cells under filename aliases | 40 |
| Real QASM hashes also present in simulator pretraining | 170 / 170 |

The old row-level adaptation split places the same QASM hash in training for
260 of 340 held-out row predictions, with a further 24 in validation. It is a
valid narrow label-adaptation setting but cannot support unseen-circuit
language. Simulator pretraining also sees every real-QPU QASM, so it is
circuit-transductive unless outer test hashes are removed from pretraining.

Filename-based family splits are also unsafe: 22 exact hashes occur under both
`realamprandom` and `twolocalrandom` names. Family evaluation must use connected
components of the family--QASM graph.

## Target contracts and non-negotiable boundaries

| Track | Target | May it be pooled? |
|---|---|---|
| Ma--Li observed IBM | Recorded `result.time_taken`, 1,024 shots, source-specific IBM domain | Only Osaka/Kyoto under explicit source/backend metadata |
| QCRE | Current-snapshot, gate-duration critical-path proxy | No: input/prior only, not observed target |
| Simulator pretraining | FakeWashington/FakeSherbrooke simulator runtime | No: auxiliary pretraining target only |

All physical features reconstructed with current FakeOsaka/FakeKyoto must be
labelled `CURRENT_FAKE_SNAPSHOT_PROXY`; the public package does not identify
the historical calibration and physical circuit used at Ma--Li collection time.

## Leakage-safe evaluation protocol

### Primary: grouped logical-QASM test

Define `group_id = SHA256(raw QASM bytes)`. Use five outer folds over the 170
groups. Every alias and every backend copy of a group must remain entirely in
outer train or outer test.

Within each outer train partition, create an inner validation partition that is
also group-disjoint. Fit masks, scalers, calibration coefficients and early
stopping only on outer-train/inner-validation data. Instantiate a fresh model,
optimizer and scheduler for every fold and seed.

### Strict backend plus unseen-QASM diagnostic

For each backend `b` and QASM fold `k`:

```text
test  = backend == b AND QASM group in k
train = backend != b AND QASM group not in k
```

This yields ten correlated diagnostic partitions (two backend directions times
five QASM folds). It is not broad multi-backend CV and must be reported by
held-out backend.

### Family connected-component test

Create a bipartite family--QASM graph and hold out connected components, not
filename tokens independently. This yields nine components, including the
combined `realamprandom + twolocalrandom` component. QWalk has only two rows;
report its absolute error and interval rather than a standalone R-squared.

### Transfer modes

Every model is evaluated in three separately named modes:

1. `scratch_real_qpu`: train only on outer-train real-QPU labels.
2. `transductive_simulator_transfer`: simulator pretraining may include exact
   outer-test QASM; this is Ma--Li-style label adaptation, not new-circuit
   transfer.
3. `strict_all_domain_transfer`: remove outer-test QASM hashes from simulator
   pretraining, real-QPU fitting and inner validation.

## Model ladder

The next model is selected only after each preceding question is answered.

| Stage | Candidate | Question answered |
|---|---|---|
| A0 | Current compiled Ridge/HGB/QCRE residual | What must any graph model beat? |
| A1 | Global-feature MLP and node-only DeepSets | Are edges useful beyond feature statistics? |
| A2 | Directed GraphSAGE, GINE, narrow TransformerConv/GATv2 | Does local topology or attention add value? |
| A3 | Topological critical-path DAG-RNN | Does serial dependency beyond three hops explain QWalk/tail cases? |
| B1 | Compact physical timing DAG plus static residual | Does the executed compiled graph help beyond QCRE? |
| B2 | Dual logical/physical encoder | Does logical structure add any remaining signal? |
| C | Transfer variants only for the two best models | Does pretraining improve data efficiency without tail risk? |

The principal new estimator is physics-guided:

```text
physical timing DAG -> QCRE weighted critical path T_static
                      +
          directed DAG residual encoder
                      -> log(T_observed) = log(T_static + eps) + residual
```

It reduces to a comprehensible static estimator when the learned residual is
zero, and it has an inductive bias compatible with a long serial dependency.

## Physical-DAG construction decision

The full raw physical graph corpus is too large for a Ma--Li-style dense PyG
pickle: 300 unique `(QASM, backend)` graphs have about 67.4m native operations
and roughly 80m dependency edges. A 178-float node representation would exceed
about 50 GB before training.

Build a compact, lazy-loaded corpus instead:

```text
work/mali_physical_dag_v1/           # ignored generated cache
  graphs/<graph_id>.npz
  backend/<snapshot_hash>.json

artifacts/validation/mali_physical_dag_v1/
  manifest.csv
  build_summary.json
  REPORT.md
```

Store each physical graph once and keep 340 labels in a separate manifest.
Arrays should use compact opcode/qubit/duration/topology encodings, with wire
identity as an edge attribute. Preserve measurements. Collapse or coarsen
zero-duration RZ chains for timing graphs, retaining their count separately.
Do not infer absence of routing from `physical_swap_count == 0`: native
decomposition removes explicit `swap` instructions.

## First execution checkpoint (2026-09-21)

The compact build completed under the frozen protocol (`Qiskit 1.4.1`,
`qiskit-ibm-runtime 0.36.1`, `optimization_level=1`,
`seed_transpiler=1234`):

| Quantity | Result |
|---|---:|
| observed label rows represented | 340 |
| unique `(QASM hash, backend)` graphs | 300 |
| native operation nodes | 67,404,145 |
| dependency edges | 79,673,830 |
| maximum active physical width | 127 |
| duration-missing nodes | 0 |

The cache is about 499 MB compressed and is ignored by Git under `work/`; the
versioned artifact keeps the label manifest, graph records, provenance and
baseline metrics. The largest graph has 337,142 native operations and 398,757
dependency edges. The QWalk graph has 107,143 native operations but only nine
active physical qubits, so it is a concrete width-versus-compiled-work
counterexample.

### Static baseline result

The first screening model uses no graph message passing. It fits `log1p` runtime
on the outer-train rows with train-only standardization and compares:

1. `qcre`: weighted physical critical path;
2. `compiled`: QCRE path + compiled depth + active physical width;
3. `physical_graph`: compiled features plus native node/edge counts.

Pooled out-of-fold metrics from the generated artifact are:

| Split | Model | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|
| grouped logical-QASM | QCRE | 0.8274 | 0.7321 | 0.6707 |
| grouped logical-QASM | compiled | 0.6889 | 0.8202 | 0.7660 |
| grouped logical-QASM | physical summary | 0.5766 | 0.8527 | 0.7925 |
| grouped logical-QASM | rich static summary | 0.4937 | 0.8896 | 0.8433 |
| strict backend + unseen QASM | QCRE | 0.8229 | 0.7314 | 0.6744 |
| strict backend + unseen QASM | compiled | 0.6924 | 0.8156 | 0.7625 |
| strict backend + unseen QASM | physical summary | 0.5817 | 0.8500 | 0.7916 |
| strict backend + unseen QASM | rich static summary | 0.5230 | 0.8755 | 0.8306 |
| family-component leave-one-out | QCRE | 0.9157 | 0.6207 | 0.5965 |
| family-component leave-one-out | compiled | 0.7228 | 0.8021 | 0.7490 |
| family-component leave-one-out | physical summary | 0.6136 | 0.8351 | 0.7736 |
| family-component leave-one-out | rich static summary | 0.5752 | 0.8700 | 0.8117 |

The richer static summary improves the static baselines on this source-specific
proxy, but it is not evidence that DAG topology helps: it uses aggregate
timing/layer/opcode descriptors, not message passing. It lowers the QWalk
absolute errors to about 5.91 s (Osaka) and 5.35 s (Kyoto), but the tail still
requires explicit treatment. The current result justifies a targeted residual
experiment, not a universal GNN claim. The full rows and split-level metrics
are in `artifacts/validation/mali_physical_dag_v1/baseline_evaluation/`.

### Coarsened ordered-DAG residual pilot

As a targeted test of long-range order, a small GRU was trained on 256 ordered
topological bins per graph. It predicted the residual
`log1p(T_observed) - log1p(T_QCRE)` with a group-disjoint inner validation split
inside each outer fold. Pooled results for the one-seed pilot were:

| Split | MAE (s) | log-R² | seconds-R² |
|---|---:|---:|---:|
| grouped logical-QASM | 0.8571 | 0.7339 | 0.6675 |
| strict backend + unseen QASM | 0.9022 | 0.6988 | 0.6291 |

This is below the richer static-summary baseline (MAE 0.4937/log-R²
0.8896 on grouped QASM and 0.5230/0.8755 on strict splits), so the pilot does
not justify a DAG-RNN claim. It does slightly reduce the two QWalk absolute
errors to about 5.45 s (Osaka) and 5.24 s (Kyoto), versus 6.95/6.51 s for the
static physical summary, but that local gain is outweighed by aggregate error.
The likely explanation is information loss from 300k native operations to 256
bins combined with only 170 independent logical circuits; the next graph model
should therefore test a small residual on richer static features before
increasing attention/GNN complexity.

### Feature-block ablation and QWalk support audit

The same outer folds were re-used to add one static feature block at a time.
Grouped-QASM pooled metrics were:

| Block | MAE (s) | log-R² | seconds-R² |
|---|---:|---:|---:|
| QCRE | 0.8274 | 0.7321 | 0.6707 |
| Compiled | 0.6889 | 0.8202 | 0.7660 |
| Graph size | 0.5766 | 0.8527 | 0.7925 |
| Layer | 0.5635 | 0.8616 | 0.8022 |
| Timing/criticality | 0.5109 | 0.8892 | 0.8446 |
| Opcode | 0.5563 | 0.8241 | 0.7656 |
| Rich all-block summary | **0.4937** | **0.8895** | **0.8432** |

This is an attribution result for static descriptors. It does not establish a
causal topology benefit. The edge block adds no signal in this corpus because
the aggregate quantum-edge fraction is constant. QWalk is audited separately:
the two rows share one QASM hash, are in grouped fold 5, and have nearest-train
standardized RMS distances around 3.0--3.1. The support table and OOF predictions
are in `artifacts/validation/mali_physical_dag_v1/qwalk_forensics/`.

### Critical-subgraph and uncertainty pass

High-criticality tail summaries were computed from each physical DAG and added
to the rich static model. Grouped log-R² rose from `0.8895` to `0.9041` and
strict log-R² from `0.8755` to `0.8869`, but grouped/strict MAE changed only
from `0.4937/0.5230 s` to `0.4987/0.5306 s`; family-held-out MAE also worsened.
This supports a targeted residual or uncertainty feature, not a claim that a
larger critical-subgraph GNN is already warranted.

The exact-QASM/backend duplicate audit finds 40 cells and 80 rows. The median
absolute label spread is `0.4171 s`, P90 `1.2775 s`, maximum `1.8158 s`, and the
pair-mean RMSE floor is `0.3541 s`. Split-conformal log-runtime intervals use a
QASM-disjoint calibration group. Empirical 90% coverage is `87.6%` grouped and
`86.5%` strict; empirical 80% coverage is `78.8%` and `76.8%`. These intervals
are dataset/proxy diagnostics, not hardware-wide confidence guarantees.

### Retrained topology controls

Saved-model edge deletion/reversal was not sufficient to establish a causal
topology benefit, so a fresh 256-bin message-passing model was trained with
identical node descriptors, residual target, folds, capacity and optimizer for
node-only, true, reversed and shuffled adjacency. Across seeds 1234, 2025 and
31415 on grouped-QASM folds, seed-mean results were:

| Variant | MAE (s) | log-R² |
|---|---:|---:|
| Node-only | **0.8145** | **0.7666** |
| True DAG | 0.8448 | 0.7382 |
| Reversed DAG | 0.8967 | 0.6991 |
| Shuffled DAG | 0.8530 | 0.7285 |

Paired bootstrap over the 170 raw-QASM groups gives true-minus-node log-R²
`-0.0284`, 95% interval `[-0.0650, 0.0008]`, and only `2.9%` probability of a
true-DAG improvement. Thus edge orientation is not useless—true edges beat
reversed/shuffled—but coarse message passing loses to node/timing summaries.
This is grouped-QASM CPU screening, not a strict backend result and not a
universal rejection of all possible DAG architectures. The next graph model
must change representation, for example to a physically meaningful critical
subgraph, rather than merely enlarge attention on these bins.

### Strict transfer, duplicate labels and support-aware intervals

The original adaptation artifact does not establish strict circuit-inductive
transfer because the simulator corpus can contain exact logical circuits later
seen in real-QPU adaptation. The strict screen removes all 170 raw-QASM hashes
from simulator pretraining: 384 of 3,020 simulator rows are excluded, leaving
2,636 rows. A common logical-QASM MLP was used because its inputs have exactly
the same meaning in both domains; it is a controlled transfer screen, not a
substitute for the original Graph Transformer.

| Three-seed mean mode | MAE (s) | log-R² |
|---|---:|---:|
| Scratch | **0.6831** | **0.3360** |
| Frozen transfer | 1.9152 | -0.2533 |
| Fine-tuned transfer | 0.6956 | 0.0816 |

The paired QASM-group bootstrap for fine-tuned minus scratch log-R² is
`[-0.8351, 0.0000]`, with only `2.6%` probability of improvement. This rejects
a transfer benefit for this strict common representation; it does not make a
claim about the unavailable original GAT/Transformer under a strict historical
snapshot protocol. Raw OOF rows and the evaluator are under
`artifacts/validation/mali_strict_transfer_logical_v1/`.

To keep repeats from silently changing optimization weight, exact
`(QASM hash, backend)` duplicate cells were collapsed only within an outer
training partition. All test labels remain uncollapsed. The improvement is
small: grouped equal-cell means yield MAE/log-R² `0.4923 s`/`0.8917` versus
row-weighted `0.4937 s`/`0.8895`; strict noise-weighted means yield
`0.5156 s`/`0.8799` versus `0.5230 s`/`0.8755`. Keep the procedure for
robustness and noise accounting, not as a post-hoc winner-selection rule.

Nearest standardized rich-feature distance to fit rows correctly puts QWalk in
the high-support-distance band. Yet neither global nor two-band conformal
intervals cover QWalk: their nominal-90% grouped coverage is `87.6%`, and the
stratified QWalk interval expands to median `2.7141 s` without coverage. Support
distance should consequently trigger abstention or targeted calibration, not be
advertised as an OOD uncertainty solution.

## Feature ablations

All are retrained under identical folds:

```text
logical DAG
physical DAG structural only
physical DAG + durations
physical DAG + duration/topological critical-path features
physical DAG + routing/layout features
physical DAG + snapshot T1/T2 (separate ablation)
```

For topology, train both true-DAG and retrained shuffled-edge controls. For
attention, use comparable hidden widths and parameter counts across Directed
GraphSAGE, GINE, GATv2 and TransformerConv. Compare mean pooling with
sink-max/logsumexp or attention pooling only after true graph topology has
shown value.

## Reporting gate

A graph model is useful only if it beats both:

1. the static compiled/QCRE residual baseline; and
2. the retrained shuffled-edge/node-only control

on grouped-QASM, strict backend-plus-unseen-QASM, and family-component tests,
with stable five-seed results and no worse tail/QWalk behavior.

Report pooled OOF MAE, MedAE, RMSE, log-R2, seconds-R2, Spearman, P90/P95
absolute error, per-backend error, family-component error, QWalk rows, model
parameter count, inference latency and peak memory.

If a graph model does not pass this gate, the valid finding is still useful:

> Current compiled physical structure and gate-aware critical path contain the
> dominant transferable runtime signal; graph topology has not yet justified
> its statistical and operational cost on this 340-label real-QPU domain.

## References

- Ma and Li, [Understanding and Estimating the Execution Time of Quantum
  Circuits](https://arxiv.org/abs/2411.15631)
- Azizov, Vela-Tambo and Guo, [Transpilation-Aware Runtime Prediction for
  Noisy Quantum Circuit Simulation](https://arxiv.org/abs/2609.12980)
- Tremba, Hovland and Liu, [Is Circuit Depth Accurate for Comparing Quantum
  Circuit Runtimes?](https://arxiv.org/abs/2505.16908)
- Scholten et al., [A Model for Circuit Execution Runtime](https://arxiv.org/abs/2307.04980)
