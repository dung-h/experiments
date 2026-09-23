# Ma–Li graph-model deep audit

Run date: 2026-09-21  
Evidence class: saved-model test-time intervention plus public-source audit  
Target: recorded IBM Osaka/Kyoto `result.time_taken`, 1,024 shots, queue excluded

## Research question

This audit asks three separate questions that are easy to conflate:

1. Does the circuit dependency graph contain useful runtime signal?
2. Does graph attention add value beyond node/global statistics?
3. Does simulator pretraining improve prediction on real-QPU labels?

The current experiment answers only diagnostic parts of these questions. It
does not yet compare retrained, parameter-matched graph architectures.

## What the public Ma–Li model actually is

The model is a two-branch regressor rather than a generic full-attention
Transformer:

```text
logical QASM DAG
  -> three PyG TransformerConv layers (one-hop sparse attention)
  -> global mean pooling -> 178-dimensional graph embedding

40/41 global circuit features
  -> 64 -> 64 MLP

concatenation -> 512 -> 512 -> 128 -> one raw-seconds prediction
```

The local real-QPU configuration has 841,993 trainable parameters for 340
labels. Each of the three graph layers communicates only across one dependency
edge, so a node receives information from at most a three-hop neighborhood.
This is very different from globally attending over the entire circuit.

The 340 stored graphs contain 2,564,797 nodes in total. Per graph, the node
count ranges from 5,487 to 9,974, with a median of 7,452. Node count alone has
only moderate association with observed runtime (Spearman 0.367; Pearson
0.260). QWalk is a depth/serial-dependency outlier rather than the largest
graph: it has 6,207 nodes and a logical depth of 6,068.

## Public-source implementation findings

These findings apply to upstream commit
`32c392a6ece276f1ff046d4e30052d0571ff6dc6`, not necessarily to unpublished
author-side code or the missing processed tensor.

1. **The graph is pre-transpilation.** The features describe the logical QASM,
   while the backend executes a mapped, decomposed and scheduled physical
   circuit. Routing, physical layout, native gates, gate durations and inserted
   operations are absent from the graph.
2. **The receptive field is short.** Three sparse `TransformerConv` layers do
   not explicitly propagate a critical-path state through thousands of serial
   gates. The global depth/count vector must carry much of the long-range size
   signal.
3. **Mean pooling is not a scheduler.** It assigns equal aggregation weight to
   every final node representation and cannot directly compute a weighted
   longest path, slack or parallel makespan.
4. **Single-qubit node metadata disagrees with the paper description.** In
   `circ_dag_converter.py`, qubit-index and T1/T2 fields are populated for
   input/output nodes and two-qubit gates, but not for a single-qubit operation.
   A direct X/CX micro-check confirmed the behavior.
5. **Measurements are removed.** `RemoveFinalMeasurements` runs before DAG
   conversion, although measurement and readout overhead can contribute to the
   observed target.
6. **No physical timing feature is used.** The code builds a partial gate
   dictionary but does not attach native instruction duration, gate error or
   schedule information to graph nodes or edges.
7. **Qubit identity is fragile.** The one-hot position is a first-seen logical
   qubit remapping. T1/T2 values are looked up using the original QASM qubit
   index, not a preserved transpiler physical layout.
8. **The upstream cross-validation script is not fold-isolated.** It creates
   one optimizer and scheduler outside the fold loop, reloads model weights but
   retains optimizer/scheduler state, and evaluates every epoch on the same
   held-out fold whose best MSE is reported. The preprocessed tensor is also
   standardized before splitting. The local 8:1:1 reproduction fixed the
   train/validation/test separation, but still uses a reconstructed
   artifact-wide standardization tensor.

The current paper revision now reports an algorithm-family split (R² 0.878,
down from 0.905 under ordinary cross-validation) and explicitly notes that
family overlap makes random evaluation optimistic. That is an important
improvement in the paper, but the public repository does not provide a clean,
fold-isolated implementation of the full revised protocol.

## Saved-model intervention protocol

The already-trained local checkpoints are evaluated on their exact out-of-fold
rows. No parameters are updated. In the stored standardized tensor, setting a
feature block to zero replaces it with the artifact-wide mean. Edge removal,
reversal and rewiring are deliberately out-of-distribution diagnostics.

Two initializations are compared:

- `pretrained`: FakeWashington/FakeSherbrooke simulator checkpoint followed by
  Osaka/Kyoto adaptation;
- `scratch`: identical architecture, rows, folds and optimizer, but random
  initialization before real-label training.

The table reports pooled out-of-fold metrics over all 340 predictions. These
pooled R² values differ from the earlier mean of ten fold-level R² values; the
prediction rows themselves are unchanged.

| Initialization | Test-time input | MAE (s) | RMSE (s) | R² seconds | log R² |
|---|---|---:|---:|---:|---:|
| Pretrained | Full input | 0.5483 | 1.3674 | 0.4865 | 0.8613 |
| Pretrained | T1/T2 -> mean | 0.5487 | 0.7467 | 0.8469 | 0.8725 |
| Pretrained | Node type -> mean | 0.7570 | 1.2949 | 0.5395 | 0.7809 |
| Pretrained | Qubit index -> mean | 3.6736 | 4.1869 | -3.8140 | -4.5244 |
| Pretrained | Node index -> mean | 0.6151 | 1.4616 | 0.4133 | 0.8393 |
| Pretrained | No edges | 5.2560 | 5.4603 | -7.1876 | -11.6309 |
| Pretrained | Reverse edges | 0.6601 | 1.4578 | 0.4164 | 0.8260 |
| Pretrained | Rewired edges | 1.8010 | 2.0532 | -0.1577 | 0.0362 |
| Scratch | Full input | **0.4783** | **0.6115** | **0.8973** | **0.9135** |
| Scratch | T1/T2 -> mean | 0.4756 | 0.6198 | 0.8945 | 0.9114 |
| Scratch | Node type -> mean | 0.8964 | 1.3997 | 0.4620 | 0.5590 |
| Scratch | Qubit index -> mean | 4.6862 | 5.1220 | -6.2045 | -9.8395 |
| Scratch | Node index -> mean | 0.5043 | 0.6382 | 0.8882 | 0.9059 |
| Scratch | No edges | 5.9652 | 6.1611 | -9.4241 | -17.5907 |
| Scratch | Reverse edges | 0.5984 | 0.7570 | 0.8426 | 0.8771 |
| Scratch | Rewired edges | 0.6258 | 0.7759 | 0.8347 | 0.8623 |

## Interpretation

### Calibration-conditioned negative transfer is a tail failure

For the pretrained model, neutralizing T1/T2 reduces the worst absolute error
from 22.335 s to 5.706 s and raises pooled seconds R² from 0.4865 to 0.8469.
The improvement is dominated by held-out Osaka
`qwalk-noancilla_indep_qiskit_9`:

```text
truth                         13.6962 s
pretrained, full input        36.0312 s
pretrained, T1/T2 -> mean     19.4024 s
scratch, full input           12.8724 s
scratch, T1/T2 -> mean        13.2135 s
```

This is not evidence that removing T1/T2 improves ordinary rows. Only 155 of
340 pretrained rows improve, median absolute error worsens from 0.401 s to
0.444 s, and MAE is effectively unchanged. The intervention removes a
catastrophic calibration-conditioned tail error. The scratch model is nearly
insensitive to the same intervention, supporting the narrower conclusion that
the simulator checkpoint transferred a harmful T1/T2-conditioned prior.

### The model uses graph connectivity, but direction is not the main signal

Removing every edge destroys both models, so the graph branch is not ignored.
Random rewiring is particularly damaging to the pretrained model. However,
reversing every edge reduces scratch R² only from 0.897 to 0.843, and random
rewiring still leaves scratch R² at 0.835. This suggests that local adjacency,
node/global statistics and skip paths carry more signal than a faithful
forward critical-path computation. A retrained edge-shuffle control is needed
before attributing accuracy specifically to DAG direction.

The QWalk prediction becoming accurate when all edges are removed is accidental
and must not be generalized: over all 340 rows, the no-edge model has R² below
-7 for pretrained and below -9 for scratch.

### Qubit index dominates, but its semantics are questionable

Neutralizing the qubit-index block catastrophically degrades both models. That
proves reliance, not validity. The current index is a logical/first-seen label,
single-qubit operations omit the field, and the physical layout used at
execution is unavailable. A strong next model should represent the actual
physical mapping or learn qubit-label invariance, rather than depend on an
arbitrary logical numbering convention.

## Stronger experiment program

The next study should not merely replace `TransformerConv` and compare a
random split. It should separate representation, encoder, transfer and timing
prior under identical leakage-safe folds.

### Stage A: determine whether a DAG is needed

Use the same train-only preprocessing and splits for:

1. compiled-feature Ridge and HistGradientBoosting;
2. global-feature MLP with a parameter budget matched to graph models;
3. original TransformerConv with mean pooling;
4. the same model retrained with shuffled edges;
5. GraphSAGE and GIN encoders;
6. a directed DAG encoder with separate predecessor/successor messages.

Graph topology earns its complexity only if the intact graph beats both the
tabular baseline and the retrained shuffled-edge control on strict splits.
Attention earns a separate claim only if it beats parameter-matched
GraphSAGE/GIN, not merely a global MLP.

### Stage B: fix the representation before enlarging the model

Compare three views:

- logical/source DAG;
- physical post-transpile DAG with native gates and physical qubits;
- dual view: logical DAG plus physical DAG/backend graph.

Attach native instruction duration, measurement duration, gate error, physical
qubit identity, routing operations and stable topological-level/criticality
features. Replace raw node ordinal with normalized topological depth, earliest
start, latest start/slack and forward/reverse critical-path distance. Use
sum/attention/soft-max pooling in addition to mean pooling.

A full dense Graph Transformer is a poor first replacement because a 10,000
node circuit would require quadratic attention. A topological DAG-RNN or
hierarchical layer/block graph can propagate long-range critical-path state in
linear time.

### Stage C: test transfer rather than assuming it helps

For the best two representations, compare:

1. random initialization;
2. full simulator checkpoint fine-tuning;
3. pretrained encoder plus a reinitialized runtime head;
4. frozen encoder followed by gradual unfreezing;
5. structural/ranking pretraining rather than raw simulator-seconds pretraining.

The last option is especially promising. The paper reports high Spearman
association between simulator and QPU runtimes even though their absolute
scales differ. Pairwise/listwise ranking pretraining can preserve ordering
signal without forcing the QPU model to inherit the incorrect 42–73 s simulator
scale observed for QWalk.

### Stage D: combine the static model and graph model

Use QCRE's weighted physical critical path as a physics prior and learn only a
correction:

```text
log(T_observed) = log(T_static + epsilon) + residual_GNN(circuit, backend)
```

The residual model should also output a quantile or uncertainty interval. This
hybrid is more data-efficient and interpretable than asking an 842k-parameter
network to relearn scheduling from 340 labels. The current direct baseline to
beat is compiled Ridge log-R² 0.8742 under backend-plus-unseen-circuit testing;
the residual Random Forest reaches 0.8753. A new GNN is not useful unless it
improves strict generalization or tail risk beyond that level.

## Required evaluation gates

- logical-QASM-grouped split;
- backend plus unseen logical circuit;
- family-held-out split;
- future calibration-snapshot-held-out split when timestamps are available;
- at least five training seeds;
- fresh model, optimizer and scheduler per fold;
- train-fold-only feature selection/scaling;
- pooled OOF MAE, MedAE, RMSE, log-R², seconds R², Spearman, P90/P95 error;
- inference latency and peak memory, because the estimator itself must be
  cheaper than the execution decision it supports.

The 340-row Ma–Li set contains only two IBM backends and one fixed shot count.
It can establish a source-specific estimator and a careful transfer study, but
not universal vendor or shot-count generalization.

## Reproduction

From the replication repository, using the Ma–Li PyTorch/PyG environment:

```bash
/path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/mali_graph_intervention_audit.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction
```

Generated evidence:

- `oof_intervention_predictions.csv`: all 5,440 row-level predictions;
- `oof_intervention_metrics.csv`: pooled metrics for 16 model/intervention pairs;
- `metadata_and_metrics.json`: paths, target semantics, seed and run metadata.

Primary references:

- Ma and Li, *Understanding and Estimating the Execution Time of Quantum
  Circuits*: <https://arxiv.org/abs/2411.15631>
- Azizov, Vela-Tambo and Guo, *Transpilation-Aware Runtime Prediction for Noisy
  Quantum Circuit Simulation*: <https://arxiv.org/abs/2609.12980>
- Tremba, Hovland and Liu, *Is Circuit Depth Accurate for Comparing Quantum
  Circuit Runtimes?*: <https://arxiv.org/abs/2505.16908>
- Scholten et al., *A Model for Circuit Execution Runtime*: 
  <https://arxiv.org/abs/2307.04980>
