# Study R coarsened-DAG topology mechanism audit

## Exact representation inventory

The model does **not** consume an unmodified native DAG. The builder first transpiles logical QASM against current FakeOsaka/FakeKyoto snapshots, then constructs an operation DAG ([builder](../../experiments/build_mali_physical_dag.py), lines 143–270). The topology control aggregates every graph’s ordered topological layers into exactly 256 bins ([trainer](../../experiments/train_mali_topology_controls.py), lines 44–104). Across 300 backend-specific reconstructed graphs, 67,404,145 native operation nodes are compressed to 76,800 bins: **877.658 native nodes per bin** on average. Native edge count is 79,673,830. The aggregated coarse edges are generated at training time and not persisted, so an exact saved coarse-edge compression ratio cannot be independently recalculated without rereading the large NPZ corpus; this audit does not do so.

| Feature / information | Category | Used by node-only? | Evidence |
|---|---|---:|---|
| log operation count per ordered bin | Aggregated circuit-level | Yes | Trainer lines 63–70 |
| 14 opcode-count channels per bin | Node-local, aggregated | Yes | Trainer lines 67–70 |
| sum/max current FakeBackend operation duration | Node-local timing, aggregated | Yes | Trainer lines 72–84 |
| max native forward path, reverse path, criticality per bin | **Topology-derived timing summaries** | Yes | Trainer lines 75–84; builder lines 199–226 |
| QCRE critical-path scalar residual anchor | Topology/timing-derived global scalar | Yes | Trainer lines 147–148 and 160–178 |
| adjacency messages | Explicit topology | No for node-only; yes otherwise | Trainer lines 86–104 and 118–135 |
| physical qubit IDs, gate-error values, edge wire/kind | Available in NPZ but excluded from this model | No | Builder lines 228–245; trainer lines 51–59 |
| historical calibration, historical backend runtime context, source ID | Unavailable/excluded | No | Corpus is a current FakeBackend proxy; builder lines 1–12 |

The `node_only` control therefore removes **explicit coarse adjacency**, but it retains ordered bin placement and several descriptors calculated from the complete native dependency DAG. Its readout also contains the final ordered-bin hidden state ([trainer](../../experiments/train_mali_topology_controls.py), lines 132–137). It is not a “bag of independent raw gates” control.

## Coarsening audit

One coarse node represents all native operations whose topological layers fall in a contiguous equal-width layer bin. Multiple operations, dependency edges, and gate occurrences collapse into counts, sum/max summaries, and log-compressed edge multiplicity. This preserves coarse layer order and a directed bin relation for true-DAG edges, but does not preserve individual operations, per-operation edge identity, wire/kind, exact multiplicity, or native critical paths.

The native DAG is acyclic by construction. The coarse graph can contain within-bin self-edges because source and destination are binned independently and no self-edge exclusion is implemented; it is consequently not guaranteed to remain a DAG in the strict graph-theoretic sense. The reversed condition swaps every stored coarse source/destination pair ([trainer](../../experiments/train_mali_topology_controls.py), lines 94–95), so it genuinely changes message direction. The shuffled condition deterministically permutes destinations while retaining source entries, edge count, destination multiset, and unweighted source-counts ([trainer](../../experiments/train_mali_topology_controls.py), lines 96–104). It does **not** preserve source-destination pairing, weighted degrees, acyclicity, or all higher-order graph statistics. Thus it is a legitimate relation-destruction control, not a fully degree-preserving null graph.

## Architecture and fairness audit

All variants instantiate the same `CoarsenedMessageResidual`: 20 inputs, hidden width 32, two mean-normalized directed message-passing updates, mean/max/final-bin pooling, and a residual head ([trainer](../../experiments/train_mali_topology_controls.py), lines 107–137). They use the same AdamW hyperparameters, training epoch cap (15), patience (4), MSE residual loss in log1p target space, group-disjoint inner validation, outer grouped-QASM folds, seeds, and per-seed initialization/order rule ([trainer](../../experiments/train_mali_topology_controls.py), lines 140–178 and 187–232; saved summaries in `topology_controls_grouped_cpu_screen/summary.json` and `topology_controls_grouped_cpu_seeds_2025_31415/summary.json`). Parameter count is exactly shared because every variant uses the same class; node-only still executes the two update blocks with zero aggregated messages.

No implementation defect was found in the saved code path: reversed edges are explicitly swapped, shuffled destinations are deterministically generated per graph, grouped test hashes do not enter their fold’s training partition, and all saved variants align exactly at OOF scoring. However, per-epoch learning curves and checkpoint histories were not saved, so optimization instability, insufficient capacity, and early-stopping sensitivity cannot be ruled out from artifacts alone. The study used three, not five, seeds and a deliberately small two-layer architecture; it cannot identify whether a different native-DAG representation/operator would benefit.

## Hypothesis verdicts

| Hypothesis | Verdict | Why |
|---|---|---|
| A. Node features already contain most runtime-relevant information. | **PARTIALLY SUPPORTED** | Node-only has the best point MAE/raw-R² and retains rich per-bin operation/timing/criticality summaries. But it cannot establish that these features contain “most” information, because explicit adjacency is only one of many omitted alternatives and the summaries themselves derive from the native DAG. |
| B. 256-bin coarsening loses critical dependency information. | **PLAUSIBLE BUT UNVERIFIED** | Compression is severe (877.658 native nodes/bin), individual dependencies/multiplicity are lost, and coarse self-edges may occur. No native-vs-coarse matched model or saved exact preservation statistic links this loss causally to accuracy. |
| C. The current message-passing architecture fails to exploit useful dependency structure. | **PLAUSIBLE BUT UNVERIFIED** | True direction beats reversal, so the operator can react to orientation, but it does not beat node-only/shuffle robustly. The bounded architecture, 256-bin aggregation, and absent training histories preclude concluding either an implementation defect or fundamental topology irrelevance. |

## Supported and prohibited claims

**Supported:** In this historical proxy cohort, retaining true coarse edge direction is better than reversing it, while the tested true-DAG pipeline fails to show robust incremental predictive value beyond the node-only and shuffled controls.

**Not supported:** “Topology causally improves runtime prediction”; “the topology controls prove topology is irrelevant”; “the true DAG is necessary”; “the finding applies to the 4,515-row Track-A benchmark”; or “current FakeBackend duration descriptors reconstruct historical hardware timing.”

## Newer related evidence, kept separate

The later frozen Track-A topology ablation at `artifacts/postclosure_topology_ablation/v1/` is a distinct 4,515-observation study. Its own conclusion is no supported topology benefit under that protocol ([final interpretation](../postclosure_topology_ablation/v1/final_interpretation.md), lines 1–10). It corroborates cautious thesis wording but cannot be pooled with this 340-row raw-R² calculation.

