# Existing logical-topology evidence reconciliation

## Scope

This is a saved-artifact reconciliation, not a new fit.  It answers the
logical-circuit connectivity question already tested on the frozen Track-A
panel.  It does **not** answer the later physical/transpiled-DAG question.

## Verified compatible study

[`artifacts/postclosure_topology_ablation/v1/protocol.json`](../../postclosure_topology_ablation/v1/protocol.json)
records 4,515 observations, 165 structural groups, the five frozen outer
folds, seeds 42/1234/31415, 500 epochs, raw-seconds MSE, and group bootstrap
of the 165 groups.  The saved validation receipt reports 75/75 complete
variant × fold × seed cells and `PASS`:
[`validation_report.json`](../../postclosure_topology_ablation/v1/validation_report.json).

All variants used identical 178-field node tensors and the same transformed
global tensor.  The controls are documented in
[`model_variants.md`](../../postclosure_topology_ablation/v1/model_variants.md):

| Treatment | Connectivity intervention | Architecture/capacity status |
|---|---|---|
| T1 node-only | no cross-node messages | DeepSets blocks; 840,388 parameters, within 0.2% of T2 |
| T2 true DAG | saved directed logical dependency edges | 841,993 parameters |
| T3 no-edge | empty edge tensor | exact T2 implementation and parameter count; message weights inactive |
| T4 shuffled | deterministic directed double-edge swaps | exact T2 architecture; preserves edge count and in/out degree |
| T5 reverse | every saved edge direction reversed | exact T2 architecture |

T4 is not a completely independent graph replacement: median original-edge
retention was 48.5% for Ma–Li, 9.4% for Qonductor, and 3.3% for QPack; it may
also contain cycles by design.  Therefore T2–T4 is a valid partial-rewiring
diagnostic, not a clean proof that every edge matters.

## Saved results relevant to P2 attribution

The ensemble is the row-wise median across all three seed predictions.  The
primary statistic below is group-balanced MAE; negative signed deltas favor
T2.  Values are transcribed from
[`topology_effect_summary.md`](../../postclosure_topology_ablation/v1/topology_effect_summary.md).

| Contrast | Group-balanced delta MAE (s) | Group-bootstrap 95% CI | Interpretation |
|---|---:|---|---|
| T2 true − T1 node-only | -0.0534 | [-0.1900, +0.0282] | Direction favors true edges on this weighting, but interval crosses zero. |
| T2 true − T3 no-edge | +0.0707 | [+0.0266, +0.1201] | Exact-capacity no-edge control is better. |
| T2 true − T4 shuffled | -0.0012 | [-0.0551, +0.0503] | No incremental true-connectivity result. |
| T2 true − T5 reverse | +0.0020 | [-0.0417, +0.0495] | No direction-sensitive result. |

The prespecified practical/robustness gate failed.  The artifact’s primary
conclusion—**no supported useful true-logical-DAG advantage under this
protocol**—is therefore retained.  This is evidence that changing logical
connectivity changed model behavior, but it does not establish an incremental
benefit of the correct dependency relation.

## What this closes and what it does not

This study is sufficient to prevent a duplicate **logical** Track-A topology
training campaign.  It is not the requested physical-DAG P2 because its node
tensor is the Ma–Li logical graph encoding (including nominal T1/T2 and node
position), rather than bins derived from the saved compiled/submitted physical
operation graph.  The P0 physical inventory records graph sizes that make
full-operation parity impractical; a physical P2, if P1 is first supported,
must be a separately locked coarsened experiment with NODE_ONLY / TRUE_DAG /
REVERSED_DAG / SHUFFLED_DAG controls.

## Claim boundaries

Allowed now: “On the frozen qualified logical Track-A protocol, the matched
controls did not demonstrate a robust incremental benefit from the true logical
dependency adjacency.”

Not established: physical-circuit dependency topology; actual QPU routing or
scheduling topology; topology as a physical cause of runtime; universal absence
of graph value.
