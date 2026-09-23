# Ma--Li coarsened topology controls

All variants use identical binned node features, splits, residual target and optimizer. Only adjacency differs. This is a retrained topology diagnostic, not a full native-DAG GNN result.

| Evaluation | Variant | Seed | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|---:|
| grouped-QASM | `node_only` | 1234 | 0.9040 | 0.7243 | 0.6373 |
| grouped-QASM | `true_dag` | 1234 | 0.9180 | 0.7098 | 0.6118 |
| grouped-QASM | `reversed_dag` | 1234 | 0.9698 | 0.6436 | 0.5548 |
| grouped-QASM | `shuffled_dag` | 1234 | 0.9230 | 0.7118 | 0.6057 |

Topology is supported only if true DAG improves over both node-only and shuffled controls consistently across seeds and split families. Reversed-vs-true is a directional-signal diagnostic.
