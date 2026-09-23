# Ma--Li coarsened topology controls

All variants use identical binned node features, splits, residual target and optimizer. Only adjacency differs. This is a retrained topology diagnostic, not a full native-DAG GNN result.

| Evaluation | Variant | Seed | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|---:|
| grouped-QASM | `node_only` | 2025 | 0.8228 | 0.7518 | 0.6823 |
| grouped-QASM | `node_only` | 31415 | 0.8603 | 0.7153 | 0.6346 |
| grouped-QASM | `true_dag` | 2025 | 0.8755 | 0.6836 | 0.6053 |
| grouped-QASM | `true_dag` | 31415 | 0.8627 | 0.7156 | 0.6308 |
| grouped-QASM | `reversed_dag` | 2025 | 0.9257 | 0.6506 | 0.5936 |
| grouped-QASM | `reversed_dag` | 31415 | 0.9054 | 0.6963 | 0.6056 |
| grouped-QASM | `shuffled_dag` | 2025 | 0.8914 | 0.6868 | 0.6282 |
| grouped-QASM | `shuffled_dag` | 31415 | 0.9622 | 0.6240 | 0.5571 |

Topology is supported only if true DAG improves over both node-only and shuffled controls consistently across seeds and split families. Reversed-vs-true is a directional-signal diagnostic.
