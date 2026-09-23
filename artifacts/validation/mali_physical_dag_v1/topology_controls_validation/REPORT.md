# Ma--Li topology-control validation

This combines three independently trained seeds of the 256-bin grouped-QASM screen. The seed mean is used only to reduce initialization variance for analysis; it is not a selected model. Bootstrap resampling is by raw-QASM hash, preserving paired backend rows.

| Variant | Mean-seed MAE (s) | Mean-seed log-R² |
|---|---:|---:|
| `node_only` | 0.8145 | 0.7666 |
| `true_dag` | 0.8448 | 0.7382 |
| `reversed_dag` | 0.8967 | 0.6991 |
| `shuffled_dag` | 0.8530 | 0.7285 |

| Comparison | Δ log-R² 95% CI | Δ MAE (s) 95% CI | P(log-R² improves) | P(MAE improves) |
|---|---:|---:|---:|---:|
| `true_dag_minus_node_only` | [-0.0650, 0.0008] | [-0.0165, 0.0762] | 0.029 | 0.113 |
| `reversed_dag_minus_node_only` | [-0.1138, -0.0320] | [0.0357, 0.1286] | 0.000 | 0.000 |
| `shuffled_dag_minus_node_only` | [-0.0684, -0.0124] | [0.0009, 0.0755] | 0.002 | 0.022 |

## Decision

The control only supports topology if true DAG beats both node-only and shuffled adjacency with intervals favoring it. If it does not, the valid result is that this coarsened message-passing representation loses to node/timing summaries; it does not rule out every possible native-DAG architecture.
