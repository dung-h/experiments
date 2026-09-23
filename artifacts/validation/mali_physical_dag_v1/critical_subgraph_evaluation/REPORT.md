# Ma--Li critical-subgraph evaluation

The `rich_plus_critical` model adds high-criticality tail and induced-subgraph summaries to the same rich static descriptor baseline. All preprocessing and fits are train-only on the existing folds.

| Evaluation | Model | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|
| grouped-QASM | `rich_summary` | 0.4937 | 0.8895 | 0.8432 |
| grouped-QASM | `rich_plus_critical` | 0.4987 | 0.9041 | 0.8681 |
| strict backend+QASM | `rich_summary` | 0.5230 | 0.8755 | 0.8306 |
| strict backend+QASM | `rich_plus_critical` | 0.5306 | 0.8869 | 0.8470 |
| family held-out | `rich_summary` | 0.5753 | 0.8700 | 0.8117 |
| family held-out | `rich_plus_critical` | 0.5934 | 0.8732 | 0.8147 |

A gain over `rich_summary` must be stable across grouped, strict and family splits before a critical-subgraph model is preferred. If it is absent, the bottleneck descriptors are redundant with existing aggregate timing features.
