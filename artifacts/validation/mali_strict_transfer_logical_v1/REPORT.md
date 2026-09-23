# Ma--Li strict no-real-QASM transfer screen

All 170 real-QPU QASM hashes are removed from simulator pretraining. The model is a lightweight shared logical-feature MLP, not the original Graph Transformer; it tests strict circuit-inductive transfer in a common source/target representation.

Simulator rows: 3,020 before filtering, 2,636 after filtering. Removed rows: 384.

| Mode | Seed | MAE (s) | log-R² | seconds-R² |
|---|---:|---:|---:|---:|
| `scratch` | 1234 | 0.7283 | 0.2008 | 0.5226 |
| `scratch` | 2025 | 0.6512 | 0.3287 | 0.5679 |
| `scratch` | 31415 | 0.7278 | 0.3745 | 0.5319 |
| `frozen_transfer` | 1234 | 1.8391 | -0.1587 | -0.3494 |
| `frozen_transfer` | 2025 | 2.2189 | -0.7625 | -0.9091 |
| `frozen_transfer` | 31415 | 2.1061 | -0.8132 | -0.9442 |
| `finetune_transfer` | 1234 | 0.7655 | 0.2674 | 0.4872 |
| `finetune_transfer` | 2025 | 0.6258 | 0.1170 | 0.5631 |
| `finetune_transfer` | 31415 | 0.8057 | -0.0937 | 0.4055 |

A strict transfer benefit requires frozen or finetuned transfer to beat scratch consistently. This screen cannot establish that the full original DAG transformer transfers, but it does remove exact circuit identity leakage from the common logical representation.
