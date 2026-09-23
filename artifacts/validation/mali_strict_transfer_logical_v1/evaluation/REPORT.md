# Strict no-real-QASM transfer evaluation

All 170 real-QPU hashes were excluded from source pretraining. Evaluation groups rows by raw QASM hash and compares three-seed mean predictions; bootstrap resamples the 170 QASM groups.

| Mode | MAE (s) | log-R² |
|---|---:|---:|
| `finetune_transfer` | 0.6956 | 0.0816 |
| `frozen_transfer` | 1.9152 | -0.2533 |
| `scratch` | 0.6831 | 0.3360 |

| Comparison | Δ log-R² 95% CI | Δ MAE 95% CI | P(log-R² improves) | P(MAE improves) |
|---|---:|---:|---:|---:|
| `frozen_transfer_minus_scratch` | [-1.3367, 0.6642] | [1.0097, 1.4335] | 0.139 | 0.000 |
| `finetune_transfer_minus_scratch` | [-0.8351, 0.0000] | [-0.0354, 0.0572] | 0.026 | 0.291 |

Frozen transfer failing establishes source/target representation mismatch in this strict screen. Fine-tuning would need to beat scratch consistently before it can be described as useful transfer; this does not adjudicate the unavailable full Graph Transformer strict experiment.
