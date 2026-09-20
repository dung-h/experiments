# Noisy-Aer source-aware runtime intervals

All point estimates and conformal radii are fitted within each outer training fold. The target excludes transpilation; it is not QPU runtime or cloud turnaround.

| Split | N | Log MAE | Log R2 | MAE seconds | R2 seconds | Interval coverage | Mean width s |
|---|---:|---:|---:|---:|---:|---:|---:|
| grouped_circuit_5fold | 1192 | 0.068 | 0.559 | 0.167 | 0.239 | 90.5% | 0.524 |
| family_held_out | 1192 | 0.067 | 0.553 | 0.154 | 0.348 | 87.5% | 0.360 |
| backend_held_out | 1192 | 0.133 | 0.356 | 0.257 | 0.405 | 21.7% | 0.247 |

Backend-held-out metrics are the relevant warning for fake-backend domain transfer. Censored resource rows remain a separate feasibility boundary.
