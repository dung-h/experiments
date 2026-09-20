# EMU-MPS runtime formula calibration

Target: `sum_step_seconds` (sum of EMU-MPS per-step durations).

The documented diagnostic form is `T ≈ α N²χ³ + β N³χ²`.
This run uses observed maximum `χ`, so it is post-run calibration, not a pre-run estimate.

- alpha: `-1.69303181472e-05`
- beta: `1.63972675639e-05`
- matrix rank: `2`
- condition number: `48.124`
- diagnostic: The fitted alpha is negative and the design matrix is ill-conditioned; this is not a stable predictive estimator yet.

| case | actual step sum (s) | predicted (s) | relative error |
|---|---:|---:|---:|
| n16 | 13.102 | 1.271 | -90.3% |
| n24 | 17.288 | 17.598 | 1.8% |
| n30 | 22.412 | 22.832 | 1.9% |

The pilot is too small and `N` and `χ` are correlated, so the two terms cannot yet be separated reliably. More runs varying `max_bond_dim`, pulse duration and sequence family are required before using this as a pre-run predictor.
