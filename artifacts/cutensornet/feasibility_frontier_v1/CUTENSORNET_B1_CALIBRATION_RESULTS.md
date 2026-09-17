# cuTensorNet B0/B1 calibration evaluation

## Target and corpus

- Source: `cutensornet` only.
- Target semantics: `T_sim_contract: local GPU tensor-network scalar contraction`.
- Target: warm CUDA-event median `contract_gpu_median_s`, not path-search,
  construction, statevector materialisation, shots, QPU execution, or cloud
  turnaround.
- Rows: **55** successful, timeout-bounded feasibility cells; no
  failed cell is silently removed.
- Grid: GHZ and QFT plus HEA (2/4/8 layers), QAOA-cycle (p=2/3/6), and random
  brickwork (4/8/16 layers), each across widths 16, 20, 24, 28, 30.

## Models

- `median_train_log`: median log-time fitted on each training fold.
- `native_runtime_est_b0`: cuTensorNet's pre-run `RUNTIME_EST` without fitting.
- `median_ratio_calibrated_b0`: training-fold median log residual applied to
  B0; it tests whether a global multiplicative correction transfers.
- `ridge_b1` and `hist_gradient_boosting_b1`: environment-local calibrators
  using pre-contraction planner/circuit features only: B0 estimate, FLOPs,
  largest intermediate, slices, width, logical gate/depth features, family and
  variant.

## Out-of-fold metrics

| split | model | log MAE | log RMSE | log R2 | MAE (ms) | RMSE (ms) | R2 (seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_5fold_descriptive | median_train_log | 0.646 | 0.846 | -0.101 | 0.572 | 1.161 | -0.119 |
| random_5fold_descriptive | native_runtime_est_b0 | 1.479 | 1.508 | -2.500 | 2.645 | 3.377 | -8.469 |
| random_5fold_descriptive | median_ratio_calibrated_b0 | 0.120 | 0.303 | 0.859 | 0.212 | 0.870 | 0.371 |
| random_5fold_descriptive | ridge_b1 | 0.091 | 0.150 | 0.966 | 0.149 | 0.463 | 0.822 |
| random_5fold_descriptive | hist_gradient_boosting_b1 | 0.138 | 0.229 | 0.919 | 0.199 | 0.623 | 0.678 |
| family_held_out | median_train_log | 0.695 | 0.897 | -0.238 | 0.600 | 1.179 | -0.155 |
| family_held_out | native_runtime_est_b0 | 1.479 | 1.508 | -2.500 | 2.645 | 3.377 | -8.469 |
| family_held_out | median_ratio_calibrated_b0 | 0.131 | 0.308 | 0.854 | 0.222 | 0.875 | 0.364 |
| family_held_out | ridge_b1 | 0.247 | 0.373 | 0.786 | 0.394 | 1.024 | 0.129 |
| family_held_out | hist_gradient_boosting_b1 | 0.253 | 0.412 | 0.738 | 0.293 | 0.857 | 0.389 |
| width_held_out | median_train_log | 0.644 | 0.846 | -0.100 | 0.571 | 1.163 | -0.124 |
| width_held_out | native_runtime_est_b0 | 1.479 | 1.508 | -2.500 | 2.645 | 3.377 | -8.469 |
| width_held_out | median_ratio_calibrated_b0 | 0.121 | 0.302 | 0.859 | 0.214 | 0.870 | 0.372 |
| width_held_out | ridge_b1 | 0.089 | 0.150 | 0.965 | 0.151 | 0.466 | 0.819 |
| width_held_out | hist_gradient_boosting_b1 | 0.132 | 0.201 | 0.938 | 0.190 | 0.608 | 0.693 |
| variant_held_out | median_train_log | 0.729 | 0.910 | -0.274 | 0.621 | 1.181 | -0.159 |
| variant_held_out | native_runtime_est_b0 | 1.479 | 1.508 | -2.500 | 2.645 | 3.377 | -8.469 |
| variant_held_out | median_ratio_calibrated_b0 | 0.127 | 0.307 | 0.855 | 0.219 | 0.874 | 0.365 |
| variant_held_out | ridge_b1 | 0.230 | 0.363 | 0.797 | 0.380 | 1.010 | 0.153 |
| variant_held_out | hist_gradient_boosting_b1 | 0.265 | 0.411 | 0.741 | 0.322 | 0.866 | 0.377 |

`random_5fold_descriptive` is not a generalisation claim. The family, width
and variant held-out rows are the decision-relevant tests. Metrics are
calculated once from all out-of-fold predictions per split.

## Decision

For unseen **families**, the train-fold median-ratio calibration has log-MAE
**0.131**, better than Ridge (**0.247**) and HGB
(**0.253**). It is the provisional robust estimator for this fixed
cuTensorNet/GPU/precision/target environment. Ridge is strongest on unseen
widths (log-MAE **0.089** versus **0.121** for the
ratio calibration), but this does not transfer as well to unseen circuit
families. Use it only as an in-family refinement.

The corpus is still a first environment-local calibration set (55
rows). Do not select a universal estimator from a random split. The selected
calibration must be re-evaluated on a new GPU/software environment; otherwise
retain native B0 plus explicit calibration uncertainty and the feasibility
frontier.

Raw measurement and failure rows remain in the frontier output directory.
