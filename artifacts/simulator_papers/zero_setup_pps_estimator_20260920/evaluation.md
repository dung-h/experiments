# PPS propagation-time estimator

The target is the local CPU propagation kernel only. Inputs are known pre-run circuit and approximation controls; post-run Pauli representation statistics are excluded.

| Split | Model | Log MAE | Log R2 | MAE seconds | R2 seconds |
|---|---|---:|---:|---:|---:|
| configuration_group_5fold | median_train_log | 1.685 | -0.348 | 0.184 | -0.311 |
| configuration_group_5fold | ridge | 0.935 | 0.521 | 0.114 | 0.606 |
| configuration_group_5fold | hist_gradient_boosting | 0.908 | 0.379 | 0.099 | 0.692 |
| delta_held_out | median_train_log | 1.731 | -0.355 | 0.186 | -0.318 |
| delta_held_out | ridge | 1.033 | 0.457 | 0.160 | -0.041 |
| delta_held_out | hist_gradient_boosting | 1.543 | 0.075 | 0.179 | -0.232 |
| trotter_steps_held_out | median_train_log | 1.673 | -0.416 | 0.184 | -0.322 |
| trotter_steps_held_out | ridge | 1.069 | 0.374 | 0.123 | 0.607 |
| trotter_steps_held_out | hist_gradient_boosting | 0.812 | 0.442 | 0.102 | 0.674 |

Delta- and Trotter-step-held-out results are extrapolation diagnostics. They do not claim cloud MPS or generic simulator runtime prediction.
