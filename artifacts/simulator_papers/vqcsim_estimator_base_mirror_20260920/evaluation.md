# VQCSim prepared-inference runtime estimator

This evaluation is VQCSim-only. The target is the GPU-synchronized prepared-circuit inference stage at batch size one.
Rows: **90**; variants: **base, mirror**; widths: **16–24**.

| Split | Model | Log MAE | Log R2 | MAE ms | R2 ms |
|---|---|---:|---:|---:|---:|
| random_5fold_descriptive | median_train_log | 0.910 | -0.112 | 193.118 | -0.171 |
| random_5fold_descriptive | ridge | 0.349 | 0.876 | 86.670 | 0.760 |
| random_5fold_descriptive | random_forest | 0.125 | 0.971 | 56.663 | 0.845 |
| random_5fold_descriptive | hist_gradient_boosting | 0.040 | 0.995 | 11.376 | 0.993 |
| circuit_group_5fold | median_train_log | 0.898 | -0.075 | 192.080 | -0.162 |
| circuit_group_5fold | ridge | 0.354 | 0.868 | 91.450 | 0.706 |
| circuit_group_5fold | random_forest | 0.180 | 0.944 | 78.157 | 0.694 |
| circuit_group_5fold | hist_gradient_boosting | 0.131 | 0.971 | 56.413 | 0.875 |
| family_held_out | median_train_log | 0.946 | -0.170 | 195.578 | -0.180 |
| family_held_out | ridge | 1.022 | -0.982 | 144.636 | 0.280 |
| family_held_out | random_forest | 0.507 | 0.651 | 130.128 | 0.321 |
| family_held_out | hist_gradient_boosting | 0.315 | 0.895 | 60.536 | 0.895 |
| width_held_out | median_train_log | 1.011 | -0.307 | 201.165 | -0.198 |
| width_held_out | ridge | 0.428 | 0.804 | 104.888 | 0.623 |
| width_held_out | random_forest | 0.337 | 0.827 | 113.376 | 0.562 |
| width_held_out | hist_gradient_boosting | 0.289 | 0.876 | 100.512 | 0.644 |

`circuit_group_5fold` is the primary split: base and mirror versions of the same family/size are not separated across train and test.
Family- and width-held-out rows are transfer diagnostics, not evidence for an all-simulator estimator.
