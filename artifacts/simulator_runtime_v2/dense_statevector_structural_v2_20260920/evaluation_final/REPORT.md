# Dense-statevector v2 estimator evaluation

The target is local prepared dense-statevector execution. Resource-limit observations are retained in the raw corpus but excluded from duration regression.

Successful duration rows: 204; logical circuits: 68; source runs: {'v2_structural': 108, 'v1_reference': 96}.

Random-row scores are diagnostic only. The primary `circuit_group` split keeps every CPU/GPU/precision context of a logical test circuit outside training.
The optional `seed_held_out_structural` split is GPU-only and restricted through q24 because seeds 43/101 were intentionally collected only for the two GPU contexts at those widths; it tests unseen random topologies under matched coverage.

## Runtime prediction

| split                    | model                       | folds | n_test | mae_log | rmse_log | r2_log  | mae_seconds | rmse_seconds | r2_seconds |
| ------------------------ | --------------------------- | ----- | ------ | ------- | -------- | ------- | ----------- | ------------ | ---------- |
| circuit_group            | analytical_calibrated_ridge | 5     | 204    | 0.4252  | 0.5411   | 0.9633  | 1.0208      | 2.7199       | 0.8591     |
| circuit_group            | analytical_hgb              | 5     | 204    | 0.3043  | 0.5240   | 0.9656  | 0.7495      | 2.7205       | 0.8590     |
| circuit_group            | graph_hgb                   | 5     | 204    | 0.2280  | 0.4117   | 0.9788  | 0.7740      | 2.9700       | 0.8320     |
| circuit_group            | graph_hgb_with_family       | 5     | 204    | 0.2272  | 0.4140   | 0.9785  | 0.7603      | 2.9385       | 0.8355     |
| circuit_group            | logical_hgb                 | 5     | 204    | 0.2218  | 0.4058   | 0.9794  | 0.6649      | 2.5180       | 0.8792     |
| circuit_group            | median_log                  | 5     | 204    | 2.5491  | 2.9409   | -0.0829 | 3.1137      | 7.8316       | -0.1683    |
| circuit_group            | structural_ridge            | 5     | 204    | 0.5025  | 0.6847   | 0.9413  | 1.0771      | 2.7970       | 0.8510     |
| family_held_out          | analytical_calibrated_ridge | 7     | 204    | 0.4209  | 0.5324   | 0.9645  | 1.0291      | 2.7392       | 0.8571     |
| family_held_out          | analytical_hgb              | 7     | 204    | 0.3101  | 0.5249   | 0.9655  | 0.7167      | 3.0284       | 0.8253     |
| family_held_out          | graph_hgb                   | 7     | 204    | 0.3012  | 0.5222   | 0.9659  | 0.8544      | 3.2708       | 0.7962     |
| family_held_out          | graph_hgb_with_family       | 7     | 204    | 0.2991  | 0.5093   | 0.9675  | 0.8754      | 3.3383       | 0.7877     |
| family_held_out          | logical_hgb                 | 7     | 204    | 0.2884  | 0.5002   | 0.9687  | 0.6829      | 2.7111       | 0.8600     |
| family_held_out          | median_log                  | 7     | 204    | 2.4777  | 2.8621   | -0.0256 | 3.0994      | 7.8329       | -0.1687    |
| family_held_out          | structural_ridge            | 7     | 204    | 0.5974  | 0.7389   | 0.9316  | 1.4288      | 3.7417       | 0.7333     |
| random_row               | analytical_calibrated_ridge | 1     | 51     | 0.4311  | 0.5725   | 0.9616  | 1.2327      | 3.0979       | 0.8726     |
| random_row               | analytical_hgb              | 1     | 51     | 0.1986  | 0.3273   | 0.9875  | 0.8011      | 3.1018       | 0.8723     |
| random_row               | graph_hgb                   | 1     | 51     | 0.1949  | 0.2874   | 0.9903  | 1.2350      | 4.4071       | 0.7422     |
| random_row               | graph_hgb_with_family       | 1     | 51     | 0.1992  | 0.2870   | 0.9904  | 1.2399      | 4.4051       | 0.7424     |
| random_row               | logical_hgb                 | 1     | 51     | 0.2102  | 0.3101   | 0.9887  | 1.0945      | 3.8352       | 0.8047     |
| random_row               | median_log                  | 1     | 51     | 2.5904  | 2.9431   | -0.0143 | 3.8847      | 9.4590       | -0.1877    |
| random_row               | structural_ridge            | 1     | 51     | 0.4331  | 0.5872   | 0.9596  | 1.1106      | 2.7642       | 0.8986     |
| seed_held_out_structural | analytical_calibrated_ridge | 3     | 72     | 0.3146  | 0.3701   | 0.9789  | 0.5300      | 1.4490       | 0.8391     |
| seed_held_out_structural | analytical_hgb              | 3     | 72     | 0.0345  | 0.0468   | 0.9997  | 0.1050      | 0.2851       | 0.9938     |
| seed_held_out_structural | graph_hgb                   | 3     | 72     | 0.0333  | 0.0447   | 0.9997  | 0.1086      | 0.2946       | 0.9934     |
| seed_held_out_structural | graph_hgb_with_family       | 3     | 72     | 0.0333  | 0.0447   | 0.9997  | 0.1086      | 0.2946       | 0.9934     |
| seed_held_out_structural | logical_hgb                 | 3     | 72     | 0.0369  | 0.0426   | 0.9997  | 0.0884      | 0.2440       | 0.9954     |
| seed_held_out_structural | median_log                  | 3     | 72     | 2.2624  | 2.5579   | -0.0070 | 1.7037      | 3.9580       | -0.2003    |
| seed_held_out_structural | structural_ridge            | 3     | 72     | 0.2962  | 0.3393   | 0.9823  | 0.4183      | 1.1422       | 0.9000     |
| width_held_out           | analytical_calibrated_ridge | 5     | 204    | 0.7858  | 0.9974   | 0.8754  | 1.5136      | 3.7634       | 0.7302     |
| width_held_out           | analytical_hgb              | 5     | 204    | 0.8833  | 1.0981   | 0.8490  | 2.3818      | 6.0181       | 0.3101     |
| width_held_out           | graph_hgb                   | 5     | 204    | 1.3712  | 1.5056   | 0.7162  | 2.5049      | 6.0993       | 0.2914     |
| width_held_out           | graph_hgb_with_family       | 5     | 204    | 1.3676  | 1.5015   | 0.7177  | 2.5054      | 6.1062       | 0.2898     |
| width_held_out           | logical_hgb                 | 5     | 204    | 1.3303  | 1.4698   | 0.7295  | 2.4731      | 6.0642       | 0.2995     |
| width_held_out           | median_log                  | 5     | 204    | 3.4714  | 3.9348   | -0.9384 | 3.2932      | 7.8714       | -0.1802    |
| width_held_out           | structural_ridge            | 5     | 204    | 0.8021  | 1.0222   | 0.8692  | 1.4940      | 3.5859       | 0.7551     |

The ablation is cumulative: `analytical_hgb` tests nonlinear capacity using only statevector bytes and gate-work; `logical_hgb` adds depth/gate counts; `graph_hgb` adds interaction-graph features; `graph_hgb_with_family` additionally exposes the coarse family label. These are calibrated estimators for this one PyTorch kernel, not universal simulator predictors.

## Context-selection task

| model                       | complete_circuit_groups | exact_fastest_rate | median_regret_ratio | mean_regret_seconds |
| --------------------------- | ----------------------- | ------------------ | ------------------- | ------------------- |
| analytical_calibrated_ridge | 36                      | 1.0000             | 1.0000              | 0.0000              |
| analytical_hgb              | 36                      | 1.0000             | 1.0000              | 0.0000              |
| graph_hgb                   | 36                      | 1.0000             | 1.0000              | 0.0000              |
| graph_hgb_with_family       | 36                      | 1.0000             | 1.0000              | 0.0000              |
| logical_hgb                 | 36                      | 1.0000             | 1.0000              | 0.0000              |
| median_log                  | 36                      | 0.0000             | 4.8552              | 2.2841              |
| structural_ridge            | 36                      | 1.0000             | 1.0000              | 0.0000              |

Oracle fastest-context distribution:

| oracle_context | circuit_groups |
| -------------- | -------------- |
| cuda:complex64 | 36             |

This task considers only complete four-context circuit groups held out by `circuit_group`. A low regret is useful only if the fastest context varies; a single dominating oracle context makes this a hardware fact rather than a model contribution.

## GPU memory envelope

| context_id      | num_qubits | successful_rows | max_peak_reserved_over_statevector | median_peak_reserved_over_statevector | q95_peak_reserved_over_statevector | device_total_memory_bytes |
| --------------- | ---------- | --------------- | ---------------------------------- | ------------------------------------- | ---------------------------------- | ------------------------- |
| cuda:complex128 | 16         | 20              | 24.0000                            | 24.0000                               | 24.0000                            | 16608788480.0000          |
| cuda:complex128 | 20         | 20              | 4.3750                             | 4.3750                                | 4.3750                             | 16608788480.0000          |
| cuda:complex128 | 24         | 20              | 4.0859                             | 4.0859                                | 4.0859                             | 16608788480.0000          |
| cuda:complex128 | 26         | 4               | 4.0215                             | 4.0215                                | 4.0215                             | 16608788480.0000          |
| cuda:complex64  | 16         | 20              | 44.0000                            | 44.0000                               | 44.0000                            | 16608788480.0000          |
| cuda:complex64  | 20         | 20              | 5.2500                             | 5.2500                                | 5.2500                             | 16608788480.0000          |
| cuda:complex64  | 24         | 20              | 4.1719                             | 4.1719                                | 4.1719                             | 16608788480.0000          |
| cuda:complex64  | 26         | 4               | 4.0430                             | 4.0430                                | 4.0430                             | 16608788480.0000          |
| cuda:complex64  | 28         | 4               | 4.0107                             | 4.0107                                | 4.0107                             | 16608788480.0000          |

The envelope is empirical and context-specific. Raw statevector bytes are a lower bound, not a peak-allocation guarantee because the gate kernel creates temporary tensors.

## Resource-frontier prediction

| context_id      | num_qubits | statevector_bytes | device_total_memory_bytes | observed_rows | observed_ok | observed_resource_limit | observed_error | calibration_largest_width | calibration_peak_ratio | raw_statevector_fits | envelope_peak_bytes | envelope_fits |
| --------------- | ---------- | ----------------- | ------------------------- | ------------- | ----------- | ----------------------- | -------------- | ------------------------- | ---------------------- | -------------------- | ------------------- | ------------- |
| cuda:complex128 | 26         | 1073741824        | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.0859                 | True                 | 4387241984.0000     | True          |
| cuda:complex128 | 28         | 4294967296        | 16608788480.0000          | 4             | 0           | 4                       | 0              | 24                        | 4.0859                 | True                 | 17548967936.0000    | False         |
| cuda:complex64  | 26         | 536870912         | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.1719                 | True                 | 2239758336.0000     | True          |
| cuda:complex64  | 28         | 2147483648        | 16608788480.0000          | 4             | 4           | 0                       | 0              | 24                        | 4.1719                 | True                 | 8959033344.0000     | True          |

This table calibrates a peak-allocation ratio only through q24, then applies it to wider actual observations. `raw_statevector_fits` is the naive lower-bound decision; `envelope_fits` is the context-calibrated decision.
