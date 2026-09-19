# Ma--Li direct compiled-runtime estimator

## Target and provenance

Target: Ma--Li recorded Osaka/Kyoto `result.time_taken` at fixed 1,024 shots.
Queue wait is not added. Compiled features and the weighted-duration path are
reconstructed with current FakeOsaka/FakeKyoto targets, so they are pre-run
proxy features rather than the authors' historical physical circuits.

- Input: `artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv`
- SHA-256: `be4e244a8079837927eaa3ea3f5472a5d4b5ad33575e82907d52ea94a0e1974e`
- Rows: 340; logical circuit groups: 170
- Backends: kyoto, osaka
- Families: 10

## Models

- `median_log`: train-fold log-median baseline.
- `physical_depth_ridge`: physical depth plus backend.
- `weighted_path_ridge`: current-target weighted critical path plus backend.
- `compiled_*`: logical, compiled physical, weighted-path and backend features.
- `weighted_path_plus_compiled_residual_rf`: cross-fitted duration-proxy Ridge
  plus a Random-Forest residual model. It uses no target-derived input at inference.

At inference, the input is a logical QASM circuit plus the intended target. The
circuit is transpiled under the frozen FakeBackend protocol, then the model sees
`log1p` logical width/depth/two-qubit-depth/gate-count; `log1p` physical depth,
physical two-qubit depth, total/two-qubit gate counts and SWAP count; `log10`
weighted critical-path seconds; and backend identity. The target is `log1p`
observed seconds. This evaluation intentionally saves out-of-fold predictions,
not a deployable final fitted model: model selection must be frozen first.

## Split protocol

- `grouped_logical_circuit_5fold`: primary within-domain test; each QASM SHA-256
  group is entirely train or test.
- `paired_backend_transfer_diagnostic`: hold out one backend, but permits that
  QASM's other-backend copy in training. It diagnoses paired-device variation only.
- `backend_and_logical_circuit_held_out`: for each of five QASM folds and each
  backend, test that backend/fold and train only on the other backend/other four
  folds. It removes the paired-circuit overlap.
- `family_held_out`: leave an algorithm-family token out. The family is used only
  to form the split, never as a model feature.

## Aggregate out-of-fold metrics

Every metric is calculated once over the complete vector of out-of-fold
predictions for that split/model, rather than averaging fold-level R² values.
This avoids unstable R² values from small circuit-family folds.

| Split | Model | OOF rows | Folds | Log MAE | Log R² | MAE (s) | MedAE (s) | R² (s) | Median relative error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| backend_and_logical_circuit_held_out | compiled_ridge | 340 | 10 | 0.0560 | 0.8742 | 0.5224 | 0.3776 | 0.8235 | 4.5% |
| backend_and_logical_circuit_held_out | compiled_random_forest | 340 | 10 | 0.0593 | 0.8499 | 0.5489 | 0.4071 | 0.7999 | 5.0% |
| backend_and_logical_circuit_held_out | weighted_path_plus_compiled_residual_rf | 340 | 10 | 0.0615 | 0.8753 | 0.5565 | 0.4285 | 0.8519 | 5.5% |
| backend_and_logical_circuit_held_out | compiled_hist_gradient_boosting | 340 | 10 | 0.0704 | 0.7826 | 0.6219 | 0.4790 | 0.7546 | 5.6% |
| backend_and_logical_circuit_held_out | physical_depth_ridge | 340 | 10 | 0.0754 | 0.8215 | 0.6792 | 0.5514 | 0.7870 | 6.7% |
| backend_and_logical_circuit_held_out | weighted_path_ridge | 340 | 10 | 0.0779 | 0.8126 | 0.7021 | 0.5678 | 0.7758 | 6.7% |
| backend_and_logical_circuit_held_out | median_log | 340 | 10 | 0.1628 | -0.0478 | 1.4406 | 1.1544 | -0.0219 | 13.0% |
| family_held_out | weighted_path_plus_compiled_residual_rf | 340 | 10 | 0.0638 | 0.8610 | 0.5630 | 0.4222 | 0.8573 | 5.2% |
| family_held_out | physical_depth_ridge | 340 | 10 | 0.0767 | 0.8155 | 0.6890 | 0.5413 | 0.7819 | 6.3% |
| family_held_out | weighted_path_ridge | 340 | 10 | 0.0798 | 0.8063 | 0.7196 | 0.5672 | 0.7672 | 7.1% |
| family_held_out | compiled_ridge | 340 | 10 | 0.0813 | 0.6734 | 0.6869 | 0.4570 | 0.7066 | 5.4% |
| family_held_out | compiled_hist_gradient_boosting | 340 | 10 | 0.0866 | 0.5562 | 0.7209 | 0.4425 | 0.6242 | 4.9% |
| family_held_out | compiled_random_forest | 340 | 10 | 0.0887 | 0.5956 | 0.7520 | 0.4894 | 0.6337 | 5.5% |
| family_held_out | median_log | 340 | 10 | 0.1686 | -0.0934 | 1.4964 | 1.2320 | -0.0807 | 13.8% |
| grouped_logical_circuit_5fold | compiled_ridge | 340 | 5 | 0.0542 | 0.8777 | 0.5116 | 0.3669 | 0.8238 | 4.5% |
| grouped_logical_circuit_5fold | compiled_random_forest | 340 | 5 | 0.0544 | 0.8545 | 0.5054 | 0.3563 | 0.8032 | 4.6% |
| grouped_logical_circuit_5fold | weighted_path_plus_compiled_residual_rf | 340 | 5 | 0.0582 | 0.8858 | 0.5303 | 0.3847 | 0.8658 | 4.7% |
| grouped_logical_circuit_5fold | compiled_hist_gradient_boosting | 340 | 5 | 0.0602 | 0.8272 | 0.5496 | 0.3841 | 0.7800 | 4.7% |
| grouped_logical_circuit_5fold | physical_depth_ridge | 340 | 5 | 0.0751 | 0.8246 | 0.6769 | 0.5426 | 0.7883 | 6.4% |
| grouped_logical_circuit_5fold | weighted_path_ridge | 340 | 5 | 0.0779 | 0.8147 | 0.7027 | 0.5564 | 0.7761 | 6.7% |
| grouped_logical_circuit_5fold | median_log | 340 | 5 | 0.1621 | -0.0382 | 1.4342 | 1.1489 | -0.0122 | 13.1% |
| paired_backend_transfer_diagnostic | compiled_ridge | 340 | 2 | 0.0537 | 0.9052 | 0.4992 | 0.3749 | 0.8785 | 4.5% |
| paired_backend_transfer_diagnostic | compiled_random_forest | 340 | 2 | 0.0563 | 0.8897 | 0.5206 | 0.4133 | 0.8582 | 5.1% |
| paired_backend_transfer_diagnostic | weighted_path_plus_compiled_residual_rf | 340 | 2 | 0.0581 | 0.8931 | 0.5287 | 0.4195 | 0.8733 | 5.4% |
| paired_backend_transfer_diagnostic | compiled_hist_gradient_boosting | 340 | 2 | 0.0601 | 0.8597 | 0.5389 | 0.3818 | 0.8335 | 4.8% |
| paired_backend_transfer_diagnostic | physical_depth_ridge | 340 | 2 | 0.0751 | 0.8236 | 0.6777 | 0.5441 | 0.7880 | 6.6% |
| paired_backend_transfer_diagnostic | weighted_path_ridge | 340 | 2 | 0.0775 | 0.8148 | 0.7002 | 0.5601 | 0.7769 | 6.5% |
| paired_backend_transfer_diagnostic | median_log | 340 | 2 | 0.1626 | -0.0438 | 1.4390 | 1.1537 | -0.0176 | 13.2% |

## Strict backend + unseen-circuit directions

There are only two device directions, so they are reported separately. The full
seven-model direction table is in `mali_direct_estimator_strict_backend_metrics.csv`.

| Held-out backend | Model | OOF rows | Log R² | MAE (s) | Median relative error |
|---|---|---:|---:|---:|---:|
| kyoto | compiled_ridge | 148 | 0.8491 | 0.5554 | 4.6% |
| kyoto | weighted_path_plus_compiled_residual_rf | 148 | 0.8617 | 0.5717 | 5.5% |
| kyoto | physical_depth_ridge | 148 | 0.8173 | 0.6666 | 5.9% |
| kyoto | median_log | 148 | -0.0139 | 1.3852 | 12.3% |
| osaka | compiled_ridge | 192 | 0.8908 | 0.4970 | 4.5% |
| osaka | weighted_path_plus_compiled_residual_rf | 192 | 0.8840 | 0.5447 | 5.5% |
| osaka | physical_depth_ridge | 192 | 0.8233 | 0.6889 | 6.9% |
| osaka | median_log | 192 | -0.0774 | 1.4834 | 13.2% |

## Reading this experiment

The primary within-domain result is grouped-logical-circuit performance. The
`paired_backend_transfer_diagnostic` is intentionally weaker: training sees
the same logical QASM on the other backend, so it isolates backend variation
but is not a new-circuit transfer result. `backend_and_logical_circuit_held_out`
removes both the backend and paired logical circuit from training. Family-held-out
is a separate algorithm-family transfer test. A model must beat the physical-depth
baseline on those strict tests before it can support a stronger claim.
This experiment is source-specific and must not be pooled with simulator,
Qonductor job-time, QPack workflow-event, or IonQ service-time labels.
