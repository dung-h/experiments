# Matched dense-statevector runtime matrix v1

**Finding recorded:** 20 September 2026.

## Scope and target

This is a local PyTorch dense-statevector reference-kernel experiment on one RTX 5070 Ti and a 28-logical-CPU machine, with PyTorch 2.11.0+cu128 and 16 configured CPU threads. It is not a Qiskit Aer, CUDA-Q, cuTensorNet, cloud-service or QPU measurement.

The target is `prepared_execute_reset_state`: allocate a fresh `|0...0>` state, apply a pre-materialized gate schedule and evaluate `<Z_0>`. Circuit construction, gate materialization, compiler/transpiler work, sampling and host-to-device result transfer are excluded. Cold post-prepare and end-to-end-first timings are retained as separate columns in the raw rows.

## Integrity and coverage

All **96/96** planned rows completed: 24 logical circuit configurations under 4 declared contexts. The target ranges from 0.000426 s to 24.011544 s. The maximum absolute difference in the final scalar `<Z_0>` between CPU complex64 and any other context is 3.185e-07; this is an output sanity check, not a state-fidelity study.

The median cold/warm ratio is 1.002; the median end-to-end-first/warm ratio is 1.129. Median repeat-level coefficient of variation is 0.334% (maximum 17.175%). CUDA peak allocated memory has median 50.23 MiB and maximum 904.36 MiB. The latter exceeds the final-state footprint because gate application creates temporary permuted/intermediate tensors.

| context_id      | rows | median_seconds | min_seconds | max_seconds |
| --------------- | ---- | -------------- | ----------- | ----------- |
| cpu:complex128  | 24   | 0.3767         | 0.0061      | 24.0115     |
| cpu:complex64   | 24   | 0.0521         | 0.0026      | 12.5194     |
| cuda:complex128 | 24   | 0.2369         | 0.0012      | 12.6629     |
| cuda:complex64  | 24   | 0.0132         | 0.0004      | 0.7344      |

## Matched device and precision effects

| paired ratio                | median  | minimum | maximum |
| --------------------------- | ------- | ------- | ------- |
| CPU complex128 / complex64  | 2.0224  | 0.9656  | 8.3897  |
| CUDA complex128 / complex64 | 15.7431 | 2.7553  | 19.6902 |
| CPU / CUDA complex64        | 5.0558  | 2.9395  | 25.0064 |
| CPU / CUDA complex128       | 1.8931  | 0.6778  | 5.2127  |

The factor is not constant. CPU complex128 is a median 2.02x slower than CPU complex64, whereas CUDA complex128 is a median 15.74x slower than CUDA complex64 in this implementation. These are measured characteristics of this software/hardware contract, not universal precision laws.

### 24-qubit frontier (warm median seconds)

| family           | depth_parameter | cpu:complex128 | cpu:complex64 | cuda:complex128 | cuda:complex64 |
| ---------------- | --------------- | -------------- | ------------- | --------------- | -------------- |
| ghz              | 1               | 1.5756         | 0.7724        | 0.3812          | 0.0310         |
| hea              | 2               | 5.9854         | 3.2061        | 3.1670          | 0.1842         |
| hea              | 4               | 12.2937        | 5.9894        | 6.3317          | 0.3668         |
| qaoa_cycle       | 2               | 7.6155         | 3.7526        | 2.8699          | 0.1886         |
| qaoa_cycle       | 4               | 13.7542        | 6.8247        | 5.0240          | 0.3425         |
| qft              | 1               | 19.8962        | 10.1120       | 5.1021          | 0.4044         |
| random_brickwork | 4               | 12.1620        | 6.4897        | 6.3651          | 0.3666         |
| random_brickwork | 8               | 24.0115        | 12.5194       | 12.6629         | 0.7344         |

The heaviest completed row is random brickwork, 24 qubits, eight layers, CPU complex128: 24.012 s. QFT-24 CPU complex128 is 19.896 s. Neither is censored or replaced by a synthetic timeout label.

## Baseline estimator evaluation

Models predict log warm runtime from logical width/depth/gate counts, family, device and precision. Reported multi-fold figures are computed over concatenated out-of-fold predictions, rather than averaging fold R² values.

| split            | model             | folds | n_test | mae_log | rmse_log | r2_log  | mae_seconds | rmse_seconds | r2_seconds |
| ---------------- | ----------------- | ----- | ------ | ------- | -------- | ------- | ----------- | ------------ | ---------- |
| circuit_group    | hist_gradient_log | 4     | 96     | 0.4309  | 0.6006   | 0.9534  | 0.8654      | 2.3315       | 0.7214     |
| circuit_group    | median            | 4     | 96     | 2.4121  | 2.8940   | -0.0808 | 2.0441      | 4.8486       | -0.2050    |
| circuit_group    | ridge_log         | 4     | 96     | 0.6790  | 0.7969   | 0.9180  | 1.3672      | 4.8818       | -0.2215    |
| context_held_out | hist_gradient_log | 4     | 96     | 1.3656  | 1.6196   | 0.6615  | 1.7934      | 3.6013       | 0.3352     |
| context_held_out | median            | 4     | 96     | 2.6212  | 3.1013   | -0.2412 | 2.0637      | 4.8450       | -0.2032    |
| context_held_out | ridge_log         | 4     | 96     | 1.3695  | 1.5080   | 0.7065  | 2.2472      | 6.1052       | -0.9105    |
| family_held_out  | hist_gradient_log | 5     | 96     | 0.5314  | 0.7602   | 0.9254  | 0.9151      | 2.4723       | 0.6867     |
| family_held_out  | median            | 5     | 96     | 2.4430  | 2.9339   | -0.1108 | 2.0468      | 4.8506       | -0.2060    |
| family_held_out  | ridge_log         | 5     | 96     | 0.9103  | 1.1116   | 0.8405  | 1.4935      | 5.4749       | -0.5364    |
| random_row       | hist_gradient_log | 1     | 24     | 0.2806  | 0.3976   | 0.9810  | 0.2820      | 0.7510       | 0.9688     |
| random_row       | median            | 1     | 24     | 2.5308  | 2.8924   | -0.0064 | 2.3344      | 4.8116       | -0.2807    |
| random_row       | ridge_log         | 1     | 24     | 0.6039  | 0.6744   | 0.9453  | 0.7628      | 1.7716       | 0.8264     |
| width_held_out   | hist_gradient_log | 3     | 96     | 2.6216  | 2.8156   | -0.0230 | 2.0490      | 4.7327       | -0.1481    |
| width_held_out   | median            | 3     | 96     | 3.4353  | 3.9386   | -1.0019 | 2.1603      | 4.8699       | -0.2156    |
| width_held_out   | ridge_log         | 3     | 96     | 1.1598  | 1.3396   | 0.7684  | 1.5832      | 3.6655       | 0.3113     |

The HGB random-row score (log R² 0.9810) is descriptive only: another context of the same logical circuit can appear in training. It falls to 0.9534 with circuit groups held out. Family-held-out HGB is 0.9254, but there are only five constructed families and 24 logical configurations. The harder width-held-out HGB score is -0.0230; Ridge reaches 0.7684, but only across three declared widths. Context-held-out HGB is 0.6615 log-R² and 0.3352 R² in seconds; a model should not be presented as device/precision agnostic on this evidence.

## Interpretation and limits

- The matrix supplies actual simulator execution labels, not analytical estimates. It can support a within-contract pilot estimator and a feasibility frontier.
- It does not yet test compilation/transpilation, sampling, noise, tensor-network contraction, a second GPU, another CPU, or a commercial/cloud simulator. Those are separate target domains.
- One fixed circuit seed and a small constructed family set are insufficient to claim broad unseen-circuit generalization. The width-held-out HGB failure also shows that a tree model trained only at 16, 20 and 24 qubits cannot be assumed to extrapolate. The next data increment should add independent circuit seeds, intermediate widths and an adaptive 26--30 qubit frontier with an explicit per-cell censoring policy.
- Aer CPU and CUDA-Q/cuStateVec GPU may be compared later only as separately labelled simulator domains. This matched matrix exists precisely to avoid confounding framework with device and precision.

## Reproduce

Use `experiments/simulator_runtime_v1/README.md`. Raw data are `records.jsonl` and `records.csv`; baseline OOF predictions, per-fold metrics and aggregate metrics are in `evaluation/`.
