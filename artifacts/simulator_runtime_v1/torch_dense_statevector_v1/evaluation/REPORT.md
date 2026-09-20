# Dense-statevector v1 baseline evaluation

The target is prepared dense-statevector execution only. It excludes construction, gate materialization, compilation/transpilation and sampling.

Successful rows: 96; logical circuits: 24; contexts: 4.

Random-row evaluation is descriptive only because the same circuit may appear under another context. Circuit-group, family-held-out, width-held-out and context-held-out results are retained as stress tests.

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

A context-held-out score cannot establish cross-device transfer: this matrix has one physical GPU and one CPU. It tests whether the declared device/precision features are sufficient to extrapolate to an unseen execution context.
