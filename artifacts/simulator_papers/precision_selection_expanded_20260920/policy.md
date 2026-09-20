# cuTensorNet precision-policy evaluation

A `complex64` decision is called safe only when its scalar relative difference against the matched `complex128` result is at most `5e-07`.
The policy is trained and evaluated out-of-fold. This does not establish statevector fidelity, hardware-execution fidelity or a universal precision policy.

| Split | AUC | 64 selection | Unsafe given 64 | Policy / 128 time | Policy / oracle time |
|---|---:|---:|---:|---:|---:|
| random_5fold_descriptive | 0.869 | 15.4% | 0.0% | 0.981× | 1.233× |
| family_held_out | 0.539 | 5.8% | 66.7% | 0.981× | 1.232× |
| width_held_out | 0.760 | 23.1% | 16.7% | 0.965× | 1.211× |
| circuit_group_5fold | 0.780 | 15.4% | 25.0% | 0.977× | 1.228× |
