# cuTensorNet pre-run estimator

This code is a local NVIDIA cuQuantum experiment, not CUDA-Q. It uses
cuTensorNet RUNTIME_EST after TIME_TUNED path optimization, then measures the
same scalar contraction with CUDA events.

Reference environment: CUDA 13, cuquantum-python-cu13 26.6.0,
cupy-cuda13x[ctk], RTX 5070 Ti, complex64. The 25-row and 55-row fixtures use
a 70% workspace policy, two warm-ups and five measured contractions. The
2026-09-23 QFT width frontier uses a 90% workspace policy and is a separate
table. Baseline outputs are fixed local fixtures; new hardware or package
versions require a new environment manifest and should not be compared as an
exact wall-clock reproduction.

Checked-in local artifacts:

- 25-row initial grid, 70% workspace: `artifacts/cutensornet/CUTENSORNET_RUNTIME_ESTIMATOR_INITIAL_RESULTS.md`
- 55-row feasibility frontier and warm B0/B1 calibration, 70% workspace: `artifacts/cutensornet/feasibility_frontier_v1/`
- QFT width frontier at 90% workspace, 2026-09-23: `artifacts/cutensornet/runtime_est_workspace90_frontier_v1_20260923/`
- Ma–Li MQT Bench QASM, q≤10, 90% workspace, 2026-09-23: `artifacts/cutensornet/mali_qasm_qle10_v1_20260923/`

`evaluate_cutensornet_b1.py` is warm-only on the 55-row 70% table. The 90%
frontier is a QFT contraction-path study: `RUNTIME_EST` versus warm CUDA-event
time, plus a plan-only q40 repeat. The Ma–Li QASM run uses the public 1,510
MQT Bench files, not generated tensors, and still times a local zero-bitstring
amplitude. Do not mix those seconds with CUDA-Q `sample` wall-clock or with
QPU labels.

The 25-row grid over-predicts cheap warm kernels. The 90% QFT rows from q36
onward under-predict by about 4–5×. The Ma–Li q≤10 QASM subset over-predicts
again (median actual/EST 0.231), with one reversal at `grover-v-chain_9`.
`RUNTIME_EST` is therefore a plan-cost proxy, not a uniformly conservative
wall-clock.
