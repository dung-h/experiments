# cuTensorNet pre-run estimator

This code is a local NVIDIA cuQuantum experiment, not CUDA-Q. It uses
cuTensorNet RUNTIME_EST after TIME_TUNED path optimization, then measures the
same scalar contraction with CUDA events.

Reference environment: CUDA 13, cuquantum-python-cu13 26.6.0,
cupy-cuda13x[ctk], RTX 5070 Ti, complex64, 70% workspace policy, two warm-ups
and five measured contractions. The baseline output is a fixed local fixture;
new hardware or package versions require a new environment manifest and should
not be compared as an exact wall-clock reproduction.
