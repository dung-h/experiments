# Simulator reader tables v1

This evidence-only reader pack preserves local simulator configurations and their
native clocks. It contains no QPU rows and no flat cross-engine or cross-clock
ranking. `sim_common_q16_v1` contains 204 unique panel members: 162 core and 42
frontier. Declared QASM hash aliases are allowed (191 unique `qasm_sha256`
values). Identity is the panel-member row plus that row's exact QASM SHA-256.
Terminal failures remain in denominators.

The Aer reduced-warm capsule is a local noisy-Aer measurement (204/204) under
reader overlay `qiskit_aer_noisy_local` (`local_measurement`). Registry V1 has
no distinct Aer measurement ID; `azizov_aer_independent` is reserved for the
Azizov-style predictor and is coverage-qualified on the 162 core members only;
frontier predictor rows are blocked, not imputed. CUDA-Q first and warm clocks, and
FP32/FP64 dense configurations, are separate. Unique-cell pins: dense 1224 per
clock (1188 ok, 36 adapter_error); MPS 612 per clock (573 ok, 21 quality_failed,
18 adapter_error); cuTensorNet 612 per clock (603 ok, 9 timeout). MPS is
quality-qualified (FP64, bond 16, cutoff 1e-10, gesvdj). cuTensorNet RUNTIME_EST
is only a selected-plan pair against matching warm scalar contraction;
network-build and interference-flagged path-search values are appendix
telemetry. Maestro and unavailable adaptations are blocked rather than
substituted; Pasqal is kept on its separate analog pulse/layout panel.
