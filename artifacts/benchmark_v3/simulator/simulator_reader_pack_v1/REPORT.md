# Simulator reader pack v1

This is a simulator-domain pack, not a QPU appendix. It copies Wave 1C
inventory tables and adds ranking/equivalence overlays. It contains no QPU
rows and no flat cross-engine or cross-clock ranking.

`sim_common_q16_v1` has 204 unique panel members (162 core, 42 frontier) and
191 unique QASM hashes with declared aliases. Aer measurement
(`qiskit_aer_noisy_local`) is not the Azizov predictor. Azizov is
coverage-qualified on 162 core members; frontier is blocked. CUDA-Q first/warm
and FP32/FP64 are never pooled. MPS is quality-gated. cuTensorNet RUNTIME_EST
is only a matching-warm selected-plan pair. Maestro, Pasqal analog, and Ma--Li
simulator adaptations are blocked rather than imputed.

Unique-cell pins copied from Wave 1C: dense 1224 per clock (1188 ok, 36
adapter_error); MPS 612 per clock (573 ok, 21 quality_failed, 18
adapter_error); cuTensorNet 612 per clock (603 ok, 9 timeout).
