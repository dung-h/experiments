# How to read this simulator pack

This is a **simulator-domain pack**. It is **not a QPU appendix** and is not
part of reader-table pack v1, v2, or v3. It copies frozen Wave 1C inventory
tables. It does not invent runtimes and does not flatten incompatible clocks
into a leaderboard.

There is **no flat leaderboard**. Local first/warm execution, quality-gated MPS
runtime, and selected-plan contraction estimates are different clocks from each
other and from archived observed QPU service time.

## Panel identity

`sim_common_q16_v1` has **204 unique panel members** (162 core, 42 frontier)
and **191 unique QASM hashes** with declared aliases allowed. Identity is the
panel-member row plus that row's exact QASM SHA-256. Do not require 204 unique
QASM hashes.

## Aer measurement is not Azizov

The 204-row reduced-warm capsule is local measurement
(`qiskit_aer_noisy_local`, `local_measurement`). The Azizov-style predictor
(`azizov_aer_independent`) is coverage-qualified on 162 core members only.
**Azizov frontier is blocked**, not imputed.

## Configuration cells stay split

CUDA-Q **first/warm and fp32/fp64 are never pooled**. MPS is **quality-gated**
(FP64, bond-16, cutoff 1e-10, gesvdj); quality_failed and adapter_error remain
in denominators. cuTensorNet **RUNTIME_EST is only vs matching warm** selected-
plan scalar contraction. Path-search and network-build are not estimator
targets.

## Blocked rather than substituted

Maestro, Pasqal (separate analog pulse/layout panel), Ma--Li simulator
source-native, structural DAG adaptation, family-aware residual, and Azizov
frontier are blocked. Missing evidence is not a zero-error score.

Local measurements and coverage-qualified predictors are `not_rankable` on the
QPU same-clock ranking. The cuTensorNet matching-warm pair is a
`paired_native_diagnostic`, still not a cross-engine rank.
