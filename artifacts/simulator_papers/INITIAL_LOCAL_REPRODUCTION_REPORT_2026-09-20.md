# Simulator papers: initial local reproduction report

**Local date:** 20 September 2026
**Status:** three paper tracks were checked against public artifacts. SP-1 and the credential-free local part of SP-3 were executed on this workstation; SP-2's existing independent result was rerun from its committed raw table and matched byte-for-byte.

This report deliberately keeps three timing targets separate. It is evidence for controlled simulator-data generation, not evidence for a universal runtime estimator.

| Track | Provenance | Target kept in the data | Outcome of this pass |
| --- | --- | --- | --- |
| SP-1 VQCSim | Public source, local execution | GPU float32 prepared-circuit inference time | 45/45 rows completed and passed parity. |
| SP-2 Azizov et al. | Independent local reconstruction; author runtime/code artifact unavailable | Noisy Aer execution after fake-backend transpilation | Committed 1,192-row evaluation reran identically. |
| SP-3 Zero-Setup | Public source, bounded local component | CPU time inside Qiskit `pauli-prop` propagation only | 25/25 rows completed over five approximation thresholds. |

The paper sources and pinned revisions are recorded in [`experiments/simulator_papers/README.md`](../../experiments/simulator_papers/README.md) and [`upstream/upstream.lock.json`](../../upstream/upstream.lock.json).

## SP-1 — VQCSim GPU statevector/inference canary

### Contract

The public VQCSim RQ2 code was checked out at `7fc274d448c5e67a7a4149088afa594203cd5707`. Its generic Torch lowering was run on an RTX 5070 Ti with Torch `2.11.0+cu128`, Python `3.13.15`, MQT Bench `2.1.0` and Qiskit `2.5.2`.

Each row is one MQT Bench variational circuit at batch size 1, float32. The circuit is prepared once; `prepare_ms` is retained separately. The reported `mean_wall_ms` is the GPU-synchronized repeated inference stage (`1` warm-up; `3` repeats of `3` inference iterations). Qiskit parity is a correctness check, not the timing reference. This target does not include Aer simulation, transpilation, CPU preparation, sampling, or end-to-end application time.

### Coverage and results

- Families: QAOA, QNN, VQE Real-Amplitudes, VQE SU2, VQE Two-Local.
- Widths: 16--24 qubits, one circuit per integer width and family.
- Rows: 45; terminal status: 45 `ok`; parity: 45/45 passed; no OOM or timeout.
- Peak process VRAM observed by the runner grew from roughly 302 MiB at 16 qubits to 812 MiB at 24 qubits. It is process peak, not a per-circuit allocation measurement, so it must not be used as a standalone memory model.

| Family | 16q mean ms | 20q mean ms | 22q mean ms | 24q mean ms |
| --- | ---: | ---: | ---: | ---: |
| QAOA | 67.681 | 134.784 | 467.825 | 1,864.056 |
| QNN | 27.558 | 43.933 | 113.438 | 440.379 |
| VQE Real-Amplitudes | 27.998 | 42.543 | 104.399 | 403.761 |
| VQE SU2 | 53.131 | 78.893 | 182.269 | 669.904 |
| VQE Two-Local | 39.028 | 82.222 | 342.176 | 1,721.697 |

Two observations matter for the estimator project:

1. Width is insufficient even with exactly one simulator, one precision and one GPU. At 24 qubits, QAOA is 4.62 times the Real-Amplitudes runtime; Two-Local is 4.26 times it.
2. The 20-to-24-qubit increase is strongly nonlinear and family-dependent. A model must have circuit structure and a declared backend configuration; a width/depth-only pooled label would conceal this boundary.

Raw records, source manifest and summary are in [`vqcsim_rq2_gpu_fp32_16_24_20260920`](vqcsim_rq2_gpu_fp32_16_24_20260920). This is a successful local **canary**, not a reproduction of the paper's multi-framework, five-GPU comparison and not yet a train/test estimator study.

## SP-2 — Azizov et al. noisy-Aer estimator reconstruction

The published paper does not link its code or measured 1,402-circuit table. Its exact numerical result is therefore not reproducible from author artifacts. The repository instead contains an independent reconstruction: QASM source circuits, current FakeSherbrooke/FakeWashingtonV2 snapshots, four optimization levels and noisy Aer at 1,024 shots.

This pass reran `evaluate_subset_splits.py` on the committed 1,192 successful rows with seed 1234. Both regenerated output files are byte-identical to the committed fixtures. This verifies the estimator analysis, not historical timing on the authors' machine.

| Split | Best feature block/model | log-R2 | seconds R2 | RMSE seconds |
| --- | --- | ---: | ---: | ---: |
| Grouped logical circuit | hybrid Random Forest | 0.8558 | 0.8230 | 0.2504 |
| Family held out | hybrid Random Forest | 0.8375 | 0.7923 | 0.3521 |
| Hold out FakeSherbrooke | source Random Forest | 0.5845 | 0.5308 | 0.5491 |
| Hold out FakeWashingtonV2 | hybrid SVR | 0.6297 | 0.5501 | 0.3712 |

The important result is the gap between within-domain grouped/family results and backend-held-out transfer. It supports compiled features as useful local signal, but does not support a backend-agnostic claim. The independent protocol, raw rows and split definitions are in [`tracks/azizov_independent`](../../tracks/azizov_independent) and [`artifacts/azizov_independent`](../azizov_independent).

## SP-3 — Zero-Setup local Pauli-propagation canary

### Contract

The public artifact was checked out at `c4485f1c267410d291a717d05aa12d493559f337`. The executed bounded runner reads the artifact's own 127-qubit IBM heavy-hex edge list and constructs the same 20-step TFI circuit:

- 127 qubits, 147 topology edges, circuit depth 378, 5,480 gates;
- `RZZ(-pi/2)` on every edge then `RX(pi/4)` on every qubit per Trotter step;
- observable `Z_62`;
- Qiskit `2.5.2`, `pauli-prop` `0.2.1`, CPU execution.

Circuit construction and conversion to rotation gates are done once and recorded separately. The target `propagation_seconds` is only the `propagate_through_rotation_gates` call. It is neither full simulator wall-clock nor BlueQubit MPS runtime. The upstream MPS/statevector scripts submit jobs to the BlueQubit cloud and were not invoked.

### Results

Every threshold has five repeated runs. The source's `max_terms` cap is much larger than the observed count in these rows, so the reported approximation threshold, rather than the cap, is active here.

| Absolute threshold delta | Mean propagation s | Sample SD s | Retained Pauli terms | Truncated norm | `<Z_62>` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1e-2` | 0.016863 | 0.000570 | 120 | 103.831 | 0.265435 |
| `5e-3` | 0.045181 | 0.002226 | 160 | 209.261 | 0.274812 |
| `1e-3` | 0.920909 | 0.053675 | 5,089 | 959.436 | 0.257891 |
| `5e-4` | 3.379542 | 0.004375 | 14,451 | 1,927.750 | 0.232409 |
| `1e-4` | 97.546063 | 1.817785 | 389,921 | 9,635.710 | 0.172506 |

Tightening delta from `1e-2` to `1e-4` makes the same circuit about 5,785 times slower and grows the retained representation about 3,249 times. This is a direct example of an execution-control feature (accuracy/truncation) that must be present in an MPS/PPS runtime estimator; circuit width and depth are fixed across all 25 rows.

The 25 raw rows and environment manifest are [`zero_setup_pps_local_canary_20260920.jsonl`](zero_setup_pps_local_canary_20260920.jsonl) and [`zero_setup_pps_local_canary_20260920.environment.json`](zero_setup_pps_local_canary_20260920.environment.json). The bounded reproducible runner is [`run_zero_setup_pps_canary.py`](../../experiments/simulator_papers/run_zero_setup_pps_canary.py).

## Decision after this pass

Simulator data generation is feasible and should continue, but only as distinct labelled domains:

- VQCSim is a GPU statevector/inference domain;
- Azizov is noisy Aer after hardware-aware transpilation;
- SP-3 local Pauli propagation is an approximate observable-propagation domain.

The next useful experiment is not to concatenate these 1,262 rows. It is to expand each domain with controlled feature variation and then evaluate an estimator within domain using circuit-family groups and held-out runtime configuration. A separately designed matched CPU/GPU x float32/float64 statevector matrix can answer cross-configuration questions, but must have its own target specification and report.

## Supplemental SP-4 to SP-6 probes

Three subsequent bounded local probes are recorded separately in
[`SP4_SP6_INDEPENDENT_PROBES_2026-09-20.md`](SP4_SP6_INDEPENDENT_PROBES_2026-09-20.md).
They cover explicit contraction-plan choice, a family-aware EMU-MPS threshold
check and a complex64/complex128 trade-off. They do not enlarge or alter the
three target domains in this report.
