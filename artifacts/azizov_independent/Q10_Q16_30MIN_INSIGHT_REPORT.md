# Q10–Q16 30-minute canary: size and timeout audit

**Audit date:** 2026-09-18
**Scope:** `FakeWashingtonV2`, optimization level 0, 1024 shots, seed 1234,
36 GiB RSS cap, 1,800-second wall timeout.

## What completed

The canary produced two rows before it was stopped:

| Circuit | Logical width | Logical depth | Logical operations | Logical 2Q operations | Result |
|---|---:|---:|---:|---:|---|
| `ae_indep_qiskit_10` | 10 | 54 | 102 | 54 | wall timeout at 1,800 s |
| `ae_indep_qiskit_12` | 12 | 66 | 135 | 77 | wall timeout at 1,800 s |

Both rows reached approximately 16.42 GiB RSS. No successful Aer execution row
was produced by this 30-minute run. The remaining orphaned child process was
terminated after the audit; no canary process remains.

## Why a 10–12-qubit circuit becomes large

A transpile-only audit with the same backend, seed and optimization level gives:

| Circuit | Declared physical register | Active physical qubits | Physical depth | Physical operations | Physical CX gates |
|---|---:|---:|---:|---:|---:|
| `ae_indep_qiskit_10` | 127 | 29 | 362 | 654 | 387 |
| `ae_indep_qiskit_12` | 127 | 30 | 498 | 1,045 | 687 |

The backend has 127 physical qubits. Routing the controlled-phase/CX-heavy
amplitude-estimation circuits onto that topology introduces many additional
CX operations and increases active width from 10/12 to 29/30. The declared
register alone is not the relevant simulator cost; the active physical width
after transpilation is.

For double-complex statevectors, the raw state size is approximately:

```text
29 active qubits: 2^29 × 16 bytes ≈ 8 GiB
30 active qubits: 2^30 × 16 bytes ≈ 16 GiB
```

Aer was created with `method="automatic"`, CPU device, noise from the fake
backend, and parallelism left at its defaults (`max_parallel_threads=0`,
`max_parallel_shots=0`, `max_parallel_experiments=0` in the canary command).
Shot/noise branching and simulator work buffers therefore add substantial
overhead. This explains why the process plateaued near 16.42 GiB and made no
progress within 1,800 seconds.

## Interpretation

1. Logical width/depth alone are not sufficient runtime predictors.
2. Physical active width, physical 2Q count/depth, routing overhead and Aer
   parallelism are first-class features.
3. The two rows are right-censored timeout observations, not measured runtimes.
4. The CSV generated before the runner fix labels every non-completed row as
   `resource_limit`; the authoritative field is `stop_reason=wall_timeout>1800s`.
   New canary runs now distinguish `timeout` from `resource_limit`.
5. A fair follow-up should explicitly set `max_parallel_threads=1`,
   `max_parallel_shots=1`, and record the Aer method (`statevector`, MPS or
   cuStateVec) before comparing simulator runtimes.
