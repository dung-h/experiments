# S83 — Family-aware and Maestro adjudication

**Decision date:** 2026-10-02  
**Status:** signed scientific disposition; no new timing or training authorized.

## Family-aware residual

The available public Family-Aware evidence and the local simulator panel do
not share a trainable target.

The public Quantum Rings artifact has 36 circuits across four CPU/GPU and
precision contexts. Its documented 0.75 runtime is a mirror-sweep proxy; its
0.99 runtime is a 10,000-shot forward label. The local 204-member panel instead
contains noisy-Aer, CUDA-Q and contraction/MPS configuration cells with their
own clocks and, in the MPS case, a different fidelity policy. Neither is a
substitute for the other.

The local EMU-MPS threshold pilot has only 12 pulse-layout labels. It predicts
the smallest bond rung attaining fidelity 0.99 against a local reference; it
does not contain a digital-QASM runtime-plus-threshold matrix. Its families
have deterministic pulse-metadata signatures. It is useful as a companion
feasibility check, not a Family-Aware runtime benchmark.

Therefore the method status is **unavailable on the digital common panel**.
No runtime score, zero-filled label, or cross-target transfer is permitted.

A future local Family-Aware run requires a new signed ladder contract with:

1. a fixed engine/configuration and finite approximation ladder;
2. one warm-execution runtime label and one out-of-timing fidelity reference
   for every eligible exact-QASM/context pair;
3. a predeclared threshold-to-quality rule;
4. strict exact-QASM OOF plus a separate family-held-out test;
5. a prediction-time family representation. True test family identity may not
   be injected into a family-held-out model.

The routine worker may inventory compatible local labels and generate a
missingness report. That is not an experiment and cannot promote the method.

## Maestro component model

The completed ten-cell calibration pilot remains terminal at
`pilot_gate_failed`: nine statevector cells failed the prospective stability
gate and one MPS cell passed. It provides no accepted calibration curve and no
held-out predictor score. Existing partial/resume manifests and raw evidence
remain immutable.

No retry, resume, coefficient selection, or panel expansion is authorized by
this decision. A future experiment must be a cause-discriminating intervention,
not a longer rerun. Before it can start, a new protocol must state:

| Required element | Minimum content |
| --- | --- |
| Hypothesis | One concrete candidate cause, such as native thread-pool configuration, process transport, or host contention; do not state an assumed cause as fact. |
| Control and treatment | The baseline context and one changed factor. All remaining software, synthetic circuit, clock and affinity fields stay pinned. |
| Replication | Predetermined sessions/repetitions and a rule retaining every timing, error and timeout. |
| Measurements | Reported execution clock plus host/outer clocks as diagnostic traces, native thread-pool fingerprint, process lifecycle and CPU-affinity evidence. |
| Acceptance | Same prospective median/MAD stability rule for baseline and treatment. The treatment cannot be chosen after seeing a favorable timing. |
| Interpretation | The intervention can establish context sensitivity only. It cannot claim a paper-level Maestro predictor until a stable, fold-local calibration and held-out candidate matrix exist. |

The agent handling routine work may validate pilot pins, build a candidate/input
inventory, and write a non-executing intervention template. A strong reviewer
and explicit user authorization are required before any new Maestro API call.

## Consequence for the benchmark

The simulator method table must show:

- Family-Aware: `unavailable — no compatible local joint runtime/quality
  labels`.
- Maestro: `pilot_gate_failed — no accepted calibration or held-out score`.

These are scientifically meaningful terminal statuses. They must not be
silently removed from the method matrix or presented as a failed score of the
original papers.
