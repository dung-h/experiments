# Maestro-style component adaptation: optimizer-off QCSim

This is a new, host-specific follow-up to the failed optimizer-on calibration.
It is a local component-model adaptation, not a reproduction of Maestro's full
Composer estimator and not a repair of the earlier optimizer-on score.

## What was measured

- Engine/target: CPU QCSim Statevector, FP64, `optimize_circuit=false`, 1,000
  shots on panel circuits. The measured value is the engine-reported execution
  time, not end-to-end Python time and not CUDA-Q/Aer time.
- Panel: 150 unique exact-QASM hashes, grouped by the frozen C44 five-fold
  split (37/28/26/25/34 hashes). Every hash remains assigned.
- Calibration: 112 synthetic cells, widths 2–9; coefficients are fit only on
  those cells. No panel runtime is used to fit coefficients.
- Pilot: 300 terminal calls; the frozen stability/identity gate passed.
- Full ledger: 3,930 unique assigned identities; 3,825 `ok`, 105
  `unsupported`. The latter are seven panel circuits, retained without
  retry or imputation. There are 143 observed panel targets out of 150.
- Calibration outcome: all eight width knots were valid. Predictions do not
  extrapolate outside the measured width grid.

The component prediction is
`2^n * (C1q(n) * normalized_1q + Ccx(n) * normalized_CX) + b(n) + m(n) * shots`.
Here `b` and `m` come from an empirical zero-state sampling fit. They are not a
measured physical launch overhead; this is a CPU target, and the formula is not
a universal quantum-simulator law.

## Same-panel comparison

All four methods were scored on the same 143 observed hashes. The other seven
hashes remain unavailable in the 150-row envelope.

| Method | MAE (s) | R² | Scored / assigned |
| --- | ---: | ---: | ---: |
| Optimizer-off component adaptation | 0.0001037 | 0.9490 | 143 / 150 |
| Nested grouped Ridge | 0.0001336 | 0.8824 | 143 / 150 |
| Outer-train median | 0.0003058 | −0.0426 | 143 / 150 |
| Source-DAG graph, three-seed median | 0.0004238 | −0.0092 | 143 / 150 |

The observed component-minus-Ridge MAE difference is `−0.0000299 s`; its
hash-clustered 95% bootstrap interval is `[−0.0000625, −0.0000016] s`.
This is evidence only for these small circuits and this QCSim CPU context. It
does not establish a general advantage over learned predictors or transfer to
another simulator, engine setting, circuit width, or timing boundary.

## Files and reproduction boundary

- [`resolved_protocol.json`](resolved_protocol.json) — optimizer-off context
  resolved against its parent protocol.
- [`plan_001.json`](plan_001.json), [`preflight_001.json`](preflight_001.json)
  — exact panel, folds, input hashes and static preflight.
- [`pilot_001/`](pilot_001/) and [`full_001/`](full_001/) — raw attempt ledgers,
  pilot gate, target rows, calibration model and run manifests.
- [`evaluation_001/`](evaluation_001/) — five-fold predictions, paired metrics
  and CUDA training environment/lease record.
- [`native_build_provenance.json`](native_build_provenance.json) — toolchain,
  patched source and linked-library hashes, with a noted post-run library-hash
  capture.

The original QCSim checkout/build and the panel's source QASM collection are
outside this candidate. Therefore the included artifacts support audit and
table regeneration, but not a clean-clone timing rerun. See
[`docs/reproduction.md`](../../../../docs/reproduction.md) for that boundary.
