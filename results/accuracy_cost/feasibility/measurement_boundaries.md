> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Measurement boundaries

Predicted execution seconds và time spent estimating là hai đại lượng khác.
Mọi timing dùng seconds, raw integer nanoseconds; tuyệt đối không lấy
`predicted_seconds`/`RUNTIME_EST` làm inference time.

| Boundary | Start | Stop | Accounting |
|---|---|---|---|
| B1 prediction-only | prepared features/tensors resident; model loaded, eval mode | required native outputs materialized, inverse target transform and frozen aggregation complete | no parse/graph/fit/setup; report single-seed component and actual three-seed estimator separately |
| B2 online | declared in-memory QASM/recipe/circuit plus already-available target metadata | final runtime/quality result on host | parse, extraction, required transpilation, normalization, H2D, prediction, aggregation/D2H included |
| B3 cold setup | parent launches a fresh isolated child | ready for first request after imports/load/device init/declared warmup | process startup, snapshot/model/calibration artifact I/O, CUDA and warmup; first-request time is a separate interval |

Cold total = setup + first B2 request only when recorded as disjoint intervals
of the same run. Do not sum medians. Host request latency is primary, synchronized
CUDA-event device time supplemental. For B1 resident GPU inputs, report host
return/aggregation honestly; a device-resident timing is a distinct boundary.
No neural CPU fallback, mixed precision, quantization, torch.compile, layer
checkpoint toggles or architecture changes for faster timings.

## Per-class contracts

- Graph and MLP: same frozen logical input qualification, global feature semantics,
  own saved fold transform. Graph additionally builds node/edge tensors. Query
  routing uses its original held-out fold; **not all five OOF models**. Three
  seeds for that fold plus median are estimator cost; seed runtimes not summed
  from independently measured medians.
- Statistical reference: a saved scalar is the complete estimator; no test
  label/source leakage. Source median needs source identity; report that privilege.
- Raw analytical: B1 receives required compiled circuit/metadata, actually calls
  native computation. A cached raw scalar lookup is not the native estimator.
  QCRE includes conversion to BQSKIT in B2; omit terminal measurements only as
  in frozen code. Qiskit includes its distinct measurement/schedule semantics.
  Hyb and Scholten proxy retain own eligibility/output clock; don't replace
  formula, infer original CLOPS, or silently fix missing snapshots.
- Calibrated analytical: B1/B2 includes raw calculation plus frozen scalar
  affine/log-affine transform with saved fold coefficients and original clamp.
  Timing only slope multiplication is labeled **bridge-only**, not whole method.
- Maestro: independent saved C1q/Ccx/b/m knots, normalization and shot count.
  No synthetic circuits are regenerated, measured or calibrated. All panel
  widths are stored knots; no new interpolation/extrapolation rule.
- Joint MPS: conditioned version includes predicted-family classifier cost,
  never true family labels as a free input. Both runtime and quality head
  outputs and frozen aggregation are included. Multi-rung query batch distinct
  from a single (hash, chi) request. Reuse classifier once per circuit only if
  cache key and amortization are reported; compare uncached route too.
- cuTensorNet: network build, path optimizer plus internal cost calculation,
  and attribute access are three boundaries. RUNTIME_EST availability generally
  requires a plan. Attribute-read latency alone does not isolate internal model
  computation; no public-stage attribution is invented. No contraction in AM1.

## Offline, shared, repeated

Training T_train, synthetic calibration T_cal, one-time setup S, shared-preparation
C_shared and per-request C_i are reported separately. For explicit N requests,
amortized online cost = (S + C_shared + sum C_i)/N. Optional lifecycle amortization
adds measured T_train/T_cal separately; absent offline timers are missing, not 0.
Historical training-loop timers cannot supply pure training compute.

Standalone B2 charges each method's own full required path. For a combined
multi-estimator service, represent preprocessing as dependency stages and charge
each shared stage exactly once; report shared total and marginal costs, not
arbitrary per-method attribution. Cache hits/misses carry keys including circuit,
fold/transform, snapshot/context and representation hashes. No cross-fold
transform reuse. Local artifact I/O is B3 or an explicitly labeled disk-request
boundary; B2 primary starts with an in-memory circuit. Network/QPU submission is
out of scope.
