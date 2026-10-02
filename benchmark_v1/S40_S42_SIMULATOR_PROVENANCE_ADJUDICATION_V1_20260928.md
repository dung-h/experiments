# S40--S42 simulator provenance adjudication v1

Date: 2026-09-28.  This decision classifies completed local common-panel
measurements without rewriting their raw ledgers.

## CUDA-Q bridge decision

The fresh CUDA-Q sibling environment did not discover the NVIDIA targets.  The
204-member dense and MPS runs therefore used the immutable, verified CUDA-Q
0.15.1 runtime and imported Qiskit 2.5.2 from the fresh sibling only after
CUDA-Q initialization.  The exact boundary is pinned by
`execution/manifests/cudaq_verified_bridge_v1.json`, and both new runs record
the interpreter path, Qiskit path, package-lock hash, GPU and runner hash.
No historical timing value was reused.

Decision:

1. Treat the dense FP32/FP64 and MPS FP64 ledgers as **new local bridge
   measurements**, not a fresh-environment replication and not a QPU result.
2. They are eligible for per-engine, per-precision/per-quality **local
   descriptive** simulator tables.  They are not ranked against Aer,
   cuTensorNet, another precision, another API clock, or archived-QPU labels.
3. Dense retains all 36 `adapter_error` cells caused by `rccx`; MPS retains 18
   `adapter_error` cells and 21 `quality_failed` cells.  Neither failure type
   receives an imputed timing value.
4. MPS runtime rows are reportable only alongside the fixed FP64/bond-16/
   cutoff-1e-10/`gesvdj` quality contract and its observed fidelity coverage.

This resolves the bridge review condition for C55/C56.  A later attempt to
claim a fresh-environment replication would require a different run and
manifest; it cannot relabel these ledgers.

## cuTensorNet CPU-interference decision

`MEASUREMENT_INTERFERENCE.json` records that a short CPU-only C25 aggregate
overlapped the parent cuTensorNet run.  The raw per-observation ledger has no
wall-clock timestamps, so the affected planner cell(s) cannot be isolated
after the fact.  The sidecar states that the possible effect is transient CPU
contention of planner wall-clock, while CUDA-event contraction clocks do not
include the aggregate command.

Decision:

1. Preserve the 612-cell raw ledger and its nine planner-timeout cells
   unchanged.  Coverage, selected-plan hashes, workspace and terminal statuses
   remain valid evidence.
2. `RUNTIME_EST`, first contraction and warm contraction remain eligible for
   the **paired selected-plan estimate-versus-warm-contraction** diagnostic:
   the plan, precision and workspace match, and their CUDA-event contraction
   clocks are outside the identified CPU aggregate effect.
3. `t_path_search_s` from this run is **interference-flagged descriptive
   telemetry**.  It must not receive a primary numerical summary, a ranking or
   a fitted conclusion in the final scorecard.  `t_network_build_s` is also
   retained as descriptive context, not an estimator target.
4. No rerun is needed for benchmark v1 because path-search wall-clock is not
   the `RUNTIME_EST` comparator.  If a future claim needs a quantitative
   planner-time distribution, it must use a fresh full-run artifact with
   per-cell timestamps and an exclusive CPU/GPU lease; no subset can be
   reconstructed from the existing ledger.

## Final table labels fixed by this decision

| Evidence | Final table class | Permitted numerical relation |
|---|---|---|
| CUDA-Q dense bridge | local descriptive | same engine, API clock and precision only |
| CUDA-Q MPS bridge | local descriptive, quality-qualified | same engine, API clock and fixed quality contract only |
| cuTensorNet `RUNTIME_EST` | local paired native-estimator diagnostic | `RUNTIME_EST` versus matching selected-plan warm scalar contraction |
| cuTensorNet path search/build | descriptive telemetry | no primary predictor/error score |

The corresponding raw and QA hashes are the authoritative provenance.  This
decision does not erase the sidecar or hide any adapter, quality or planner
failure.
