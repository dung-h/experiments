> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Proposed fair-comparison protocol — approval required

## Populations and deterministic sampling

Track A retains the 4,515 qualified observations/165 conservative groups and
frozen outer-test assignment. Track B retains each context's exact hash/rung
population and target/quality eligibility. Timing selection never uses target
runtime, residual, winning model, best seed or availability-conditioned accuracy.

Proposed screening panel: at most **50 QPU observations (10 per frozen outer
fold)** and **30 circuit hashes per executable simulator context**. Freeze a
selection manifest before timing; use the same label-free selection within
each QPU fold, then union without exceeding the per-fold cap.
Using frozen metadata only, form strata by source/context, family, backend,
width bins (q2–4/5–7/8–9/10–16/>16 where present), within-context depth/gate/graph
node tertiles. These are not full Cartesian quotas. Deterministic greedy
coverage selects the SHA256-smallest candidate for each uncovered nonempty
stratum, cycling axes in this order; deduplicate IDs. If mandatory coverage
exceeds cap, require approval or list uncovered strata, never claim full
representativeness. Fill to cap by SHA256 of `AM1-ESTIMATOR-OVERHEAD|track|context|ID`.
Exact repeated model-ready inputs retain observation IDs; record both unique
structural/feature-input timing and observation-weighted service interpretation.

Previously identified QNN/Random or large/collision groups can have a separately
declared diagnostic panel selected by metadata/family, not extreme error.
Do not merge these enriched strata into unbiased population estimates. No
source-held-out retraining or new model generation. Joint MPS retains all six
assigned chi rungs for a selected hash and original quality policy. Do not
evaluate quality by new state simulation or invent labels for missing rungs.

All methods attempt the same selected requests where contracts permit; keep
unavailable/failure rows in denominator. Report full eligibility and exact
shared-success intersections, with input requirements disclosed. No single
cross-context Track B ranking. Preserve raw versus calibrated output semantics.
Classical models missing fitted states remain blocked, not presumed cheap.

## Hardware/software record and isolation

Observed inventory on 2026-10-11 (read-only, not a reservation): Intel i7-14700F,
20 cores/28 logical CPUs; RAM 49,239,826,432 bytes (~45.86 GiB); RTX 5070 Ti,
16,303 MiB VRAM, driver 595.91.07; Linux 7.0.0-38-generic x86_64/glibc2.43.
At inspection ~40.1 GiB RAM available, ~49.05 GiB disk available, GPU ~996 MiB
used/4% utilization. These volatile readings do not prove an idle/exclusive host.

CUDA environment exists at
`/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python`:
Torch 2.7.1+cu128, PyG 2.6.1, NumPy 1.26.4, sklearn 1.3.0, Qiskit 0.44.2.
CPU metadata environment `/home/server/Documents/.venv-qonductor/bin/python`:
Qiskit 2.5.2, runtime 0.49.0, NumPy 2.2.6, sklearn 1.7.2, Torch 2.14.0+cpu.
Use the first for neural, never the CPU Torch installation. QCRE's pinned
BQSKIT 1.2.0/adapter lives in `.venv-cdaa-new/lib/python3.10/site-packages`,
appended as in the producing route. B2 may require a clearly timed CPU-feature
worker to CUDA-worker IPC: include serialization/transfer, never conceal it.
No single merged environment is assumed. Metadata availability is not an import
or ABI test; future smoke/parity gate must establish compatibility without refit.

cuTensorNet's current dedicated environment is Python3.14.4/Qiskit2.5.2/
NumPy2.5.3, cuquantum-python-cu13 26.6.0, cupy-cuda13x 14.2.0,
cutensornet-cu13 2.14.0 and cutensor-cu13 2.8.1. Metadata proves installed
distributions, not import/ABI viability or historical environment parity.
The missing historical selected-plan state remains a separate blocker.
No dependency installation/update in this task.

At approved execution: record OS/kernel, CPU affinity/P–E core class, SMT,
CPU governor/frequency, thread settings, host/GPU load, GPU UUID, clocks/power,
driver/runtime, package versions and checkpoint placement/precision. Primary
CPU native/BLAS threads=1; neural Torch CPU threads=2, CUDA exclusive one worker.
Pin to a documented physical core class, not arbitrary CPU0 on this hybrid CPU.
No heavy CPU/GPU workload in parallel; no changing governor/clocks silently.
One measurement worker, one independent resource observer. Report residual
background load. Compare complete deployable CPU/GPU routes without claiming
the difference is caused only by architecture.

## Measurements (not executed)

- For each warmed fold/model bundle: 10 unscored warmups, then 30 repetitions
  of each selected request. Fresh identical input objects per uncached B2
  request; existing valid cache for explicitly cached B1. Eval/no_grad and
  frozen inverse transform/aggregation; no compile/autotuning added.
- Three isolated fresh processes per bundle for B3; OS page cache remains
  uncontrolled/warm and is disclosed. No sudo/drop_caches. Capture launch,
  import, load, CUDA init, warmup and first request separately.
- Primary batch=1; optional exact batch32 if 32 distinct eligible requests
  exist within the same context/fold. Never mix folds to fill a batch. Joint
  MPS six-rung request is its own batch. If <32, omit, don't pad silently.
- Paired rotating method order fixed by deterministic hash order and repetition
  index, no best-latency run. Monotonic perf_counter_ns; synchronize CUDA before
  start and after completion including host result, then stop. GPU events are
  supplemental device-only timers, not host pipeline latency.
- Record median, p90, all raw samples and failure counts. p99 with only30
  repeats is descriptive/weak, not an SLA. Across-request quantiles are distinct
  from within-request jitter. Structural-group bootstrap paired median/MAE
  differences; repeats do not increase independent workload support. <10 groups:
  descriptive uncertainty only, following existing support rule.
- Empty timer pair, no-op dispatch and sync-only controls measured identically.
  Retain unadjusted primary latency, report control distribution; do not subtract
  noisy medians to manufacture negative/near-zero cost. If timer/control floor
  dominates, report unresolved; any loop amplification is separately approved
  and not batch latency. No selective outlier removal.
- Isolated-process RSS high-water + sampled child memory; Torch allocated/
  reserved peaks and GPU process memory separately. Observer interval/baseline
  recorded; sampled peaks are lower bounds. Profiled-memory run separated from
  primary low-overhead latency if instrumentation perturbs timings.
- Timeout/OOM/nonfinite output/parity failure retained. Environment interruption
  receives reason, not deletion or automatic retry. Require approval before
  changing repetition count, workload, device, budget or method implementation.
