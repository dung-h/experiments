> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Frozen measurement plan

Machine-readable authority: measurement_plan.json, selected_variants.csv,
selected_workloads.csv, written before replay/timing. Source pins checked before
and after. One serial GPU worker, Torch threads=2/interop=1, native threads=1,
fixed prior pilot affinity. Batch one, frozen seeds 42/1234/31415; no fitting.

All candidate raw requests replay before ANY timing; gate per context/method
requires every eligible request to match consumed input and each seed. Three
full-pipeline warmups per request/method; 30 repeats. Rotate Graph/MLP order
by repeat. CUDA sync before/after full B2 and GPU stages; host perf_counter_ns.
Direct total independent of stage sum; instrumented stage sums include only
completed stages, residual dispatcher time not silently allocated to a stage.
Report per-query median/p95, across-query summaries and paired query/group
bootstrap (2000 replicates, fixed analysis seed=20261011, no repeat pseudo-N).
Memory is process/bundle high-water plus stage allocated/reserved peaks, not
isolated incremental model memory. All outliers retained.

8GiB RAM/VRAM; 256MiB output; disk floor3GiB; 30s request; 120s setup;
7200s aggregate Phase2B wall. No budget extension. Per-request timeout logged
with ID/stage; gate failure blocks timing rather than retry or change inputs.
No simulator timings, planning, QPU, training, queues, historical rewrites,
commit/push. Classical states missing: request new scoped amendment if needed,
not automatic reconstruction/refit. No standalone B3 distribution is promised.
