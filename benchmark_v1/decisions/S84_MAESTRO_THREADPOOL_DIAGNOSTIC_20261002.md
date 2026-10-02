# S84 — Maestro native-thread-policy diagnostic

**Date:** 2026-10-02  
**Status:** Protocol only; execution awaits independent runner review.  
**Scope:** One cause-discriminating diagnostic for the unstable statevector pilot.

## Decision

A paired thread-policy intervention is scientifically admissible as a narrow
diagnostic. The recorded pilot found all nine q16 statevector cells unstable,
while a separate MPS cell passed. The actual worker fingerprint showed both
GNU OpenMP (`libgomp`) and SciPy OpenBLAS configured for 28 threads, with one
timing worker and affinity to 28 logical CPUs. That makes native thread policy
a plausible candidate context factor, but it does **not** establish
oversubscription or identify which pool caused the spread.

The experiment can test whether changing the observed native-pool policy
changes the statevector timing stability. It cannot establish Maestro model
accuracy, recover a calibration curve, or promote the method. S83 and all
existing pilot artifacts remain unchanged.

## Frozen comparison

Use only the nine q16 statevector cells that failed the S83 pilot gate. Each is
the same prepared QASM and configuration in both arms: seed `12345`, one shot,
QCSim statevector, process-isolated call, and the existing pinned Maestro,
NumPy, SciPy, Qiskit, and runner binaries.

| Operation | Repeats | Existing cell ID |
| --- | ---: | --- |
| one_qubit_noncommuting | 256 | `35d04914103e48a1e741ec2a608c7dd3c4e1fa811dc3f3fa41bf07fa86b31bc6` |
| one_qubit_noncommuting | 512 | `d69b0d8ec6c08da40f8aabbd898064001b538c9572998a3c39a0893e299124a9` |
| one_qubit_noncommuting | 1024 | `ea5c38a39f65bca076245b349fe4a06a3ae264f6a6734fcf9041e1236ff9072f` |
| two_qubit_wrapper_control | 256 | `cf265906872b55f394d45c6cb56c013bd3f3e4dd775e9d6e23e496b75c32e06a` |
| two_qubit_wrapper_control | 512 | `6f72abf103d1a994f7ca0a0a4ff63139e19e8679cafbcae30fa14382e66115a0` |
| two_qubit_wrapper_control | 1024 | `ae05f4262cc29de97a8bac964bc5e9eb7f03ac11cd96bf1503cd168b254cc24a` |
| two_qubit_cx_interleaved | 256 | `f8d2f3cb7e573cdc305fd343f1513dd3f5ef0bfa6d69e73b25ab988061430526` |
| two_qubit_cx_interleaved | 512 | `16c0a6ad5539f3051d685aa89945dab16ee116dcd4a4f7f7feb128c894534dc0` |
| two_qubit_cx_interleaved | 1024 | `7286cf5792fb186c880edf1a928c25a299cdf75a3446fbb2081082d9c2459a77` |

The stable MPS row is excluded. The raw pilot row is q12/chi32, while the
v3 contract describes an intended q16/chi32 pilot control. Do not silently
substitute that row or claim that this diagnostic covers the intended MPS
control; the discrepancy remains an inventory item.

### Arm A — pinned default

Keep `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` unset, matching the recorded pilot environment. Before
any timed call, verify the actual worker reports `libgomp=28` and
`libscipy_openblas=28`.

### Arm B — native pools limited to one

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in the spawned worker
environment **before importing** Maestro, NumPy, or SciPy. Keep
`MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` unset in both arms, as those pools
were not observed in the pinned worker. Before any timed call, verify the
actual worker reports `libgomp=1` and `libscipy_openblas=1`.

The intervention is the native-pool policy as a whole. It does not distinguish
OpenMP from OpenBLAS effects. All other worker identity fields, including
affinity `0..27`, binary/library hashes, resolved simulator config, seed, QASM
hashes, and process-isolated call behavior, must match. If the thread limits do
not take effect or any other identity field changes, stop before timing.

For the cross-arm identity check only, `native_threadpools` is compared as an
order-insensitive multiset because the inspection library does not guarantee
enumeration order. The runner copies each pool record, removes only
`num_threads`, and sorts by canonical JSON over every remaining field. It
preserves duplicate entries and all library-identifying fields, including
`filepath`, nested library path/hash, API names, prefix, version, architecture,
and threading layer. Any difference in those fields remains a preflight
failure. The complete original per-arm inventories, including order and thread
counts, remain in the recorded worker contexts; normalization is not applied
to those stored observations.

## Schedule and measurements

- Three sessions, five timed repetitions per cell per arm: `9 × 3 × 5 × 2 = 270` timed API calls.
- Pair arms within each cell/session/repetition; randomize A-before-B versus B-before-A using schedule seed `20261002`. Freeze and hash the complete order before the first call.
- One timing worker; no concurrent CPU-heavy or GPU workload. Each call remains a fresh spawned process; no persistent-process warm claim.
- No extra untimed API calls. The selected clock is `reported_time_seconds`; retain `host_wall_seconds` and `outer_wall_seconds` for every attempt as diagnostics.
- Retain every scheduled result, error, and timeout. No retries, replacement observations, or favorable reruns. Each worker call has a maximum timeout of 180 seconds. The 3600-second operational no-new-worker budget begins immediately after the unique output directory is reserved and covers initial/recurring host samples, both no-API worker-context preflights, timed calls, worker cleanup, and terminal persistence. It is not a hard real-time wall cap: OS scheduling, an unreapable worker, or filesystem/fsync stalls can cause an observed overrun. Before launching any worker, reserve the final 15 seconds for nominal timeout cleanup and durable ledger writes; a worker receives at most `min(180 s, remaining − 15 s)`, and no worker starts when remaining budget is 15 seconds or less. Persist budget start/completion timestamps, elapsed time, and observed overrun in the run manifest. A budget stop before any timed call is recorded as `blocked_preflight`; a later stop is explicitly partial.
- Write new outputs under `artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002/`. Do not resume or write into the S83 pilot directory.

### Narrow append-only continuation for `attempt_003`

The first attempt at this protocol is pinned to its original protocol SHA-256
`35b2e7a36148c80ee13b1095f9f2e474a2c0bfb925745974e9d75daf3801e0ae` and
runner SHA-256
`e0b52ef864c3621707432790c90e4a94f678d3615294b6527dce6999a2f0cf02`. Its
append-only ledger currently contains two successful API calls followed by a
third scheduled observation blocked by the host-activity gate before the API.
A `--resume` path may continue only the exact
`artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002/attempt_003`
artifact after fail-closed validation of those historical pins, the current
protocol's unchanged deterministic schedule, manifest and ledger hashes,
row-count equality, and every scheduled descriptor/pair/order in prefix
`1..attempt_count`. The only resumable terminal state is `partial` with a
ledger whose rows are `ok`/API-called or `blocked_host_activity`/not-called,
whose first three rows are `ok, ok, blocked_host_activity`, and whose last row
is the not-called host block. Every earlier successful or blocked row is
preserved; the next operation is scheduled index `attempt_count + 1`, never a
retry or replacement.

Before continuing, the runner rechecks the host gate and both worker contexts.
Resume provenance (new protocol/runner/input hashes, previous ledger hash,
starting row, and original budget origin) is appended to `resume_invocations`;
the historical top-level protocol/runner pins and prior rows are retained. The
3600 s budget continues from the original `wall_budget_started_unix`; resume
does not reset it. A resume-only host/context preflight block is retryable only
when the ledger remains byte-for-byte unchanged, `attempt_count` is unchanged,
and the invocation records zero appended rows and `api_called=false`; each
retry is retained in invocation history and uses the same cumulative budget.
This exception does not apply to an original `blocked_preflight` run, which has
no eligible scheduled-row checkpoint. A timeout, API/worker error, incomplete
`running`/I/O state, or budget/reserve stop is terminal and not resumable. A
clean host block on a newly appended scheduled row may be continued again
under the same rules. This continuation contract does not authorize a run; the
exact `--resume --execute` invocation remains subject to independent review.

The runner must verify both arm fingerprints before it calls Maestro. A fingerprint,
QASM, config, software, or affinity mismatch blocks the run. Transport or API
errors are retained as outcomes; a worker that cannot be reaped or a detected
overlapping workload stops the run. Do not discard the attempts already recorded.

### Host-activity gate and approved idle desktop baseline

The timing host is a remote desktop, so “no compute processes at all” is not an
appropriate idle definition. The approved baseline is frozen by process
identity, not by process-name allowlisting. It was observed on 2026-10-02:

| NVIDIA process type | PID | UID | /proc start ticks | Resolved executable | Executable SHA-256 | pmon command | Baseline framebuffer memory |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| G | 42623 | 1000 | 37701 | /usr/bin/gnome-shell | cf699357b6b6bbbacda5250261200ab92f6cccd491b126c6e30142c9fdafd619 | gnome-shell | 178 MiB |
| G | 43388 | 1000 | 37797 | /usr/bin/Xwayland | 700d3cba7f946c3ff3100ee36f15012c1c027bb67f29e84def1a1b67bb8f3587 | Xwayland | 70 MiB |
| C+G | 181789 | 1000 | 190789 | /usr/bin/ptyxis | d896539ba35478c4fc307be5f0fc2f3b2c6997114b90189810233e450d11d1f3 | ptyxis | 41 MiB |
| G | 1169761 | 1000 | 6291646 | /snap/firefox/8969/usr/lib/firefox/firefox | 5e800815dacd4c2a80e17f69dcd46e4cba2c82871ce0867efc8e4a8eabcd1efe | firefox | 134 MiB |
| C+G | 2925123 | 1000 | 3406758 | /usr/share/rustdesk/rustdesk | 58ef1e984727d827836c8ad84ad50a4971db4a3cb80ad7a646982594af155c52 | rustdesk | 284 MiB |
| G | 3197852 | 1000 | 3748766 | /usr/share/rustdesk/rustdesk | 58ef1e984727d827836c8ad84ad50a4971db4a3cb80ad7a646982594af155c52 | rustdesk | 39 MiB |

The associated GPU baseline is UUID
GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e, NVIDIA GeForce RTX 5070 Ti,
2% utilization, 797 MiB device memory in use, and 746 MiB summed framebuffer
memory across the six pmon rows. The runner uses NVIDIA's per-process monitor
table (nvidia-smi pmon -c 1 -s um), whose type column reports compute (C),
graphics (G), or both (C+G); the compute-only application query is not used as
the complete process inventory. It records the full baseline snapshot in the
run manifest and a fresh C/G plus device-resource snapshot before each
scheduled attempt. A changed PID, UID, process start tick, executable
path/hash, C/G type, command, missing baseline process, or additional GPU
process blocks the run. A host reboot, desktop restart, driver/device change,
or binary update invalidates this baseline; it must be reviewed and frozen in
a new decision, not silently re-learned at run start.

The gate allows small desktop-rendering variation while rejecting added or
heavy work: GPU utilization must remain at or below 10%; device memory must
remain at or below 925 MiB (baseline +128 MiB); each baseline process may
increase by at most 64 MiB; their combined framebuffer memory must remain at
or below 874 MiB (baseline +128 MiB).

For CPU, the runner brackets a requested 250 ms sample with readings from
/proc/stat and each readable /proc/<pid>/stat. It records the actual monotonic
interval, system busy percentage, per-PID user+system tick deltas, one-core-
equivalent CPU percentages, process-table counts, and PID churn. Processes
born within the interval are recorded using their accumulated ticks as a
lower-bound sample. CPU is busy if any non-runner PID reaches 100% of one
core, all non-runner deltas sum to at least one core, or one-minute load
reaches 28 (the pinned affinity size). Unmatched PID entries are retained as
unscored; the system counter remains an aggregate cross-check. CPU or GPU
telemetry unavailable is fail-closed.

The 3600 s operational no-new-worker budget starts at output-directory
reservation, not after preflight. It covers the initial and per-attempt
host-activity samples, both worker-context preflights, timed calls, nominal
worker cleanup, and terminal writes; pinned-input verification and schedule
construction happen before reservation and are outside that budget. It is not
a hard real-time wall cap. An OS scheduling stall, an unreapable worker, or a
filesystem/fsync stall can delay cleanup or persistence and cause elapsed
process wall time to exceed 3600 s. The manifest records budget start and
completion timestamps, monotonic elapsed seconds, and observed overrun seconds
so that this is visible. Elapsed time is sampled after the first durable
terminal-manifest write; the final metadata rewrite is not self-timed.

Each worker timeout is at most `min(180 s, remaining − 15 s)`. The 15 s reserve
is nominal headroom for the terminate/join/kill/join path plus durable attempt
and manifest writes; it reduces but cannot eliminate overrun risk. No worker is
launched when remaining budget is 15 s or less. If the reserve is reached
during the initial host sample, before/during either preflight, or after
preflight but before timing, the runner durably records `blocked_preflight` and
makes no timing/API call. After timed execution begins, the runner recalculates
remaining budget after every host sample and immediately before worker launch;
if the reserve is reached, it writes a no-API terminal row for that scheduled
attempt and stops with a partial run. Every observed deviation is stored; any
deviation before an attempt blocks it and stops the run. These are operational
isolation thresholds, not claims that the desktop performs no work; interval
samples cannot rule out brief activity between observations.

## Acceptance and interpretation

Apply the unchanged S5 engineering stability gates independently to each cell
and arm:

1. In every session, five timed results must be present and
   `MAD <= max(0.00005 s, 0.20 × session median)`.
2. Across the three session medians,
   `range <= max(0.00005 s, 0.20 × median of session medians)`.

These are the existing prospective QA thresholds, not significance tests.
Publish per-cell/per-session medians, MADs, ranges, paired timing differences,
and all failures. Do not fit component coefficients or evaluate a predictor.

Classify the diagnostic as **consistent with thread-policy sensitivity** only
if the concurrent default arm reproduces the S83 instability on all nine cells
and the single-thread arm passes both stability gates on all nine cells, with
both worker fingerprints and all integrity checks passing. Any other outcome
is `inconclusive_or_partial`; describe the cell-level results without claiming
that thread policy explains the original spread. Even the full pass establishes
only sensitivity to the combined native-pool limit, not a unique causal pool
or a valid Maestro predictor.

No full calibration grid, panel run, coefficient fit, or leaderboard promotion
follows automatically. Any follow-up needs a separate review and protocol.

## Execution gate

The existing `run_maestro_component_calibration.py` command does not implement
randomized paired thread-policy arms and must not be used for this intervention.
No executable command is approved by this protocol. A dedicated runner must
first implement the schedule, pre-import environment assignment, actual-worker
fingerprint checks, immutable attempt ledger, and stop behavior above. The
coordinator must independently inspect that runner and the exact invocation
before any Maestro API call. The user's conditional request is recorded as the
motivation for preparing this protocol; this document alone does not start a
run.

## Evidence pins reviewed

| Evidence path | SHA-256 |
| --- | --- |
| `benchmark_v1/decisions/S83_FAMILY_AWARE_AND_MAESTRO_ADJUDICATION_20261002.md` | `58e2af1c879ad59dfda3c9ead99c6f14deb165eb0eb4f1afc77a1d25b553597d` |
| `benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json` | `bc2a3bc150421698ff09e08a2055a40979f96d2965fe2dcb3e20402c1016574d` |
| `artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/run_manifest.json` | `8989735f4291621b2ee18c04ae7c326dd029183661bb510d1f0acc74147d92c2` |
| `artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/raw_records.csv` | `9122f3b28047e52f33654eb7f83296255ff13b1f2ab285d44d1970cb27610709` |
| `artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/acceptance.json` | `90bbc6b6f819d316e7fd4b40c77668856337de31d05a7d97eb4c4d8371019b44` |
| `artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/maestro_pilot_summary.csv` | `d6107c87721695537cee9c67f3e7c51ab916f620a665b532487d1374e7d02ad5` |
