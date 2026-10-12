> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Resources

Replay FP32/FP64 took 4.007/3.875 s. Timing executions, including setup,
warmups, resource guards and controls, took 26.883/23.041 s. Combined execution
was 57.805 s, below the 7,200 s aggregate cap. This excludes time spent
authoring and validating the reporting tooling.

Peak sampled aggregate RSS was 1,355,509,760 bytes (1.262 GiB). Host high-water
was 1,239,949,312 bytes. The maximum child-reported high-water in raw requests
was 789,917,696 bytes, which can include fork/startup memory inherited before
exec. Their conservative sum is 2,029,867,008 bytes (1.891 GiB), not a
simultaneous observed peak. Both bounds are below 8 GiB.

CUDA allocated peak was 68,263,936 bytes and reserved peak 79,691,776 bytes
(76 MiB reserved). These counters include loaded models/setup and exclude
desktop GPU usage. No Phase5 workers remain.

Timer-pair medians were 57/54 ns for FP32/FP64. Empty-stdio round-trip medians
were 8.540/8.541 microseconds. No control overhead was subtracted from B2.
Empty echo does not measure serialization or transmission of Graph payloads.

{
  "replay_fp32": {
    "child_exit_code": 0,
    "child_stderr": "",
    "gpu_peak_allocated_bytes": 68263936,
    "gpu_peak_reserved_bytes": 79691776,
    "host_peak_rss_bytes": 1232379904,
    "peak_aggregate_rss_bytes": 1335955456,
    "peak_child_rss_bytes": 102416384,
    "peak_host_rss_bytes": 1233547264,
    "subprocess_reaped": true,
    "violation": "",
    "wall_seconds": 4.006685315999675
  },
  "replay_fp64": {
    "child_exit_code": 0,
    "child_stderr": "",
    "gpu_peak_allocated_bytes": 68263936,
    "gpu_peak_reserved_bytes": 79691776,
    "host_peak_rss_bytes": 1232535552,
    "peak_aggregate_rss_bytes": 1335169024,
    "peak_child_rss_bytes": 101736448,
    "peak_host_rss_bytes": 1233436672,
    "subprocess_reaped": true,
    "violation": "",
    "wall_seconds": 3.8746340730003794
  },
  "timing_fp32": {
    "child_exit_code": 0,
    "child_stderr": "",
    "gpu_peak_allocated_bytes": 68263936,
    "gpu_peak_reserved_bytes": 79691776,
    "host_peak_rss_bytes": 1239617536,
    "peak_aggregate_rss_bytes": 1355509760,
    "peak_child_rss_bytes": 115363840,
    "peak_host_rss_bytes": 1240322048,
    "subprocess_reaped": true,
    "violation": "",
    "wall_seconds": 26.882907651000096
  },
  "timing_fp64": {
    "child_exit_code": 0,
    "child_stderr": "",
    "gpu_peak_allocated_bytes": 68263936,
    "gpu_peak_reserved_bytes": 79691776,
    "host_peak_rss_bytes": 1239949312,
    "peak_aggregate_rss_bytes": 1352925184,
    "peak_child_rss_bytes": 112832512,
    "peak_host_rss_bytes": 1240432640,
    "subprocess_reaped": true,
    "violation": "",
    "wall_seconds": 23.04073934200005
  }
}

All four replay/timing processes sequential. Aggregate RSS sampled50ms, per-process high-water retained; peaks may not coincide. CUDA allocated/reserved include setup but exclude desktop board memory. RAM refers to physical RSS, not virtual address space. Caps:8GiB aggregate RSS/VRAM,7,200s combined execution,256MiB outputs,3GiB diskfloor,30s request,120s setup. Startup not in B2. Timer-pair/empty-echo controls retained without subtraction; instrumentation and monitor overhead remain included. Empty echo is not real-payload IPC estimate. Children reaped.
