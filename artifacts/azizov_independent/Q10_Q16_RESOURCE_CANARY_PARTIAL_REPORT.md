# Width-10--16 resource canary (partial)

Finding date: 2026-09-18.

This is a bounded resource screen after the successful q<=9 matrix. The source
manifest contains 41 circuits with logical width 10--16. The intended canary
was one `optimization_level=0` run on both current fake backends, with 1,024
shots, a 16 GiB per-process-tree RSS safety cap and a 180-second wall cap.
Only the first seven `FakeWashingtonV2` rows were completed before the probe
was stopped; the raw rows are therefore explicitly labelled partial.

| Circuit | Width | Status | Peak RSS | Wall time | Execution time |
|---|---:|---|---:|---:|---:|
| `ae_q10` | 10 | RSS cap | 16.42 GiB | 6.5 s | censored |
| `ae_q12` | 12 | RSS cap | 16.42 GiB | 5.6 s | censored |
| `dj_q10` | 10 | RSS cap | 16.42 GiB | 6.6 s | censored |
| `dj_q11` | 11 | RSS cap | 16.41 GiB | 5.6 s | censored |
| `dj_q16` | 16 | ok | 0.42 GiB | 4.1 s | 0.681 s |
| `ghz_q10` | 10 | ok | 0.96 GiB | 69.8 s | 66.367 s |
| `ghz_q11` | 11 | ok | 1.05 GiB | 71.5 s | 67.936 s |

The immediate conclusion is that logical width alone is not a capacity rule.
At the same nominal width, `dj_q10` exceeded the safety cap while `ghz_q10`
completed, and `dj_q16` was cheap because its circuit structure is sparse.
The cap events occurred during the default Aer parallel execution path, so
they are evidence of this simulator/backend configuration, not a theorem that
the circuits cannot run with another Aer method, thread policy or machine.

The probe is not a training matrix and must not be pooled with the 1,192-row
q<=9 result. It is retained to justify the next experiment design: compare
Aer thread/shot parallelism under the same circuit/backend pair, then expand
only the rows that pass a resource and timeout screen. The process-isolated
runner and `/proc` sampler are in
`tracks/azizov_independent/scripts/run_resource_canary.py`.
