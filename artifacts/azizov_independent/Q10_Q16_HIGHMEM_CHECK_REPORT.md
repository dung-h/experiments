# High-memory cap check for the width-10--16 frontier

Finding date: 2026-09-18.

To test whether the earlier 16 GiB safety cap was the limiting factor, two
amplitude-estimation rows were rerun in isolated processes with the cap raised
to 36 GiB. The host has 45 GiB total RAM; 36 GiB is the largest cap used here
that still leaves practical headroom for the operating system and background
services. The wall limit remained 180 seconds, with 1,024 shots and Aer's
default parallel settings.

| Circuit | Peak RSS | Wall result | Interpretation |
|---|---:|---|---|
| `ae_q10` on FakeWashingtonV2 | 16.42 GiB | timeout at 180 s | More cap did not complete the row |
| `ae_q12` on FakeWashingtonV2 | 16.42 GiB | timeout at 180 s | More cap did not complete the row |

Both processes plateaued around 16.8 GiB rather than approaching the 36 GiB
cap. Therefore the immediate bottleneck is Aer execution throughput or circuit
simulation complexity, not simply the 16 GiB guard. Raising the cap to 40 GiB
would leave roughly 5 GiB for the rest of a 45 GiB host and would not remove
the observed 180-second wall-time failure; it would mainly increase OOM/swap
risk. These rows remain censored and are not added to the q<=9 training table.

The raw rows are in `q10_q16_highmem_canary.csv`. The earlier 16 GiB canary
remains the primary resource-boundary report; this file is a targeted
high-memory sensitivity check.
