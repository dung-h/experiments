# cuTensorNet precision-selection proxy

Each pair contracts the same local scalar tensor network once in `complex64` and once in `complex128`. Warm CUDA-event time is measured separately from path search. The `complex128` scalar is a local comparison reference.

| Quantity | Value |
|---|---:|
| Matched precision pairs | 20 |
| Median complex128 / complex64 time | 3.369× |
| Range of time ratios | 2.480–4.685× |
| Maximum absolute scalar difference | 1.989e-08 |
| Maximum relative scalar difference | 1.936e-06 |

This is a local precision trade-off measurement, not a reimplementation of the SGEMM-emulation paper's automatic selector or reported speedups.
