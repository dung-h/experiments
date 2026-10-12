> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 environment

name, uuid, driver_version, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu, clocks.current.sm [MHz], power.draw [W]
NVIDIA GeForce RTX 5070 Ti, GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e, 595.91.07, 1174 MiB, 16303 MiB, 4 %, 45, 2512 MHz, 43.81 W

CPU: Intel i7-14700F. Affinity [8, 9], selected highest advertised-frequency core class. Native/BLAS threads 1; Torch CPU threads 2.

Affinity 8/9 are SMT siblings of physical core4 (P-core, advertised max5.4GHz), not two physical cores. This CPU cap is part of the measured configuration. Neural: Python 3.10.21, Torch 2.7.1+cu128, PyG 2.6.1, NumPy 1.26.4. CUDA smoke: sm120, allocation and matmul PASS. Neural computations are FP32 even for the FP64-simulator target. CPU analytical environment is recorded in worker receipts. No installations/upgrades.

One benchmark worker at a time. Desktop/RustDesk GPU activity is residual background load, not proven exclusive whole-device ownership. Native timing shares this live workstation. OS page cache is warm/uncontrolled. No cache flush, governor/clock changes, AMP or compilation.

Detailed packages, PID, affinity, CUDA build and absolute memory peaks are saved per worker. Timer: perf_counter_ns; CUDA synchronized before/after each neural interval. Controls retain the same timing wrapper. Scalar references in GPU bundles include synchronization-wrapper overhead and cannot establish sub-microsecond intrinsic scalar latency.
