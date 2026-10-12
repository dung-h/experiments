> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Environment

{
  "compute_inventory": "46342, /usr/share/rustdesk/rustdesk, 284 MiB\n",
  "contexts": {
    "fp32": {
      "ids": 30,
      "status": "PASS"
    },
    "fp64": {
      "ids": 30,
      "status": "PASS"
    }
  },
  "cpu_environment": {
    "packages": {
      "numpy": "2.2.6",
      "qiskit": "2.5.2"
    },
    "python": "3.10.21"
  },
  "cpu_info": "\t: Intel(R) Core(TM) i7-14700F",
  "disk_free_bytes": 52454842368,
  "gpu_environment": {
    "packages": {
      "numpy": "1.26.4",
      "qiskit": "0.44.2",
      "torch": "2.7.1+cu128",
      "torch-geometric": "2.6.1"
    },
    "python": "3.10.21"
  },
  "gpu_inventory": "name, uuid, driver_version, memory.total [MiB], memory.used [MiB], utilization.gpu [%]\nNVIDIA GeForce RTX 5070 Ti, GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e, 595.91.07, 16303 MiB, 1014 MiB, 3 %\n",
  "meminfo": "MemTotal:       48085772 kB\nMemFree:        36850968 kB\nMemAvailable:   43439260 kB\nBuffers:          158784 kB\nCached:          6895196 kB\nSwapCached:            0 kB\nActive:          5267152 kB\nInactive:        4380224 kB\nActive(anon):    2691340 kB\nInactive(anon):        0 kB\nActive(file):    2575812 kB\nInactive(file):  4380224 kB\nUnevictable:         216 kB\nMlocked:             216 kB\nSwapTotal:       8388604 kB\nSwapFree:        8388604 kB\nZswap:                 0 kB\nZswapped:              0 kB\nDirty:              3732 kB\nWriteback:             0 kB\nAnonPages:       2593980 kB\nMapped:          1386800 kB\nShmem:             98228 kB\nKReclaimable:     223332 kB\nSlab:             479124 kB\nSReclaimable:     171636 kB\nSUnreclaim:       307488 kB\nKernelStack:       25552 kB\nPageTables:        53612 kB\nSecPageTables:      3696 kB\nNFS_Unstable:          0 kB\nBounce:                0 kB\nWritebackTmp:          0 kB\nCommitLimit:    32431488 kB\nCommitted_AS:   14988900 kB\nVmallocTotal:   34359738367 kB\nVmallocUsed:      140404 kB\nVmallocChunk:          0 kB\nPercpu:            33040 kB\nHardwareCorrupted:     0 kB\nAnonHugePages:         0 kB\nShmemHugePages:        0 kB\nShmemPmdMapped:        0 kB\nFileHugePages:    100352 kB\nFilePmdMapped:      2048 kB\nCmaTotal:              0 kB\nCmaFree:               0 kB\nUnaccepted:            0 kB\nBalloon:               0 kB\nGPUActive:             0 kB\nGPUReclaim:            0 kB\nHugePages_Total:       0\nHugePages_Free:        0\nHugePages_Rsvd:        0\nHugePages_Surp:        0\nHugepagesize:       2048 kB\nHugetlb:               0 kB\nDirectMap4k:     1037232 kB\nDirectMap2M:    11366400 kB\nDirectMap1G:    39845888 kB\n",
  "pins": 1569,
  "status": "PASS"
}

One CUDA coordinator+one CPU preprocessing child. Affinity8/9, native threads1,Torch threads2/interop1. Qiskit2.5.2/NumPy2.2.6 provide verified compatibility, not proof of unknown original producer package versions. Torch2.7.1+cu128/PyG2.6.1/NumPy1.26.4 consumer. RustDesk desktop activity remains disclosed; no other benchmark workload. JSON/base64 stdio implementation-specific cost. No feature cache. Imports/startup/model loading/input disk reads excluded; setup recorded separately.
