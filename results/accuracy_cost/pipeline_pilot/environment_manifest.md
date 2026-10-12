> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Environment before execution

{
  "executable": "/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python",
  "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.43",
  "affinity": [
    8,
    9
  ],
  "cpu_info": {
    "lscpu": [
      {
        "data": "x86_64",
        "field": "Architecture:"
      },
      {
        "data": "32-bit, 64-bit",
        "field": "CPU op-mode(s):"
      },
      {
        "data": "39 bits physical, 48 bits virtual",
        "field": "Address sizes:"
      },
      {
        "data": "Little Endian",
        "field": "Byte Order:"
      },
      {
        "data": "28",
        "field": "CPU(s):"
      },
      {
        "data": "0-27",
        "field": "On-line CPU(s) list:"
      },
      {
        "data": "GenuineIntel",
        "field": "Vendor ID:"
      },
      {
        "data": "Intel(R) Core(TM) i7-14700F",
        "field": "Model name:"
      },
      {
        "data": "6",
        "field": "CPU family:"
      },
      {
        "data": "183",
        "field": "Model:"
      },
      {
        "data": "2",
        "field": "Thread(s) per core:"
      },
      {
        "data": "20",
        "field": "Core(s) per socket:"
      },
      {
        "data": "1",
        "field": "Socket(s):"
      },
      {
        "data": "1",
        "field": "Stepping:"
      },
      {
        "data": "31%",
        "field": "CPU(s) scaling MHz:"
      },
      {
        "data": "5400.0000",
        "field": "CPU max MHz:"
      },
      {
        "data": "800.0000",
        "field": "CPU min MHz:"
      },
      {
        "data": "4224.00",
        "field": "BogoMIPS:"
      },
      {
        "data": "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize arch_lbr ibt flush_l1d arch_capabilities",
        "field": "Flags:"
      },
      {
        "data": "VT-x",
        "field": "Virtualization:"
      },
      {
        "data": "768 KiB (20 instances)",
        "field": "L1d cache:"
      },
      {
        "data": "1 MiB (20 instances)",
        "field": "L1i cache:"
      },
      {
        "data": "28 MiB (11 instances)",
        "field": "L2 cache:"
      },
      {
        "data": "33 MiB (1 instance)",
        "field": "L3 cache:"
      },
      {
        "data": "1",
        "field": "NUMA node(s):"
      },
      {
        "data": "0-27",
        "field": "NUMA node0 CPU(s):"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Gather data sampling:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Ghostwrite:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Indirect target selection:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Itlb multihit:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability L1tf:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Mds:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Meltdown:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Mmio stale data:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Old microcode:"
      },
      {
        "data": "Mitigation; Clear Register File",
        "field": "Vulnerability Reg file data sampling:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Retbleed:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Spec rstack overflow:"
      },
      {
        "data": "Mitigation; Speculative Store Bypass disabled via prctl",
        "field": "Vulnerability Spec store bypass:"
      },
      {
        "data": "Mitigation; usercopy/swapgs barriers and __user pointer sanitization",
        "field": "Vulnerability Spectre v1:"
      },
      {
        "data": "Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S",
        "field": "Vulnerability Spectre v2:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Srbds:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Tsa:"
      },
      {
        "data": "Not affected",
        "field": "Vulnerability Tsx async abort:"
      },
      {
        "data": "Mitigation; IBPB before exit to userspace",
        "field": "Vulnerability Vmscape:"
      }
    ]
  },
  "inventory": "name, uuid, driver_version, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu\nNVIDIA GeForce RTX 5070 Ti, GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e, 595.91.07, 640 MiB, 16303 MiB, 28 %, 40\n",
  "disk_free_bytes": 52555386880,
  "threads": 2,
  "native_threads": 1,
  "precursor_software": "GPU: Python3.10/Torch2.7.1+cu128/Qiskit0.44.2/NumPy1.26.4; worker records actual versions",
  "background": "desktop GPU/RustDesk activity not stopped; no exclusive hardware or operational break-even claim"
}
