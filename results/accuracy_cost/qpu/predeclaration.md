> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Matched QPU full B2 predeclaration

Frozen before inference or timing. Machine-readable authority: predeclaration.json (SHA256 4046a5230eb86382491d6085a870d2b4d7ac3ec1ea03ce138e376d30221c4a6e).

{
  "authorization": "user_attachment_7c20ec85-cab5-4ed0-9830-c7e7c8c1f57d",
  "phase": "AM1_Phase4",
  "created_utc": "2026-10-11T15:59:39Z",
  "methods": [
    "mali_logical_graph",
    "mali_global_mlp"
  ],
  "seeds": [
    42,
    1234,
    31415
  ],
  "affinity": [
    8,
    9
  ],
  "environment": {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "QISKIT_IN_PARALLEL": "FALSE"
  },
  "cpu_environment": {
    "python": "3.10.21",
    "packages": {
      "qiskit": "2.5.2",
      "numpy": "2.2.6"
    }
  },
  "gpu_environment": {
    "python": "3.10.21",
    "packages": {
      "torch": "2.7.1+cu128",
      "torch-geometric": "2.6.1",
      "qiskit": "0.44.2",
      "numpy": "1.26.4"
    }
  },
  "limits": {
    "cold_seconds": 120,
    "disk_floor_bytes": 3221225472,
    "output_bytes": 268435456,
    "ram_bytes": 8589934592,
    "request_seconds": 30,
    "vram_bytes": 8589934592,
    "wall_seconds": 7200
  },
  "batch": 1,
  "repetitions": 30,
  "warmups": 3,
  "atol_seconds": 1e-06,
  "rtol": 1e-05,
  "aggregation": "host median of three sequential original seed predictions, each original output already seconds",
  "boundary": "resident raw QASM bytes plus nominal properties and frozen transform to host scalar ensemble prediction",
  "architecture": "one resident CUDA coordinator/inference process holds 30 models (5 folds x 2 methods x 3 seeds); one resident CPU stdio preprocessing child; no feature cache",
  "stages_cpu": [
    "request_decode",
    "parse_QASM",
    "extract_51_globals",
    "decode_nominal_T1T2",
    "prune_and_compact_DAG",
    "frozen_transform",
    "array_serialization",
    "payload_JSON_encode"
  ],
  "stages_host": [
    "request_serialization",
    "request_write_flush",
    "response_wait_inclusive",
    "response_JSON_decode",
    "typed_array_decode",
    "tensor_construction",
    "batch_and_H2D",
    "three_seed_inference_inverse_and_D2H",
    "host_median_aggregation"
  ],
  "nested_spans": "CPU spans are nested inside host response_wait_inclusive; never add both; IPC/unattributed residual = write+wait minus sum child durations, using duration differences only, not cross-process timestamps",
  "ordering": "Phase2 frozen row order; per repetition Graph/MLP order alternates by (row index+rep)%2; 15 of each first-position per row",
  "bootstrap": {
    "replicates": 2000,
    "seed": 20261011,
    "unit": "paired structural-group cluster, pooled row-weighted statistic; separate source strata"
  },
  "constants": {
    "0": 6.597566366195679,
    "1": 6.6256725788116455,
    "2": 6.2906553745269775,
    "3": 6.628589630126953,
    "4": 6.649125337600708
  },
  "precision": "original float32 parameters/features, no autocast, no altered deterministic/TF32 settings",
  "setup_excluded": "imports, code extraction, model load, disk input acquisition/fold routing, process startup; separately recorded once, not B3",
  "stop": "any replay/parity/hash/placement mismatch; any request>30s, setup>120s, aggregate pipeline RSS/allocated or reserved VRAM>8GiB, disk<3GiB, new outputs>256MiB, wall>7200s, foreign compute workload",
  "no_training_quantum_execution_refit_dependency_change": true,
  "optional": "1000 timer-pair control and 30 empty IPC echo controls; no additional B1 or shared pipeline measurement"
}

Exact 45 IDs/raw hashes and 30 checkpoint identities are in the JSON. No feature caches, QPack replacements or test-result-based selection. Warm loaded estimators are standalone per-method requests. CPU transforms execute exactly once before transport.
