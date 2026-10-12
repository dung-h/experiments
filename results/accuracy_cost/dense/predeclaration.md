> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Dense matched full-B2 predeclaration

Frozen before CUDA inference or timing. SHA256 `83b0ce5a29f5083a4ba18d4df163455724a54aef3204e23a093d02f158150bca`.

{
  "authorization": "user_attachment_485bfa96-70bc-41fa-b711-f1113cee4c8c",
  "phase": "AM1_Phase5_Dense_B2",
  "created_utc": "2026-10-11T16:48:28Z",
  "methods": [
    "mali_style_dense_graph",
    "matched_metadata_mlp"
  ],
  "seeds": [
    42,
    1234,
    31415
  ],
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
  "environment": {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "QISKIT_IN_PARALLEL": "FALSE"
  },
  "affinity": [
    8,
    9
  ],
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
  "primary_timed_requests": 3600,
  "atol_seconds": 1e-06,
  "rtol": 1e-05,
  "aggregation": "host median3; each CUDA seed predicts log1p seconds, original expm1 then clamp_min(0), D2H",
  "boundary": "resident raw QASM bytes plus frozen context/transform to host scalar ensemble prediction at coordinator",
  "architecture": "one resident CUDA coordinator and one historical CPU typed-array stdio child; 30 models/precision resident; no feature cache",
  "actual_model_precision": "float32 in BOTH contexts; FP32/FP64 denotes historical simulator precision, not estimator arithmetic",
  "stages_cpu": [
    "request_decode",
    "raw_base64_decode",
    "parse_QASM",
    "normalize_circuit",
    "extract_7_globals",
    "construct_DAG",
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
  "MLP_work": "parse, original RCCX normalization, seven globals, frozen transform; omit DAG construction only, not normalization",
  "nested_spans": "CPU durations nested in host wait; residual=host write+wait minus child durations, never cross-process timestamps; not pure IPC",
  "ordering": "FP32 then FP64; frozen circuit order; alternating Graph/MLP by (row_index+rep)%2; 15/15 first positions",
  "bootstrap": {
    "replicates": 2000,
    "seed": 20261011,
    "unit": "paired existing QASM-hash structural groups within each precision; no repetition resampling"
  },
  "accuracy": "frozen median3 predictions, separate exact30 timed subset and full150 OOF/context; saved train-only median reference; no model selection",
  "constants": {
    "fp32": {
      "0": 0.0002760520001174882,
      "1": 0.0002901919942814857,
      "2": 0.00026321249606553465,
      "3": 0.0002889989991672337,
      "4": 0.00026321249606553465
    },
    "fp64": {
      "0": 0.00029387298854999244,
      "1": 0.000307861999317538,
      "2": 0.0002921329942182638,
      "3": 0.00030580299790017307,
      "4": 0.0002803370007313788
    }
  },
  "scaling": "descriptive Spearman and Pearson versus existing width/depth/gates/nodes/edges; report range/family confounding; no asymptotic law",
  "historical_execution": "ID/context/seconds-matched R5 targets only, labeled historical cross-environment simulator runtime; no operational break-even",
  "setup_excluded": "imports, process start, resident model load, input disk retrieval and fold routing; recorded separately, not cold B3",
  "stop": "any digest/parity/hash/placement drift, 30s request,120s setup,8GiB aggregate RSS/VRAM,7200s all replay+timing,256MiB outputs,3GiB diskfloor or foreign GPU workload; no retry",
  "optional_controls": "1000 timer-pair and30 empty stdio echo samples, no subtraction",
  "no_training_refit_simulator_execution_new_circuits_dependency_change": true
}

Exact60 context requests and60 checkpoints are in the JSON. No circuit or seed may be selected by outcomes. Stage medians are not additive.
