> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Validation

{
  "QPU_accounted": 45,
  "QPU_after_exact": 45,
  "QPU_before_exact": 7,
  "QPU_source_failure_identity_exact": true,
  "QPack_raw_unavailable_preserved": 5,
  "bridge_CUDA_calls_field_definition": "0 explicit CUDA calls in harness, not a process-wide interception count. Torch __config__ import attempted device discovery and warned under address-space cap; tensors remained CPU. No forward/kernel workload was requested.",
  "canonical_graph_equivalent": 45,
  "commit_push": false,
  "dense_context_attempts": {
    "dense_fp32": 30,
    "dense_fp64": 30
  },
  "dense_context_passes": {
    "dense_fp32": 30,
    "dense_fp64": 30
  },
  "dependency_changes": false,
  "diagnostic_concurrency_note": "Two read-only diagnostic subprocesses overlapped; per-process8GiB cap. Bridge used one preprocessing server. No measurement interference. Future measurement remains serial.",
  "frozen_file_pins": 1458,
  "frozen_files_unchanged": true,
  "independently_encoded_model_ready_comparisons": 210,
  "independently_encoded_model_ready_exact": 210,
  "readiness": "READY FOR B2 MEASUREMENT via declared dual-interpreter preprocessing bridge; not single-interpreter compatibility",
  "tests": {
    "bridge_import_warning": "Torch __config__ import attempted CUDA device discovery under the address-space cap and emitted an initialization warning. The receipt's CUDA_calls=0 means explicit harness calls, not global library-call interception. All compared tensors remained on CPU; no model forward or kernel workload was requested.",
    "command": "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/server/Documents/.venv-qonductor/bin/python -B -m pytest -q -p no:cacheprovider benchmark_v1/tests/test_am1_preprocessing_recovery.py benchmark_v1/tests/test_am1_phase2.py benchmark_v1/tests/test_am1_postpilot_audit.py benchmark_v1/tests/test_am1_overhead_pilot.py",
    "existing_phase2_postpilot_pilot_tests": 18,
    "first_authoring_test_run": {
      "failed": 1,
      "passed": 30,
      "reason": "The test incorrectly assumed that the legacy parser preserves an arbitrary explicit same-named RCCX body. Built-in registration supersedes it.",
      "resolution": "Scope remains the pinned panel's implicit library RCCX. Replace this incorrect test assumption with a direct original-lowerer negative control on an unvalidated gate definition; do not change historical parser or benchmark inputs.",
      "test": "test_dense_custom_RCCX_not_overwritten"
    },
    "fresh_final_adapter_bridge_revalidation": {
      "comparisons": 210,
      "method": "Reexecute all raw-to-model-ready requests after static-extractor setup caching; assert exact equality to the original independent bridge receipt without overwriting it.",
      "model_forward_calls": 0,
      "new_timing_measurements": 0,
      "receipt": "bridge_validation.json",
      "status": "PASS"
    },
    "git_diff_check_exit_code": 0,
    "new_preprocessing_tests": 14,
    "no_model_inference_training_latency_or_quantum_workload": true,
    "passed": 32,
    "status": "PASS",
    "stdout_tail": "32 passed in 14.56s"
  },
  "training_inference_latency_quantum_workloads": 0,
  "unchanged_split_checkpoint_transform_prediction_protocol_QASM": true
}

Exact sparse and model-ready checks use independent frozen artifacts,not model outputs. Numeric rawglobal tolerance1e-12 only reflects decimal CSV storage;all transformed arrays matched exactly. Static RCCX fullmatrix evidence is narrowly about the declared gate definition,not historical logical recovery.

Commands:recover_am1_preprocessing.py --stage prepare; --stage diagnose --flavor historical in historical Python; --flavor phase2 in CUDA Python; --stage bridge in CUDA Python with CUDA_VISIBLE_DEVICES empty; --stage report/--stage check. Targeted pytest receipt is recorded separately; do not substitute old Phase2 PASS. gitdiff--check is recorded with the test receipt.

Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json.
