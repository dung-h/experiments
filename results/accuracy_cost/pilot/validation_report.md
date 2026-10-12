> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Final saved-evidence validation

{
  "status": "PASS_WITH_SCOPE_LIMITATIONS",
  "required_artifacts": [
    "README.md",
    "environment_manifest.md",
    "artifact_integrity.md",
    "selected_variants.csv",
    "selected_workloads.csv",
    "replay_validation.md",
    "replay_validation.csv",
    "raw_timings.csv",
    "timing_summary.md",
    "accuracy_cost_analysis.md",
    "failures_and_exclusions.md",
    "scientific_decision.md",
    "validation_report.md"
  ],
  "selected_requests": 140,
  "raw_timings": 18891,
  "replay_comparisons": 1050,
  "numerical_replay_failures": 0,
  "unavailable_saved_outputs": 1,
  "immutable_source_pins": 1081,
  "loaded_asset_pins": 300,
  "selection_unchanged": true,
  "timing_arithmetic": true,
  "repetitions_per_request": 30,
  "source_pins_unchanged": true,
  "output_bytes_before_final_receipt": 16466106,
  "rss_peak_bytes": 1446191104,
  "cuda_reserved_peak_bytes": 216006656,
  "memory_scope": "completed workers; native manual failure peak not measured",
  "full_B2_executed": false,
  "training_simulation_QPU_QCEC": false,
  "complete_Pareto_frontier": false,
  "commit_push": false
}

PASS applies to the completed pilot subset and saved evidence arithmetic/identity, not unmeasured boundaries or a full benchmark frontier. Repeated measurements are retained, not independent workload support. Original harness failures remain in execution receipts. Detailed setup decomposition and peak-memory attribution limitations are explicit.

Source digests, selection hashes, 30-per-request counts, no negative intervals, exact saved-accuracy joins, replay gates and resource caps were checked. Recheck with `python -B benchmark_v1/scripts/validate_am1_overhead_pilot.py`; no measurement or fitting.
