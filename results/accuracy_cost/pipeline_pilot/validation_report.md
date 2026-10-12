> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Saved-evidence validation

PASS for scoped measurements; failed qualification gates are preserved, not called measurement PASS.

{
  "source_pins_match": true,
  "original_artifacts_unchanged": true,
  "phase2b_authorized_by_conditional_gate": true,
  "raw_input_replay_requests": 210,
  "raw_input_replay_pass": 164,
  "raw_input_replay_fail": 46,
  "numerical_replay_comparisons": 761,
  "numerical_replay_pass": 761,
  "gate_false_methods_not_timed": true,
  "requests_with_complete_B2": 45,
  "complete_B2_requests_including_repeats": 1350,
  "native_scalar_queries": 105,
  "raw_stage_rows": 12600,
  "timed_query_units": 150,
  "measurement_repetitions": 30,
  "phase2b_wall_seconds": 360.7575493579934,
  "ram_peak_bytes": 1373908992,
  "vram_reserved_peak_bytes": 39845888,
  "highest_gpu_stage_end_allocation_bytes": 23073792,
  "train_refit_simulator_QPU_planning": false,
  "standalone_cold_B3": false,
  "complete_graph_MLP_B2_comparison": false,
  "full_Pareto_frontier": false,
  "full_RQ4_closed": false,
  "commit_push": false,
  "positive_clock_arithmetic": true,
  "quantile": "linear interpolation within30 requests, median acrossqueries",
  "no_outlier_deletion": true,
  "exact_parent_ID_fold_and_target_alignment": true,
  "frozen_med3_predictions_unchanged": true,
  "qpu_common_target_authority": "panel.csv target_seconds; saved actuals within1e-10s",
  "maximum_saved_target_serialization_delta_seconds": 1.7763568394002505e-15,
  "preparation_fraction_same_requests": 0.9813402497333147,
  "unmatched_target_execution_environment": true
}

Commands: python3 -B benchmark_v1/scripts/analyze_am1_phase2.py --check
and focused Phase2/postpilot/pilot saved-evidence pytest. Evidence reproducibility
recalculates metrics, repetitions, ID joins, interval weighting, caps and pins.
Tests never launch estimators, training, QASM parsing or target execution.
Do not substitute a prior release/test PASS for Phase2 validation.
