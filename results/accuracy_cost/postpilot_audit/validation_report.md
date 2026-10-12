> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Saved-evidence audit validation

```json
{
  "status": "PASS_WITH_REPORTING_CORRECTIONS",
  "historical_pilot_payload_files": 155,
  "historical_pilot_bytes": 16469748,
  "immutable_original_source_pins": 1081,
  "raw_timing_rows": 18891,
  "raw_boundaries": {
    "first_after_ready": 141,
    "CONTROL": 480,
    "B1": 11670,
    "P1": 6600
  },
  "replay_statuses": {
    "PASS": 1049,
    "UNAVAILABLE_SAVED_OUTPUT": 1
  },
  "seed_comparisons": 660,
  "ensemble_comparisons": 220,
  "scalar_formula_comparisons": 170,
  "neural_query_method_groups": 220,
  "dense_execution_identity_rows": 60,
  "original_execution_repetitions_checked": 900,
  "simulator_ratios": [
    {
      "context": "dense_fp32",
      "circuit_hashes": 30,
      "graph_slower": 27,
      "mlp_slower": 20,
      "median_graph_ratio": 13.498039752917748,
      "median_mlp_ratio": 2.0307128552730322
    },
    {
      "context": "dense_fp64",
      "circuit_hashes": 30,
      "graph_slower": 28,
      "mlp_slower": 22,
      "median_graph_ratio": 16.0967698307204,
      "median_mlp_ratio": 2.446228184639695
    }
  ],
  "accuracy_population_rows": 22,
  "paired_latency_contexts_verified": 3,
  "selected_requests": 140,
  "total_sources_audited": 1269,
  "qiskit_candidate_inputs": 50,
  "qiskit_failed_input": "UNRESOLVED",
  "new_inference": false,
  "new_parser_call": false,
  "new_measurement": false,
  "training": false,
  "original_data_modified": false,
  "complete_Pareto_frontier": false,
  "full_RQ4_closed": false,
  "stop_am1": true
}
```

Fresh checks recompute timings from nanosecond intervals, exact 30-repetition membership, selected fold identities, dense target reduction from three sessions of five original measurements, QASM/IR/precision/shots, full and subset prediction metrics, seed medians, declared tolerances, missing-target semantics and 48 setup bundles. Final pilot payload bytes are measured read-only; historical interim byte fields are not final totals. Sources and all old AM1 artifacts are hash-pinned before and after.

Reproduce this audit without any model/engine call: `python -B benchmark_v1/scripts/build_am1_postpilot_audit.py --check`. No imports of Torch, CUDA-Q, Qiskit parser or producing runners. Source/payload hashes are in provenance_manifest.json. Unknown historical GPU UUID and timeout culprit remain unknown; arithmetic validation is not operating-condition equivalence.
