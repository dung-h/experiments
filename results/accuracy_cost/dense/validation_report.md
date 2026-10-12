> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Validation

Fresh focused suite: **55 passed in 14.36 s**, comprising 12 Phase5 tests and
43 retained AM1/compatibility tests. This is the current test run, not a copied
Phase3/Phase4 receipt. `targeted_tests.json` preserves its command and output.

Independent readback audit passed: sklearn MAE, MedAE, RMSE and R² reproduce
the saved exact30 and full150 metrics at atol1e-12/rtol0. Original per-seed
and median3 predictions, targets and input digests match. Raw per-ID latency
medians and p95 reproduce the tables. Every exclusive stage partition matches
direct B2 to the nanosecond without double-counting child CPU inside host wait.

Maximum absolute prediction replay difference was 7.450580596923828e-9 s,
below the predeclared tolerance. All four context/model gates passed 30/30.
1,569 frozen/source pins remain unchanged. Git diff whitespace check passed.
No timing rerun, training, simulator execution, new circuits or dependency
changes occurred. Execution receipts confirm child exit0 and reaping.

{
  "aggregate_execution_seconds": 57.8049663820002,
  "errors": [],
  "git_diff_check_exit": 0,
  "ids_per_context": 30,
  "methods": 2,
  "repetitions": 30,
  "replay_comparisons": 480,
  "seeds": 3,
  "source_pins_verified": 1569,
  "status": "PASS",
  "timed_requests": 3600
}

Exact replay/timing keys,context precision,balanced order,unit/shape/dtype digests,seed outputs,checkpoint identities,unchanged pins,nested stage reconciliation and caps checked on saved results. Fresh focused tests and independent numerical audit recorded separately. No previous-phase PASS substitutes for this execution.
