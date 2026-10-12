> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Validation report

Integrity: 8726 pin/manifest rows; PASS 8538, MISSING 0, HASH MISMATCH 6, UNVERIFIED 182; absent expected manifests 0.

Metric/timing/alignment checks: 748588; PASS 748588; discrepancies 0. Link audit: 590 local links, 0 broken; RQ-index missing artifact paths 0. Validators: 7/7 pass. Registry duplicate identities: metrics 0 (conflicting 0); comparisons 0 (conflicting 0). `git diff --check` exit 0.

Tests: 1268 passed, 0 failed, 1 skipped of 1269 in 181.80 s summed sequential wall time. Machine-readable rows: metrics 78625, comparisons 81415, RQ evidence index 80.

## Deterministic validator matrix

| Validator | Exit | Status | Command |
|---|---:|---|---|
| `final_research_saved_evidence` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B artifacts/final_research/verify.py` |
| `am1_overhead_pilot` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_am1_overhead_pilot.py` |
| `am1_phase4` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/report_am1_phase4.py --stage validate` |
| `am1_phase5_dense_b2` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_am1_phase5_dense_b2.py` |
| `benchmark_interpretation` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_benchmark_interpretation.py` |
| `simulator_reader_tables` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_simulator_reader_tables_v1.py` |
| `reader_table_pack_v4` | 0 | PASS | `/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_reader_table_pack_v4.py` |

`git diff --check`: exit 0. No whitespace errors reported.

This report is an evidence-integrity and deterministic-reconciliation result only; it is not a thesis-level scientific verdict. Source artifact mutations were not requested.
