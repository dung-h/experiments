> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Compatibility tooling changes

New isolated module benchmark_v1/scripts/am1_preprocessing_compat.py enforces Qiskit2.5.2 and NumPy2.2.6. It reuses existing parser,global extractor,compact Graph and RCCX lowering. It reconstructs consumed arrays only, preserves ordering/dtypes and uses supplied frozen transform values. No expected graph,observation ID,label,prediction,checkpoint or cache is accessible through its serving interface. Unknown sources fail closed.

The raw-to-array --serve endpoint receives JSON with base64 QASM/properties and frozen transforms, returns base64 typed arrays. One resident worker avoids per-query process spawning in a prospective measurement; IPC/encoding must remain inside future B2 accounting. Static pinned extractor functions load once during setup; no circuit features or graphs are cached. Independent consumer verification uses original _make_example and dense encode_dag/normalized_globals, not new generated targets.

New benchmark_v1/scripts/recover_am1_preprocessing.py pins source evidence, diagnoses both environments and validates the bridge. New report_am1_preprocessing_recovery.py derives CSV/reports and checks preserved timing arithmetic. New test_am1_preprocessing_recovery.py tests environment rejection, transport,dtype/schema parity,RCCX controls and complete receipt coverage.

Historical run_am1_phase2.py,mali_full_features.py,prepare_logical_benchmark.py,cudaq_qiskit_adapter.py and trained schemas are unchanged. This is a dependency-isolated recovery, not an unverified cross-version port. Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json.
