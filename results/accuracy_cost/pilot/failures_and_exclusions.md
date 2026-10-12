> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Failures and exclusions

- analytical fold -1: RuntimeError: unqualified_asset:/home/server/Documents/.venv-qonductor/lib/python3.10/site-packages/qiskit_ibm_runtime/fake_provider/backends/kyoto/conf_kyoto.json
- dense_fp32 fold 0: RuntimeError: unqualified_asset:/home/server/Documents/quantum-runtime-estimator-replications/work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl
- Native Qiskit manual same-scope reconciliation: snapshot pins verified; parse exceeded the predeclared 30-second request cap. Gate 1 incomplete, no B1/bridge timing linked to its accuracy. Details: native_replay_timeout_receipt.json. Peak RSS of this failed manual attempt NOT MEASURED.

The first dense attempt stopped before inference because an indirect feature pin was not recognized. A coordinator RSS process-exit race also interrupted that execution. Both are AM1 harness issues, not model failures. Original failure receipts remain. A manually recorded adapter-only correction resolved the indirect dense pins; workloads, checkpoint selection and tolerances unchanged. Additional native snapshot qualification, if present, is separately recorded.

Maestro missing saved output remains UNAVAILABLE_SAVED_OUTPUT, never a zero prediction; selected workload unchanged. Joint/fixed-MPS not selected in narrowed pilot, not failed. No TN planner, contraction, MPS state simulation, QPU job or QCEC. Full B2 remains NOT EXECUTED. No second/expanded pilot, model refit, environment modification or hidden classical replacement.
