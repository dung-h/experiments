> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Dense RCCX reconciliation

FP32 failures:3d37464cf8c53831d2867c0c111a83e03a820e90fa26f77d01a4a85f4f9d6875 (36 RCCX),5042c6dd513b1366c0a145850de9e534b0e38aabc818b994ab8c4b6cad7ad5c8 (6),84056291d9ff227b3c615cc378349fa1154f5a122bcd0eb6bc04045bbf3affa0 (16). FP64 includes only5042… among its selected30. Same hash means same gate issue, not a precision-dependent semantic difference.

Their raw QASM includes qelib1.inc and no explicit gate rccx declaration. Old RCCXGate itself has U1/U2 definition even with original from_qasm_str parser dispatch; changing dispatch alone in the old interpreter does not fix this. Restoring existing2.5.2 library provides H/T/TDG/CX, retaining the original relative-phase gate. Definition qubit order/parameters are captured in diagnostic receipts.

Full static8x8 matrix control (not prepared-state-only):{"negative_control_CCX_difference": 2.0, "old_definition_vs_static_rccx": 3.349388238726994e-16, "old_vs_restored_definition": 3.0425610782034266e-16, "restored_definition_vs_static_rccx": 3.3465051065045716e-16}; atol1e-14,rtol0 predeclared. Both old and restored decompositions match the RCCX static gate matrix; CCX negative control is rejected. No QCEC,circuit simulation,statevector or quantum workload was run.

Scope is the pinned panel's implicit library RCCX. An initial test assuming the legacy parser would preserve an arbitrary explicit same-named QASM body failed; the parser registers built-ins. The corrected control directly verifies the lowerer's rejection of an unvalidated definition. No custom raw benchmark gate was replaced. This recovery does not qualify arbitrary custom RCCX declarations for future inputs.

All30 FP32 and30 FP64 consumed DAG/globals and frozen-transform Graph/MLP arrays match exactly. Terminal measurements and global phase remain preserved by original lowerer. No RCCX-to-CCX substitute or retrospective custom gate mapping. Raw QASM and frozen IR hashes unchanged. Detailed context rows:dense_replay_validation.csv. Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json.
