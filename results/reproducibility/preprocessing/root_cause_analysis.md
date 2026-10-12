> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Root causes

QPU: all45 raw parser instruction sequences and register indices agree across environments. Literal instruction-class/library-definition descriptors differ earlier in all45,including7 passing Qonductor cases. That metadata is not consumed by this encoder; it cannot explain the38-case failure by itself. The38 Ma–Li cases first diverge in model-consumed order during RemoveBarriers's DAG-to-circuit reconstruction. Terra0.25.2 (Qiskit distribution0.44.2) uses DAGOpNode.sort_key=str(qargs) and lexicographic topological ties. The2.5.2 pipeline produces a different deterministic order. RemoveFinalMeasurements subsequently preserves a version-specific order. The graph schema consumes that order, and its node_index column177 makes ordering model-visible.

All45 named-operation dependency graphs and wire/T1/T2 attributes agree under a demonstrated causal-node relabeling (node_index excluded only for this secondary equivalence test). This is not an assertion that the old ordered tensors are model-compatible. Exact compatibility is restored only through historical preprocessing and the frozen transforms. No graph-isomorphism-only PASS or per-observation reordering is used.

Mismatch fields: node_type,wire0/1,T1/T2_0/1,edge_index. Shapes/counts/dtypes and51 global features already match. Frozen NPZ archive bytes are not regenerated; content arrays are compared exactly. Edge features do not exist in this schema. Nominal calibration values, unit conversion, active-wire pruning, vocabulary, sparse deduplication and global schema remain unchanged.

Dense: the old library RCCX definition is U2(0,pi),U1(+/-pi/4),CX. The restored library definition is H,T,TDG,CX. The existing lowering correctly refuses the former unvalidated basis. This is a library-definition divergence before adapter lowering, not merely a wrong circuit name or shots/precision effect. Restored historical dispatch plus environment reproduces consumed DAG/globals exactly.

No model prediction or runtime target was used to choose the fix. Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json.
