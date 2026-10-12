> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# QPU stage reconciliation

Raw QASM text: pinned unchanged. Parser:45/45 ordered instructions equal. Registers/global bit positions:45/45 equal. Gate definitions may serialize/decompose differently between libraries; detailed signatures are retained in each diagnostic. No new semantic gate replacement is applied. Transpilation:absent in both paths.

First order divergence:RemoveBarriers on38/45 (all selected Ma–Li). Pruned circuit/DAG order differs on those38; same named-operation dependency topology proven by causal-wire keys and canonical sparse metadata/adjacency. Nodes/edges/sparse fields then inherit this permutation. Integer node_index stays arange, not permutation-invariant; old reordered input is therefore NOT accepted for the trained model.

178-column encoding:input/output prefix then op order; wire slots46 onward, T1/T2 at173..176, node_index177. 51 globals:gate counts44 plus7 structural metrics. Frozen retained indices/masks/scalers unchanged. Final arrays were regenerated from raw QASM, transported across interpreters, and compared against original _make_example using independent NPZ/CSV evidence. All45 Graph and45 MLP inputs exact including float32 values,shape,edge order,int64 dtype. No numeric tolerance needed for transformed inputs.

Per-ID deltas are in qpu_graph_mismatch_inventory.csv and graph_replay_validation.csv. Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json.
