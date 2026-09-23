# Ma–Li flow, Azizov circuit object

**Finding date:** 2026-09-22.

Ma–Li encode the logical/source QASM as a DAG, attach backend T1/T2 onto
logical qubit indices, train a Graph Transformer on simulator
`time_taken`, then fine-tune on Osaka/Kyoto `result.time_taken`.

This experiment keeps that two-stage estimator, but replaces the circuit
object. Wherever Ma–Li would call `QuantumCircuit.from_qasm_file` and
`circuit_to_dag`, this run first does `transpile(qc, target_backend)` and
builds the DAG from that native circuit. Node qubits are physical qubits
after layout. T1/T2 are those physical qubits' calibration values.

Target backends:

- simulator stage: current `FakeWashingtonV2` / `FakeSherbrooke`
- QPU stage: current `FakeOsaka` / `FakeKyoto`

Labels stay the public CSVs. Clocks are never pooled. Snapshots are not
historical job-day calibrations.

The native DAG of a 127-qubit mapped circuit has on the order of 10^5
operation nodes. Ma–Li's `TransformerConv` on the full graph is not
runnable here (the 340-row Osaka/Kyoto corpus alone is 67 million nodes).
The model therefore sees the **same transpiled DAG**, coarsened in
topological order to at most 256 bins, with coarse dependency edges and
physical T1/T2 pooled inside each bin. That is still the mapped circuit,
not a handful of scalar T1/T2-free features, and not the logical QASM.

It is not Azizov's published GNN, and it is not a bit-for-bit Ma–Li
checkpoint with one line changed.
