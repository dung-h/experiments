# QPack MCP structural reconstruction audit

Status: **pass**. Rows: **3,945**; workflows: **46**; backend-specific configurations: **26**; structural configurations: **6**.

The pinned source constructs a regular-graph MCP QAOA circuit from `(size, P)` and executes it once per optimizer evaluation. A Qiskit 2.5.2 logical replay matches every recorded `(size, P, qubits, depth)` configuration before this audit passes. The resulting structural DAG is valid only for an angle-insensitive diagnostic adaptation. Exact submitted QASM, submitted routing and per-iteration angles remain unavailable.
