# Ma–Li flow on transpiled native DAGs

**Finding recorded:** 2026-09-22.

The circuit object is `transpile(logical QASM, target backend)`.
The GNN reads a 256-bin coarsening of that native DAG, including
physical T1/T2. It is not Ma–Li's logical DAG and not scalar T1/T2-free features.

- device: `cuda`
- QPU rows: 340
- simulator native-DAG rows attached: 0

| Model | log-R² / MAE |
|---|---|
| QPU scratch, transpiled DAG | -93.7172 / 19.334 s (n=340) |
| Simulator in-domain, transpiled DAG | pending |
| Sim DAG → affine QPU | pending |
| Sim DAG → fine-tune QPU | pending |
