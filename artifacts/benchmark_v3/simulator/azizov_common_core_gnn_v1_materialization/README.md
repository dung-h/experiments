# Azizov common-core static inputs (partial)

This pack freezes the 162-row `core_q2_q9` join to the existing C44 warm-Aer labels and materializes source-QASM DAG topology for the unique exact hashes. It does **not** execute a simulator, transpile circuits, fit a predictor, or authorize training.

## Contents

- `fold_assignments_and_hash_audit.csv`: 162 panel members, exact-QASM hashes, frozen folds and joined observed warm-execution labels.
- `source_dag_topology.jsonl`: 150 unique source DAGs; this is topology-only and is not claimed to be the Azizov GNN input representation.
- `feature_dictionary.json`: evidence-backed counts and explicit unresolved fields. No 41/54/41 vectors or unnamed proxies are fabricated.
- `environment_and_context_lock.json`: candidate parser environment and missing frozen historical target/noise context.
- `static_qa.json`, `source_hashes.json`: join, fold-leakage, source-hash, structural and provenance audits.
- `materialize_static_pack.py`: static-only materializer.

## Reproduce the static pack

From the repository root, with the candidate Qiskit environment activated:

```bash
python artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization/materialize_static_pack.py
```

The script reads the existing panel, C44 labels, protocol, and external source QASM files; it does not modify those inputs. The pinned frozen target/noise file named by the protocol is absent, so the transpiled representation remains blocked. The 36 Ma–Li-derived field names and exact 13 post-transpilation fields are not evidenced by the pinned sources, so all model-fit variants remain blocked pending method review.
