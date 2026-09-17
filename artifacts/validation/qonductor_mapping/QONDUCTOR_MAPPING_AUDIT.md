# Qonductor circuit/job mapping audit

This audit is credential-free and deliberately refuses to infer a join.

| Object | Count/range |
|---|---:|
| Resource-estimator rows | 100 |
| Resource-estimator columns | `predicted, real, dag` |
| Database jobs | 7449 |
| Database circuits | 166093 |
| Distinct database backends | 17 |
| Archived QASM files | 166093 |

Stable key in resource CSV: **False**.

The CSV has no `job_id`, `circuit_id` or `ibm_quantum_id`. Its `real`
values cannot be safely matched to the database by row order or nearest
runtime; the units and population also differ. Consequently, a QCRE
comparison on the 100 headline rows is not currently identifiable.

Conclusion: No safe row-level join is established: the 100-row resource CSV has only predicted/real/dag columns, while circuit/job keys live in a separate database. Do not attach QCRE values by row order or nearest time.
