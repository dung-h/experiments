# S39 addendum — Qonductor archive-resolved logical subset

This supersedes the blanket Qonductor exclusion in S39 v1 before any
structural-DAG prediction was run.

Qonductor stores every submitted physical QASM in `circuits.zip`. Its own
`benchmarks.zip` names mapped members `family_backend_logical-width.qasm`.
`src/utils/benchmark.py` constructs those members from
`mqt.bench.get_benchmark(..., level=INDEP, circuit_size=width)` and transpiles
them. Exact SHA-256 equality therefore joins a submitted QASM to the archived
family and logical width; it is not inverse transpilation.

| Qonductor rows | status for Ma--Li logical-DAG adaptation |
|---:|---|
| 230 | archive hash resolves family + logical width, and MQT Bench 1.0.3 is deterministic at all observed widths (3/4/5/7); diagnostic adaptation eligible |
| 14 | exact archive identity is `graphstate`; structural-only sensitivity, excluded from first diagnostic scorecard |
| 4,238 | no archive hash join; unavailable for the logical-DAG method |
| 4,482 | still eligible for exact submitted-*physical*-DAG methods (Azizov/compiled baselines) |

The 230 rows are an **archive-resolved logical recipe reconstruction**, not a
claim that Qonductor preserved original pre-transpile QASM bytes. They must be
reported as a separate diagnostic panel, with exact-QASM-hash split and a
provenance flag. They cannot turn the remaining 4,238 rows into logical inputs.

Evidence: 244 exact hash joins in
`qonductor_archive_logical_identity_audit.csv`; MQT wheel 1.0.3 SHA-256
`42affe45f6e6c63d0bdc37093399622ce7c53232c9f486fdc43f345d3b08d30c`; and
a three-repetition Qiskit Terra 0.24.2 probe where all 20 family-width cells
are parameterized and structurally reproducible.

Next gate: materialize the 230 row-to-logical-graph mappings from the pinned
wheel, then apply the normal no-QASM-overlap and fold-local checks.
