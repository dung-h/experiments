# Public data boundary

[Simulator evidence](simulator/README.md) contains own local runtime, quality,
failure records and saved predictions under context-specific settings.
Additional [AM1 Dense records](../results/accuracy_cost/dense/README.md) retain
measured prediction-cost data, not new simulator executions.

No third-party real-QPU labels, row-level derivatives, source QASM, serialized
DAGs or pretrained checkpoints are bundled. [Dataset construction](../docs/data_preprocessing.md)
describes QPU joins and qualification; public QPU scores are aggregates.
Qualified recipes are not exact recovered historical logical instances.

[Distribution](../docs/distribution.md) and [provenance](../provenance/README.md)
separate own measurements, external assets and private references.
Never fill a missing private input with a guessed label, circuit or checkpoint.
