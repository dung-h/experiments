# NoTaDS / HyperQ application on Track A

Status: **COMPLETE_VALIDATED**. Read [REPORT.md](REPORT.md) for method fidelity, all results, source slices, paired intervals and input findings.

- Protocol: [original](PROTOCOL.md), [pre-scoring input/clock amendment](PROTOCOL_AMENDMENT_01.md).
- All 4,515 observations / 165 groups, original five folds, nine scored routes/reference.
- NoTaDS is a relative native-circuit score, evaluated as an explicitly labeled train-only affine adaptation for seconds prediction. Raw score is not called seconds.
- HyperQ paper weighted-path calculations equal existing Qiskit duration predictions on every row; do not add an independent leaderboard entry for the duplicate.
- Independent validation: [226,083 checks PASS](independent_validation.json).
- [Metrics](metrics.csv), [paired comparisons](paired_comparisons.csv), [predictions](predictions.csv), [feature receipt](feature_receipt.json), [completion receipt](completion_receipt.json).

Preserve frozen/historical results. No new method search, circuit execution, commit or push was performed.
