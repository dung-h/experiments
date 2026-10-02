# Unified learned derived comparisons v1

This stable derived artifact compares frozen out-of-fold predictions only; it performs no training or inference. The two comparisons are V3-large versus polynomial on 8,766 common rows and V3-large versus V3 on 8,763 common rows. Coverage remains on the 8,767-row canonical envelope. Both clocks are `archived_observed_service_execution_time`.

Paired deltas are candidate absolute error minus reference absolute error, so negative favours the candidate (always V3-large). S49 grouped bootstrap uses 1,000 complete source-specific clusters and the registered `20260925-bootstrap` stream. Source-balanced three-source macro MAE values are diagnostics, not cross-source ranking claims.

Rows `qonductor_single_circuit_ibm|row39`, `row40`, and `row41` are representable but inaccurate extrapolation for V3-large; they are predicted, not unavailable. Row `qonductor_single_circuit_ibm|row4477` is unavailable for V3-large and V3 because dynamic control is not representable by the declared flat DAG. V3 is also unavailable on the three tail IDs due to the graph operation-count resource limit. Unavailable rows are never scored as zero error and are not spliced into V3 or V3-large.
