# Benchmark data and protocol

This folder holds the frozen inputs, protocol, registry and scripts for the
two-domain benchmark. The benchmark has two separate targets: archived
one-circuit QPU service/execution labels and local simulator runtime under a
declared engine/configuration. Do not combine them into one ranking.

## Reader path

1. Start with the root [methodology](../docs/methodology.md),
   [results](../docs/results.md), and [reproduction guide](../docs/reproduction.md).
2. Use the current [two-domain scorecard v3](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md)
   for numerical tables. Earlier scorecards are immutable historical snapshots.
3. The [fidelity registry](registry/method_fidelity_registry_v2.json),
   [current registry pointer](registry/CURRENT.json), and
   [report-finalization contract](execution/manifests/report_finalization.json)
   describe method status, data authority, target clocks and known gaps.

## Data and comparison contract

The archived QPU ledger contains 8,767 observations: 340 Ma–Li, 4,482
Qonductor and 3,945 QPack MCP. Unified methods use the same frozen outer and
inner grouped splits; source-specific tables are slices of those predictions,
not separately trained models. Source-native Ma–Li and Qonductor results are
separate. QPack's structural circuits are reconstruction-qualified, not exact
historical submissions. The original archived labels are never replaced by
reconstructed circuit timings.

The simulator panel has 204 source members and 191 exact-QASM hashes over
widths 2–16. Measurements identify engine, precision, execution clock, quality
gate and failures. A local simulator timing is not a QPU label; a native
estimator output is not automatically end-to-end simulator runtime.

## Rebuild and limits

Use the copy/paste commands in [reproduction](../docs/reproduction.md). They
verify pinned evidence and rebuild supported metrics from frozen predictions
and measurements; they do not retrain models or rerun simulator timings.
Scientific coverage is partial. V4 control-flow has no trained score, and
several simulator predictors remain unavailable or incomplete.

This package is shared on `benchmark-review` at the author's request, not as
a tagged final release. See the
[earlier release-rights decision](S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md)
for unresolved licensing and provenance questions. No repository license or
additional third-party data permission is inferred from publication.
