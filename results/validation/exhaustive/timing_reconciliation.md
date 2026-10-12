> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Timing reconciliation

Phase 4: raw request rows 2700 (expected 45 IDs × 2 estimators × 30 repeats = 2,700); Phase 5: 3600 (expected 2 precisions × 30 IDs × 2 estimators × 30 repeats = 3,600). Timing checks: 7836; PASS 7836; discrepancies 0.

Per-ID median and p95 (linear quantile over each 30-request sample), request prediction-to-frozen-output alignment, paired per-ID differences/ratios, and stage denominators/partitions/shares are recalculated from saved nanosecond rows. Unit conversion is ns ÷ 1,000,000 = ms. The analysis separately reports median of per-request medians, median of paired ratios, and ratio of marginal medians; these are not substituted for one another.

Stage components are exclusive/residual partitions and are checked to sum exactly to the direct `total_B2_ns` per request. CPU spans nested inside host wait are counted once. Stage medians across circuits are not summed to reconstruct total latency. Phase 1/2 pilot clocks remain separate and are not pooled with Phase 4/5.
