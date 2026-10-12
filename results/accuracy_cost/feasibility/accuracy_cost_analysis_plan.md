> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Accuracy–cost analysis plan

Primary accuracy remains the existing frozen OOF/quality-conditioned results;
do not recalculate predictions, improve hyperparameters or choose a seed.
Track A uses the 4,515-row qualified cohort and route-specific eligibility.
Track B uses each exact engine/task/precision/shots/quality boundary. Do not
rank QPU service predictors against simulator predictors, or contraction EST
against sample/get_state/API timings.

For every future table, retain context/method/fold/model-state/seed-aggregation,
accuracy population N and groups, timing population N and groups, coverage,
B1/B2/B3 scope, cache state, device and units. Full-cohort accuracy next to
subset timing is allowed only with separate denominators and explicit warning:
it is **not** a per-workload paired cost frontier. Workload-aligned plots join
exact ID/hash/rung and outer-fold to frozen predictions; unmatched rows excluded
from paired effects but retained in coverage. Source-ID privileged references
remain distinct. QPack dominates pooled QPU rows; show pooled, equal-source and
equal-structural-group views without treating three sources as three large iid
samples or case assignments as unique rows.

Planned panels, only after execution approval:

1. Frozen MAE/MedAE/RMSE/R² and paired uncertainty versus B1, B2 and B3 separately.
   Negative R² retained. Never compare method-specific available-row MAEs as
   if they had the same test population; show shared-row scores from existing
   predictions alongside the 4,515-row attempt envelope.
2. Accuracy versus RSS/GPU peak, with host/device memory separate. A Pareto
   statement only within one context, target, eligibility, population,
   aggregation and measurement boundary. No strict dominance when paired
   uncertainty overlaps; fewer executable models limits frontier claims.
3. B2 stage stacks/latency versus width, depth, gates, nodes/edges, source/family
   and unique feature count. Size effects are descriptive associations; do not
   fit a universal scaling law from q2–9 or confounded source/width bins.
4. Cache hit/miss, batch1 versus batch32 and per-query six-rung amortization.
   Each uses measured bundle total, not sum of medians. Missing/blocked models
   never receive zero or guessed cost. Timeout/OOM rates are part of feasibility.
5. Optional break-even: for measured setup S and mean request c under specified
   request mix, T(N)=S+N*c. For two methods with distinct setups/c, solve the
   crossing N only when denominator/signs admit a positive crossing; report
   uncertainty and stable-mix assumption. This is computational cost crossing,
   not scheduling benefit or “latency repaid by MAE”. Do not add unmeasured
   training/calibration costs as if zero.

Accuracy gain can justify extra cost **conditional on** an externally specified
request budget and error tolerance; report both axes and leave utility to the
user. No universal MAE-seconds/latency-seconds exchange rate. Fractional MAE gain
is not fractional scheduler throughput gain. If Graph accuracy increment is
unresolved while extra cost is clear, say so; lower estimator latency does not
establish deployment safety, reliable queues or improved quantum utilization.

Important scope loss: absent QPU classical states and dense RF/XGB weights mean
AM1 cannot resolve graph-versus-strong-tree cost on the original winning fits.
Do not substitute an arbitrary cheap regression or claim neural Pareto advantage
because a competitor's artifact was not retained. Report this limitation beside
any selected-subset frontier, not in an obscure appendix.
