# How to read this benchmark

This pack is a derived reader layer. It copies published numbers from pinned
artifacts. It does not recompute MAE, R², or predictions.

## There is no flat leaderboard

Archived observed QPU service time, scheduled duration, simulator wall-clock,
and contraction-only plan estimates are different clocks. A scientifically
valid ranking does not flatten them into one table of winners.

Fidelity class is a claim boundary, not a quality rank. Native, source-native
reimplementation, independent reimplementation, adaptation, analytical proxy,
unavailable, baseline, and local measurement are not interchangeable.

## What the main comparison is

The main numerical comparison is unified adaptations on the declared 8,767-row
archived-real-QPU envelope, using the frozen outer split and the
`archived_observed_service_execution_time` clock.

Report both denominators:

1. full envelope (8,767 assigned rows, including explicit unavailable/failure);
2. common successful intersection (same clock, same split/population, same
   successfully predicted rows).

Do not treat Ma--Li source-native 340 or Qonductor-local 4,482 as that 8,767
envelope.

## Unavailable is valid

A missing prediction is not zero error and not a silent failure. Graph V3
does not predict four rows, including `qonductor_single_circuit_ibm|row4477`.
Graph V3-large rescues rows 39/40/41 and still leaves row4477 unavailable.
Do not splice row4477 into V3 or V3-large.

## V4 is not scored

`mali_graph_architecture_v4_control_flow` is a signed representation contract
only. It is not trained and has no score. A future row4477 V4 value, if ever
authorized, would be structural-OOD for fold 0 unless an execution contract
adds independent dynamic-control support.

## Source slices are not a second winner table

Ma--Li / Qonductor / QPack slices of the unified adaptations are transport
diagnostics. Source-local scorecard rows stay source-local. The only
cross-source macro copied here is the one already published in v4r2 as an
equal-weight diagnostic, not a primary micro metric.

## Appendix clocks stay in the appendix

Simulator first/warm execution, cuTensorNet selected-plan RUNTIME_EST, and
planning telemetry (network_build / path_search) remain separate. The two
`cutensornet_planning_telemetry` rows are historical-unmapped descriptive
rows; they must not inherit native RUNTIME_EST.

Ma--Li 340 fold 3 is catastrophically unstable at seed 1234. That fold is
historical source-native evidence, not a three-seed median. The three-seed
plan is NOT_AUTHORIZED.

## How to use the files

- `unified_envelope.*` — view A
- `common_successful_intersection.*` — view B
- `source_stratified.*` — view C
- `appendix_native_sensitivity_simulator.*` — view D
- `blocked_unavailable.*` — every blocked/unavailable/contract-only row
- `manifest.json` / `source_hashes.json` — pins for every source
