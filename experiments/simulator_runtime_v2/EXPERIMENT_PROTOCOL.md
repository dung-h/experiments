# Dense-statevector v2 experiment protocol

**Protocol recorded:** 20 September 2026. This document fixes the questions,
comparators and negative-result criteria before interpreting the v2 outputs.

## Target and scope

The target is `prepared_execute_reset_state`: create a fresh dense statevector,
apply a pre-materialized direct gate schedule and calculate `<Z_0>`. It
excludes circuit construction, gate materialization, transpilation,
sampling and cloud/QPU time. It is one PyTorch reference-kernel contract, not
a claim about Qiskit Aer, CUDA-Q, cuTensorNet or hardware execution.

## Questions

1. Does calibrated analytical work—statevector bytes and
   `gate_count × 2^width`—predict duration for a previously unseen logical
   circuit under a known execution context?
2. Do static interaction-graph features add predictive value beyond that
   analytical baseline?
3. How much does random-row evaluation overstate performance relative to a
   split that excludes all device/precision variants of a test circuit?
4. Is final statevector size sufficient to decide GPU feasibility, or is a
   context-specific peak-allocation envelope required?
5. When all contexts are feasible, can the estimator choose the fastest one
   without measuring that circuit first? This result is reported only if the
   oracle fastest context is not constant.

## Dataset phases

- **Reference:** the committed v1 five-family, 16/20/24-qubit, four-context
  matrix, seed 17.
- **Structural matched extension:** `random_matching` and `random_star`,
  widths 16/20/24, seed 17, all four contexts, with one warm-up and three warm
  measurements. This isolates topology at a fixed kernel/measurement contract.
- **Structural seed extension:** the same two families with seeds 43 and 101
  on GPU complex64/complex128. A seed changes topology for these families, not
  merely rotation angles.
- **GPU frontier:** selected q26/q28 cells. Successful rows remain duration
  labels; CUDA out-of-memory rows are `resource_limit` labels and are excluded
  from duration regression.

Each raw row carries its source run. Compatible duration rows require the same
target and 1-warm-up/3-repeat protocol before `build_corpus.py` may combine
them.

## Models and splits

All duration models use `log(warm_execution_median_seconds)` and report MAE,
RMSE and R² on both log seconds and seconds:

- `median_log`;
- `analytical_calibrated_ridge`: only log statevector bytes, log gate work and
  the known context;
- `logical_hgb`: adds logical depth and one/two-qubit gate counts to the
  analytical workload/context features;
- `graph_hgb`: adds interaction-graph features; `graph_hgb_with_family` also
  exposes the coarse generator family as an explicit ablation;
- `structural_ridge`: a regularized linear comparator with the full static
  feature set.

The primary `circuit_group` split holds `circuit_id` out. Since a circuit ID
includes its topology and seed, every CPU/GPU/precision row for that test
circuit is absent from training. Family-held-out and width-held-out splits are
stress tests. Random-row is descriptive only.

## Decision rules

- A structural model is useful only if it improves the analytical baseline on
  circuit-group holdout and does not collapse under width holdout. Otherwise
  the correct finding is that the current static features have no demonstrated
  incremental value.
- A high random-row score alone is not evidence of generalization.
- `statevector_bytes <= VRAM` is treated as a lower-bound check, never as a
  sufficient feasibility guarantee. An observed q28 resource limit with that
  lower bound passing is direct evidence for the peak-envelope requirement.
- Context-selection accuracy is reported with oracle context distribution and
  selection regret. If CUDA complex64 is always fastest, selection is a
  trivial property of this machine rather than a model contribution.
