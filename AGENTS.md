# Project agent rules

## Active recovery queue — 2026-10-03

The user authorized the remaining strong decisions and a routine-agent
execution/monitoring handoff. The controlling queue is now R0–R12 in
`docs/benchmark_recovery_plan.md`. Read that file completely before recovery
work. It locks the RCCX adapter repair, 900-second MPS worker recovery,
compiled-unified QPU reruns and dense-statevector predictor comparisons.
It explicitly permits independent subagents with disjoint file ownership.
Implementation and execution were not performed by writing that decision.

Within this bounded queue, its automatic technical gates and execution
authorization supersede the older role/hand-off wording below. No additional
strong-agent sign-off is needed after each successful fold. Historical
protocols/results remain immutable; new contexts get distinct identities.
The development root is the execution workspace; `publish_candidate` is
the packaging tree. Do not silently run two divergent copies of a runner.
The queue does not authorize live QPU, full V4 training, changing canonical
targets/splits, unrestricted experiments, or git/publication decisions.

## Scientific objective

Preserve the two original benchmark goals:

1. compare runtime-estimation methods on the canonical 8,767 archived
   one-circuit real-QPU observations;
2. compare simulator runtime estimators on frozen local engine/configuration
   cells and the common exact-QASM panel.

The controlling plan is
`benchmark_v1/MASTER_UNIFIED_BENCHMARK_PLAN_V2.md`, as constrained by signed
V3 recovery overlays `decisions/snapshot_policy_v3.md`,
`decisions/circuit_representation_v3.md`, and
`decisions/learned_method_contract_v3.md`. Historical task reports are
evidence, not permission to narrow or redefine this objective.

## Authority and immutability

- Do not overwrite raw data, raw predictions or historical reports.
- Resolve contradictions with a dated, explicit supersession decision.
- The Qonductor S39 archive-recovery addendum supersedes blanket exclusion of
  its 230-row L1 logical-recipe subset.  Do not call that subset byte-exact
  original logical QASM.
- Do not promote a result until its target, clock, representation, split,
  quality contract and attempt denominator are explicit.

## Comparison rules

- All methods receive the same canonical observation ledger and frozen outer
  split.  A method emits a prediction or an explicit unavailable/failure state
  for every row.
- Trainable methods use nested grouped validation inside outer-train only.
  Preprocessing, tuning, early stopping and proxy calibration never inspect
  outer-test labels.
- Compare numbers only on identical held-out rows; always report full-envelope
  coverage and failures.
- Source-stratified results and source-balanced macro summaries are primary.
  Raw pooled rows are diagnostic.
- Never infer logical identity from width, family or backend capacity alone.
  Use provenance-bearing paths/hashes and preserve lifecycle stage.

## Fidelity and naming rules

- `benchmark_v1/registry/method_fidelity_registry_v2.json` controls every
  reader-facing method label and fidelity class. It is an overlay: do not
  rename historical method IDs or mutate their artifacts merely for display.
  Historical fidelity v1 is immutable and superseded; it is not the
  controlling overlay.
- Never call a unified feature/representation port a paper-exact replication.
  In particular, V3, V3-large and V4 are **Ma--Li-style hybrid graph +
  metadata adaptations**, while `mali_graph_estimator` is the separately
  scoped Ma--Li source-native reimplementation on 340 rows.
- Display `fidelity_class`, `reader_label` and `claim_boundary` beside any
  benchmark score. Do not flatten native, adaptation, proxy, unavailable and
  local-measurement rows into one published-method leaderboard.
- Wave 4 is a signed control-flow representation contract for the unified
  hybrid adaptation. It is not training authority and it is not Ma--Li
  original; `row4477` remains structural-OOD for its fold-0 evaluation.
  V4 is contract-only (`five_fold_training_authorized` is false).

## Historical reviewer entry points (Wave 3C)

At Wave 3C, the reviewer-facing derived scorecard was two-domain scorecard
**v1**, not v4r3. The finalization queue must pin its newer snapshot explicitly.
`artifacts/benchmark_v1/results/benchmark_summary/scorecard/INDEX.json` and the
45-row v4r3 catalogue remain historical. There is no flat leaderboard.
Graph V3 is better than polynomial on the 8,763 shared rows, not universally.
Aer measurement is not Azizov (`qiskit_aer_noisy_local` / `local_measurement`).
Scheduled duration is not archived observed service time. S56 public-release
blockers remain open. This is not a license, CITATION, or git-tag decision.
Applying CURRENT fidelity v2 is not a public-release package.

## Compute policy

- Neural training/inference must use a CUDA-enabled framework and record a CUDA
  smoke test, GPU/driver/framework versions, precision, seeds and memory.
- Analytical formulas have no training.  Classical CPU baselines remain CPU
  unless a validated GPU-equivalent implementation is explicitly frozen.
- Simulator GPU timing has exclusive GPU ownership: one timing worker, no
  concurrent training or unrelated GPU workload.
- Parallelize CPU hashing, data validation, snapshot inventory and independent
  preprocessing when this cannot perturb a timed cell.
- Validate the first fold or first 10 cells before completing a long run.

## Agent roles

- The active queue is **F-S1–F-S4 / F-C1–F-C6** in the first section of
  `benchmark_v1/REPO_FINALIZATION_PLAN.md`. Strong roles freeze the report
  contract/claims/adapter before routine final aggregation and packaging;
  independent review closes the report. No new experiment/training is in
  this queue. The prior S6 execution queue in `benchmark_v1/CODING_AGENT_HANDOFF.md`
  is historical, corrected by the signed
  `benchmark_v1/decisions/maestro_zero_state_width_control.md` (S9).
  `maestro_common_panel_completion_resume.json` records the latest terminal
  run; predecessor bundles remain immutable and cannot execute. Do not
  execute historical handoff queues or call the partial policy an unchanged
  v3 run. S9 completed its 10-cell pilot and ended pilot_gate_failed because
  all 9 SV cells failed timing stability. That bundle cannot resume or expand.
  Preserve the result; a future context intervention needs a new decision.

- Strong-method tasks freeze scientific meaning: representations, formulas,
  snapshot policy, split protocol, method eligibility and claims.
- Routine tasks materialize frozen manifests, monitor runs, resume, aggregate,
  plot, hash and validate.  Routine workers must stop rather than silently
  change a scientific contract.
- Run independent strong tasks in parallel.  Run routine tasks only after their
  controlling contract is signed.
- Historical V3 recovery order: S66 snapshot policy, S67 representation, and S68 learned
  contract must be signed before C133 registry, C134 feature sidecar and C135
  target/transpilation preflight. Only then may Wave-3 analytical runs, the
  C122 semantic adjudication, and the new unified V3 runners begin. C125/C130,
  scorecards and final leaderboard remain blocked until final aggregation.
  Do not skip or reorder these gates.

## Release hygiene

- Use stable semantic artifact paths; task IDs/dates belong in manifests, not
  as the only public-facing meaning of a directory.
- Keep scratch, canary and temporary outputs outside the public allowlist.
- Every public number must resolve to raw evidence, code revision, environment,
  split and command.
- Do not commit secrets, machine-local credentials, caches or large external
  checkouts.
