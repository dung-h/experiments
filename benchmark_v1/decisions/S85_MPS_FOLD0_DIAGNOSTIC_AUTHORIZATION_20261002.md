# S85 — Conditional authorization: fixed-χ=16 MPS five-fold OOF

**Date:** 2026-10-02  
**State:** Conditionally authorized when simulator timing is idle, with a newly generated passing MPS preflight.  
**Scope:** One fixed-configuration, runtime-only method adaptation on the frozen C44 five-fold exact-QASM split. No family-OOD claim, paper-reproduction claim, or leaderboard promotion.

## Decision

The fixed CUDA-Q MPS runtime-only adaptation is scientifically suitable for a five-fold out-of-fold (OOF) evaluation. The C44 folds are already frozen by exact QASM hash, the MPS configuration and model choices are fixed, and no method/seed/hyperparameter selection is planned. A fold-0 checkpoint therefore should test pipeline integrity, not whether the method performs well.

If fold 0 passes the technical criteria below, the runner **automatically proceeds through folds 1–4 unchanged**. It must not inspect or use fold-0 MAE, R², model ranking, prediction quality, or seed quality when deciding whether to continue. A technical failure stops the run; any code or representation change requires a new freeze and a consistent rerun, not a patch limited to later folds.

The global `default_execution_state` remains `not_authorized`. This method-specific authorization is recorded in `benchmark_v1/protocol/simulator_completion_gates_v1.json`; it does not authorize other methods.

### M3 correction — 2026-10-02

This revision supersedes S85's earlier requirement that S84 reach 270/270 before
MPS fitting. S84 studies Maestro thread policy; it provides no input, label, split,
or calibration for this MPS predictor. Requiring its scientific completion was an
unrelated dependency. The actual requirement is resource exclusion: hold the
existing Maestro timing flock throughout any CUDA probe or MPS training and
verify no live timing controller or possible orphan worker before CUDA starts.
The host-wide `/tmp/qre-benchmark-gpu-lease.lock` used by CUDA-Q timing runners
must also be acquired with nonblocking flock and held for that entire interval.
Both locks are necessary: the Maestro lock alone does not exclude CUDA-Q timing.
Partial/blocked S84 evidence remains valid history and does not prevent this run.
Historical raw measurements, preflights and predictions remain immutable. Since
the runner, gate and decision hashes changed, a new preflight output is required.

## Frozen experiment

- Method family: `cudaq_mps_fixed_chi16_runtime_only_adaptation`.
- Method ID: `mps_fixed_bond16_warm_runtime_prediction`.
- Target clock: `cudaq_mps_fp64_bond16_warm_state_execution_seconds`, the median of existing warm `get_state` timings. Build, first execution, result extraction, fidelity-reference, end-to-end, and shots are outside this target.
- Evaluation unit: one exact input-QASM SHA-256 group. Reduce valid warm repetitions within each session/member record by median, then reduce those record medians by hash-level median.
- Data: the 150 exact-hash core panel. Keep the 144 finite runtime labels (142 quality-pass and 2 quality-failed) in fitting/scoring. Keep the 6 adapter-error targets unavailable and unimputed. Quality outcome is separate and never a predictor.
- Split: the pre-existing C44 exact-QASM hash folds, unchanged. Fold sizes are 37, 28, 26, 25, and 34 unique hashes. Each fold is test exactly once; training uses finite labels in the other four folds. Families are represented in training; this is not family-OOD evaluation.
- Same-fold baselines: training-fold median and standardized six-feature structural Ridge (`alpha=1`).
- Graph adaptation: the frozen source-DAG TransformerConv model, train-only vocabulary/preprocessing, `log1p(seconds)` target, 500 epochs, batch size 32, Adam (`lr=0.0005`, `weight_decay=0.0001`), no HPO or early stopping, fixed seeds 42, 1234, and 31415. Preserve each seed's predictions and the rowwise three-seed median.
- Compute: the pinned CUDA/PyG environment, sequential seed fits. Reuse existing measurements only; no new simulator timing, QASM execution, or QPU work.

## Fold-0 technical checkpoint — metric blind

Fold 0 is assigned 37 hashes, including its finite/unavailable statuses from the frozen target table. The runner may continue only if all of these technical checks pass:

1. Its predictions contain exactly the assigned C44 fold-0 hashes once each, with correct fold IDs and no exact-hash train/test overlap.
2. Every assigned hash has a finite, non-negative prediction from every fixed baseline, each graph seed, and the three-seed median. The 6 globally unavailable targets remain blank when encountered; they are not imputed as labels.
3. Target values/statuses, the 142/2 finite quality accounting, C44 assignment, code/input hashes, and CUDA environment match the frozen preflight.
4. All three seed outputs and both baselines are present, and the CUDA/PyG forward/backward smoke passes.

Fold-level metric computation is deferred until all five folds have passed technical QA; the continuation validator does not read metric artifacts or scores. Once these integrity checks pass, folds 1–4 start automatically with the same code, features, configuration, and seed list. MAE, MedAE, R², error quantiles, ranking, relative performance, and seed quality cannot affect continuation.

## Reporting and interpretation

The final OOF artifact must contain all 150 hashes exactly once. Report coverage separately from metrics: metrics use the common 144 finite-label rows; all six unavailable rows remain in the prediction/coverage envelope. Report each fixed baseline and graph seed plus the rowwise three-seed median. Keep quality status audit-only.

This is a local, fixed-configuration **runtime-only method adaptation**. It is not the Family-Aware paper method, not a threshold selector, not a family-held-out result, and not evidence for other MPS configurations or simulator selection. The result is not promoted into the cross-method leaderboard by this decision.

## Start conditions

Before any MPS CUDA process starts:

1. Acquire `work/locks/maestro_qcsim_v2_cpu_timing.lock` followed by `/tmp/qre-benchmark-gpu-lease.lock`, both using real nonblocking `flock`, then verify that no live S84 timing controller or Python spawn worker is present. An existing but unlocked lock file is acceptable; its mere presence is not evidence of an active worker. Both locks remain held until all CUDA activity from the current preflight or five-fold training exits, and both release on failure. Possible orphan spawn workers are conservatively blocked even if their controller has disappeared. S84's 270-row schedule and results are independent of this predictor; absent, partial or terminal-blocked S84 status is acceptable.
2. Regenerate the MPS preflight into a new output directory after this S85/gate revision. It must pass provenance, 150-hash C44 join, 144/142/2/6 target accounting, quality separation, and CUDA environment checks.
3. Run only `--action fit-five-fold-oof`. The runner verifies timing exclusion, authorization, frozen inputs, and CUDA environment before fitting. It refuses partial-fold CLI entry points and refuses to overwrite existing fold artifacts.

No work may overlap S84's exclusive host baseline. Do not alter historical raw measurements, S82/S83, prior MPS score tables, or other method gates.

## Evidence paths

- Protocol gate: `benchmark_v1/protocol/simulator_completion_gates_v1.json`.
- Runner: `benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py`.
- Focused tests: `benchmark_v1/tests/test_mps_fixed_chi16_runtime_adaptation_v1.py`.
- Existing pre-authorization preflight: `artifacts/benchmark_v1/mps_fixed_chi16_runtime_only_adaptation_v1/` (historical; regenerate to a new directory after this revision).
- Shared timing context: `benchmark_v1/decisions/S84_MAESTRO_THREADPOOL_DIAGNOSTIC_20261002.md` (operational exclusion only).
- S84 audit history: `artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002/attempt_003/run_manifest.json`; no completion/metric dependency.

## Routine execution after M3

Use `/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python`
with CUDA PyTorch/PyG. Run `--action preflight --output-dir
artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof`
first, then the same output path with `--action fit-five-fold-oof`. Both calls
acquire the shared lock before CUDA import/probe. Expected environment from the
prior GPU setup is PyTorch 2.7.1+cu128, CUDA 12.8 and PyG 2.6.1; the new preflight
records the actual environment and training must match it. CPU fallback is forbidden.
