# Azizov et al. independent reproduction

This track independently reconstructs the experiment in Azizov, Vela-Tambo and
Guo, *Transpilation-Aware Runtime Prediction for Noisy Quantum Circuit
Simulation* (arXiv:2609.12980). The paper studies measured Qiskit Aer noisy
simulation time after backend-aware transpilation. It is not a real-QPU
runtime study.

The authors have not released the code and raw runtime table at the time this
track was started. Consequently this directory is an **independent
reproduction of the experimental design**, not an exact numerical replication.

## Target semantics

The primary target is:

```text
T_exec = wall-clock time for Aer to execute the transpiled circuit
```

Transpilation time and simulator construction time are recorded separately.
The secondary quantity is `T_total = T_transpile + T_exec`; it is not mixed
into the primary target.

The primary protocol uses 1,024 shots, a noise model reconstructed from the
current Qiskit fake-provider snapshots, and optimization levels 0--3. The
paper used FakeWashington and FakeSherbrooke. Current Qiskit releases expose
the Washington snapshot as `FakeWashingtonV2`; the script records the exact
class and package versions in its output.

## Source circuits

The pinned Ma--Li checkout contains the 1,510 source QASM files used as the
starting point described by the paper. Materialise it with:

```bash
./scripts/bootstrap_upstreams.sh
```

Then use `work/mali/data/quantum_circuits`. A direct local checkout can also be
passed with `--mali-root /path/to/Quantum-Execution-Time-Prediction`.

The manifest removes duplicates by source SHA-256 and records the circuit
family, logical width/depth and gate counts. No source circuit is silently
discarded; execution failures and timeouts remain status rows.

## P0 smoke run

The first run attempts one representative circuit per source family with at
most eight logical qubits. It covers both fake backends and all four
optimization levels. A family with no unique circuit under that bound is
reported explicitly rather than silently replaced:

```bash
python tracks/azizov_independent/scripts/build_manifest.py \
  --mali-root work/mali \
  --output work/azizov_independent/source_manifest.csv

python tracks/azizov_independent/scripts/run_p0_smoke.py \
  --mali-root work/mali \
  --manifest work/azizov_independent/source_manifest.csv \
  --output artifacts/azizov_independent/p0_smoke.csv

python tracks/azizov_independent/scripts/compare_feature_blocks.py \
  --input artifacts/azizov_independent/p0_smoke.csv \
  --output artifacts/azizov_independent/p0_feature_ablation.json
```

The runner performs one untimed backend warm-up and then times only
`AerSimulator.run(...).result()`. It stores the measurement protocol, source
and transpiled metrics, status/error text, seed and environment versions in
the CSV. A single timed run is intentionally labelled as P0; the robustness
pass will add repeated cold/warm measurements.

## Local feasible subset

For a bounded local run, build an auditable screening subset first:

```bash
python tracks/azizov_independent/scripts/build_feasible_subset.py \
  --manifest work/azizov_independent/source_manifest.csv \
  --eligible work/azizov_independent/feasible_q9.csv \
  --rejected work/azizov_independent/rejected_local_screen.csv
```

The default screen is `width<=9`, `logical_ops<=4000` and
`logical_depth<=4000`. It is a **local-machine eligibility screen**, not a
claim that every eligible circuit will finish under every backend or shot
count. It is calibrated from the successful P0 observations: the hardest
confirmed P0 circuits are the eight-qubit Grover/QWalk cases. The screen
excludes the unconfirmed nine-qubit QWalk circuit with 6,199 operations and all
higher-width circuits. Run the eligible matrix with:

```bash
python tracks/azizov_independent/scripts/run_matrix.py \
  --mali-root work/mali \
  --manifest work/azizov_independent/feasible_q9.csv \
  --max-qubits 9 --shots 1024 --timeout-s 900 \
  --output work/azizov_independent/feasible_q9_runtime.csv
```

Always keep the rejected manifest. A later process-isolated runner may promote
some rejected circuits after they pass a separate memory/timeout test.

Evaluate the completed subset with grouped and transfer-oriented splits:

```bash
python tracks/azizov_independent/scripts/evaluate_subset_splits.py \
  --input artifacts/azizov_independent/feasible_q9_runtime.csv \
  --output-json artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.json \
  --output-md artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.md
```

The report treats grouped circuit ID as the primary split. Random-row scores
are retained only as an optimistic diagnostic because the same logical circuit
has multiple backend/optimization rows. Family-held-out and backend-held-out
scores test the two transfer questions that a row-random split cannot answer.

## Planned P1/P2 extensions

The checkpointed matrix runner is available for the full or bounded P1 pass:

```bash
python tracks/azizov_independent/scripts/run_matrix.py \
  --mali-root work/mali \
  --manifest work/azizov_independent/source_manifest.csv \
  --max-qubits 16 \
  --output work/azizov_independent/p1_q16.csv
```

It appends one row at a time and can be safely resumed with the same command.
Rows that exceed `--timeout-s` are retained as `status=timeout` rather than
being removed from the dataset. A bounded pilot may use
`--exclude-families ae` when a high-width amplitude-estimation circuit exceeds
local memory; that exclusion must be reported and is not a substitute for the
paper's 900-second/HPC run.

- P1: all unique circuits, both backends, optimization levels 0--3, 1,024
  shots, with a 900-second per-run timeout and explicit censoring records.
- P2: grouped circuit-ID splits, family-held-out and backend-held-out tests;
  three to five transpiler seeds; cold versus warm execution sensitivity.
- Features: source global metrics, post-transpilation metrics, and source or
  transpiled DAGs. Features that cannot be reconstructed exactly are labelled
  `proxy` rather than being presented as the paper's implementation.
- Models: median, Ridge, SVR, Random Forest/HistGradientBoosting, then GNNs.
  Evaluation reports R2, RMSE, MAE, median absolute error and Spearman on both
  log-runtime and seconds.

The independent report must compare trends and failure modes, not claim the
paper's exact R2 values without the missing artifact.

## Resource-bounded width frontier probe

`run_resource_canary.py` runs one matrix row per isolated child process and
samples aggregate RSS/CPU from `/proc`. It accepts an RSS cap and wall cap so a
memory-heavy row cannot take down the complete probe. Aer parallelism can be
made explicit with `--max-parallel-threads`, `--max-parallel-shots` and
`--max-parallel-experiments`; these settings are part of the protocol and must
not be mixed with the default-parallelism q<=9 result.

The committed q10--q16 canary is intentionally partial: seven rows were
retained, four exceeded the 16 GiB cap and three completed. Read
`artifacts/azizov_independent/Q10_Q16_RESOURCE_CANARY_PARTIAL_REPORT.md`
before attempting a larger frontier run.
