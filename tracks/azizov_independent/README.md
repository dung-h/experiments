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
being removed from the dataset.

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
