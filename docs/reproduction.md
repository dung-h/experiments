# Reproduction

This guide rebuilds the included benchmark tables from saved predictions
and simulator measurements. It does not retrain models, repeat timings or
submit QPU jobs. Some paper-level methods remain unavailable or are local
adaptations, so the scientific report remains `PARTIAL`. The current QPU
scorecard is in the
[two-domain scorecard v3](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md).

The current scientific completion queue is
[benchmark completion](benchmark_completion.md). It distinguishes existing
table rebuilds from newly designed predictor runs. A locked protocol is not
evidence that its feature extraction, training or accuracy evaluation finished.

The current `presentation-review` candidate has been tested in a fresh local
clone; its receipt is in [reproduction validation](reproduction_validation.md).
The previously published GitHub-clone receipt predates the E1–E6 additions.
The procedure rebuilds metrics, not the full experiment from every original
dataset. The complete
Ma–Li logical QASM corpus and Qonductor `circuits.zip`/database export are not bundled; the
QPack source execution payload and optimized-angle circuits were not
recovered. Frozen labels, feature/prediction rows and aggregation inputs for
the documented table rebuild are included. See the [`external` and
`pending_review` entries](release_inventory.csv) for inputs intentionally kept
outside this candidate and [methodology](methodology.md) for reconstruction
boundaries. Original-data retrieval/retraining is not claimed as clean-clone
reproducible while the S56 rights/provenance gate is open.

## Verify and rebuild the scorecard

Clone the review branch, then restore the three large CSVs from their
row-aligned parts. Python's standard library is sufficient; Git LFS is not
required.

```bash
git clone --branch benchmark-review https://github.com/dung-h/experiments.git
cd experiments
python3 scripts/materialize_csv_parts.py
python3 scripts/verify_release_inventory.py
```

Each part is a CSV with the original header and complete records, at most
48 MiB. The partition manifest records order, row counts, sizes and SHA-256
hashes. The restore command removes repeated headers, preserves row order
and verifies that each original file is byte-identical. It refuses to
overwrite an existing file with a different hash. The full source circuit
collections listed below are not supplied by this command.

Run these commands from the repository root in a fresh Python 3.10 checkout.
The target directory must not already exist; choose a new path if repeating a
build.

```bash
python3.10 -m venv .venv-qre-verify
.venv-qre-verify/bin/python -m pip install -r benchmark_v1/requirements-verification.txt
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
.venv-qre-verify/bin/python scripts/verify_release_inventory.py
.venv-qre-verify/bin/python scripts/verify_artifacts.py
.venv-qre-verify/bin/python benchmark_v1/scripts/validate_two_domain_scorecard_v2.py --pack artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3
.venv-qre-verify/bin/python benchmark_v1/scripts/build_simulator_reader_tables_v1.py --output-dir work/rebuilds/simulator_reader_tables
.venv-qre-verify/bin/python -m benchmark_v1.scripts.validate_simulator_reader_tables_v1
diff -qr --exclude=manifest.json artifacts/benchmark_v3/simulator/reader_tables_v1 work/rebuilds/simulator_reader_tables
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv-qre-verify/bin/python benchmark_v1/scripts/aggregate_unified_qpu_method_comparisons_v2.py --analytical-attempts artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/merged_analytical_attempts_for_c4.csv --hyb-status repaired --hyb-repair-manifest artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/hyb_repair_manifest.json --output-dir work/rebuilds/qpu_method_comparisons
cmp artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/attempts.csv work/rebuilds/qpu_method_comparisons/attempts.csv
cmp artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/method_metrics.csv work/rebuilds/qpu_method_comparisons/method_metrics.csv
cmp artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/source_metrics.csv work/rebuilds/qpu_method_comparisons/source_metrics.csv
cmp artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/pairwise_comparisons.csv work/rebuilds/qpu_method_comparisons/pairwise_comparisons.csv
.venv-qre-verify/bin/python benchmark_v1/scripts/build_two_domain_scorecard_v2.py --report-manifest benchmark_v1/execution/manifests/report_finalization.json --output-dir work/repo_finalization/rebuilds/two_domain_scorecard
.venv-qre-verify/bin/python benchmark_v1/scripts/validate_two_domain_scorecard_v2.py --pack work/repo_finalization/rebuilds/two_domain_scorecard
diff -qr --exclude=manifest.json artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3 work/repo_finalization/rebuilds/two_domain_scorecard
.venv-qre-verify/bin/python -m pytest -p no:cacheprovider -q benchmark_v1/tests/test_report_finalization.py benchmark_v1/tests/test_simulator_reader_tables_v1.py
```

Expected results: the simulator table rebuild validates and its seven
non-manifest files match the published table pack; QPU aggregation reports 24
methods and 72 source-local pairs, and its `attempts.csv` plus three metric
tables match the published aggregation byte-for-byte. The scorecard validator
reports 24 QPU variants, 72 source pairs and 162 supplementary local Aer graph
rows; the build status is
`BUILT_FROM_PINNED_INPUTS_MAESTRO_PILOT_FAILED`; and the 22 non-manifest report
files match the published v3 scorecard. Manifests may differ where they record
the new output path. The two selected regression modules contain 38 tests.
Builders write under `work/rebuilds/` or
`work/repo_finalization/rebuilds/`, as shown above. None starts a simulator,
trains a model or submits a QPU job.

One historical scorecard-builder copy is deliberately retained at
`work/repo_finalization/release_candidate/benchmark_v1/scripts/` because the
immutable v3 manifest pins its bytes. It is source evidence, not a generated
rebuild or a general-purpose `work/` payload; its exception is explicit in
the release inventory.

The Python 3.10 verification lock includes CPU-only PyTorch, PyTorch Geometric,
Qiskit core and XGBoost because the full unit suite imports those runner
modules. It does not install CUDA, CUDA-Q, cuTensorNet, Qiskit Aer or
paper-specific SDKs, and it does not run model fitting. This workflow is not
a clean-room replay of every paper's original training or execution
environment. It checks frozen outputs and the supported metric rebuild only.

### Candidate simulator-predictor aggregate (E1–E6 additions)

The new aggregate command consumes the saved E3 fixed-MPS, E4 Aer and E5
family-residual out-of-fold predictions, then rebuilds the two separate
predictor tables and paired hash-bootstrap intervals. It does not fit a model,
launch CUDA, repeat simulator timing, or rewrite `CURRENT.json`. In a checkout
that contains these candidate additions, run it from the repository root:

```bash
python3.14 -m venv .venv-qre-aggregate
.venv-qre-aggregate/bin/python -m pip install -r benchmark_v1/requirements-predictive-aggregate.txt
test "$(.venv-qre-aggregate/bin/python -c 'import sys; print(sys.version.split()[0])')" = 3.14.4
.venv-qre-aggregate/bin/python -B -c 'import numpy; assert numpy.__version__ == "2.3.5"'
.venv-qre-aggregate/bin/python -B benchmark_v1/scripts/aggregate_predictive_oof_v1.py
```

The default output path is
`artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/`. The
builder is fail-closed: if that path exists, it accepts it only when every
expected file matches byte-for-byte; otherwise it exits without overwriting
it. Input hashes, method cards, target/output clocks, bootstrap seeds and
metrics are pinned by `aggregate_manifest.json` and `source_hashes.json`.
The E3/E4/E5 runner outputs and scripts must be present in the same checkout.
They are not part of the older `benchmark-review` clone receipt above. The
current review-package candidate has a separate local clean-clone receipt;
public-branch verification remains a follow-up to any later push.

The E6 artifact records Python 3.14.4 and NumPy 2.3.5; this pin matters for
byte-for-byte regeneration of bootstrap summaries. Running the builder under
the general verification environment (Python 3.10 / NumPy 2.2.6) produces
identical coverage and point metrics but can change the final floating-point
digits of bootstrap means. Because the builder is fail-closed, it refuses to
replace the pinned output on that numerical-only difference. The general
Python 3.10 environment is still the one used for the full regression suite.
The candidate regression command is:

```bash
.venv-qre-verify/bin/python -B -m pytest -p no:cacheprovider -q benchmark_v1/tests
```

The E1–E6 candidate run passed 155 tests in the matching Python 3.10.21
verification environment. The package file now declares the CPU-only test
dependencies used by this suite. The same suite also passed in the local
clean-clone check; that check reused the documented verification interpreter
rather than freshly provisioning a new environment.

## What is and is not reproduced

### Restore existing Aer context before new feature extraction

The original workspace contains
`artifacts/benchmark_v1/sim_aer_q9_full_20260927/aer_q9_full.environment.json`
with SHA-256
`cf4c721c88aed65f6ed39345925eeb790bb58ae790332909c31eff890e47c5ec`.
It records the historical FakeSherbrooke snapshot hash
`509ab97f3bf4cd0c2ddbabe2cdc790e2e440726cb2b65b468c05fc54846ec9c1`,
noise-model hash
`64d58378f06c47a1bd3d115a5ebea38748c2a1b9702937d497f99d0691b43aa5`,
and the package lock at `aer_q9_full.pip-freeze.txt` with hash
`195dd1a5f87204994a84a32a64f191802d73f295949f01ac71f17385891410d6`.
The candidate checkout omitted these context files; the previous static
Azizov materialization recorded them as missing. This is a packaging gap,
not proof that the original experiment lacked context evidence.

E1 restored these audited files and recorded their hashes. The installed
environment itself was not replayed; its recorded versions and the checked
Target/noise assets were validated against the existing local environment. E2
recreated the recorded Target, noise model and deterministic transpilation
before compiling the common-core features. Copying the environment JSON is
evidence restoration, not environment replay.
If the original snapshot or resulting compiled features cannot be reproduced,
stop that feature pipeline and report the discrepancy; do not pair a different
current FakeBackend's compiled inputs with the old timing labels as though
they were measured together.

### Retraining paths

Source retrieval is pinned in `upstream/upstream.lock.json`. For the new
common-panel work, fetch only the Ma–Li circuit source into a previously
unused directory; the following commands are a retrieval recipe, not a
claim that the full retraining campaign has already been reproduced:

```bash
git clone --no-checkout https://github.com/mooselab/Quantum-Execution-Time-Prediction.git work/sources/mali
git -C work/sources/mali checkout --detach 32c392a6ece276f1ff046d4e30052d0571ff6dc6
git -C work/sources/mali rev-parse HEAD
```

Resolve panel filenames under this checkout and verify each file against
`sim_common_q16_manifest.csv`. Do not change canonical hashes or depend on
historical `/home/server/...` paths. Qonductor's pinned revision is
`5d1ac8a90cd574a23e7544e1044681641354ff67`; its DB/ZIP association must be
re-audited before any QPU feature reconstruction. QPack reconstruction uses
the revision and six structures recorded in `docs/methodology.md`. These
retrieval/reconstruction steps do not create new observed QPU labels.

The fixed-MPS runtime graph runner and the fixed-MPS family-residual runner have
completed their five-fold OOF runs on the frozen 150-hash panel. The latter is
a runtime-only local adaptation with a family-agnostic ablation, not the
Family-Aware paper's joint runtime-quality model. Both used the recorded MPS
targets and CUDA-training environment; these scores do not establish
family-OOD performance.

The Azizov three-view runner completed five folds against the frozen
150-hash/162-member panel; its E6 aggregate passed hash-level reconciliation.
It uses explicit local source/hybrid/transpiled feature dimensions, so it is a
common-core adaptation rather than the paper's full 1,402-circuit evaluation.
A first launch was stopped by the runtime-lock guard before GNN fitting
because CUDA visibility differed from the frozen lock; it produced no GNN
predictions and is excluded. The corrected run passed fold-0 technical QA
and continued automatically through folds 1–4.

The E1–E6 runs were performed in this workspace, not in the clean clone used
for the older published package. Exact saved-table reproduction and
independent retraining remain separate capabilities. Another GPU or library
context may change numerical training results and measured runtime; the
external absolute interpreter and ignored fitting-overlay paths are not
portable dependencies.

The 8,767-row QPU report preserves source labels, common grouped outer split
and source identities. The recorded methods distinguish exact submitted
Qonductor physical QASM, exact Ma–Li logical QASM and reconstruction-qualified
QPack structure; the latter does not recover optimized angles or submitted
routing. The complete original QASM/source datasets are not all distributed
in this candidate. See [methodology](methodology.md) for exact input and
availability disclosures.

The simulator reader pack rebuilds metrics from local, frozen measurements;
the E6 predictive aggregate rebuilds metrics from saved OOF predictions.
Neither repeats Aer, CUDA-Q, MPS, cuTensorNet or Maestro timing on another
machine. Runtime values can vary by hardware, driver and execution context;
the recorded clock, precision, shots, workspace and quality gate must remain
attached to each result. The Maestro ten-cell pilot is terminal at
`pilot_gate_failed` and has no predictor score. V4 control flow is contract-only
and has no trained score.

Method-specific original-paper retraining is a separate task. Method cards,
run manifests and the [fidelity registry](../benchmark_v1/registry/method_fidelity_registry_v2.json)
state whether an evaluated result is a reproduction, reimplementation,
adaptation, native API output or unavailable. Do not call the scorecard rebuild
a reproduction of methods whose training/execution was not repeated.

The author requested publication of `benchmark-review` on 2026-10-02. The
branch contains the selected reproduction package, not complete external
source archives. No repository license, citation author list or release tag
has been chosen on the author's behalf. The
[earlier release-rights decision](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md)
continues to document unresolved licensing and source-provenance questions.
The publication request does not change scientific status, `CURRENT.json`
or third-party licenses.
