# Reproduction

This guide rebuilds the included benchmark tables from saved predictions
and simulator measurements. It does not retrain models, repeat timings or
submit QPU jobs. Some planned method evaluations are unfinished;
the report therefore records `PARTIAL`. The current tables are in the
[two-domain scorecard v3](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md).

The procedure has been tested in a fresh local Git clone. It rebuilds
metrics, not the full experiment from every original dataset. The complete
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

The verification lock is Python 3.10 aggregation/test dependencies only.
It does not install CUDA, CUDA-Q, cuTensorNet, Qiskit Aer, PyTorch Geometric,
or paper-specific SDKs. This workflow is not a clean-room replay of every
paper's original training or execution environment. It checks the frozen
benchmark outputs and supported metric rebuild only.

## What is and is not reproduced

The 8,767-row QPU report preserves source labels, common grouped outer split
and source identities. The recorded methods distinguish exact submitted
Qonductor physical QASM, exact Ma–Li logical QASM and reconstruction-qualified
QPack structure; the latter does not recover optimized angles or submitted
routing. The complete original QASM/source datasets are not all distributed
in this candidate. See [methodology](methodology.md) for exact input and
availability disclosures.

The simulator reader pack rebuilds metrics from local, frozen measurements.
It does not repeat Aer, CUDA-Q, MPS, cuTensorNet or Maestro timing on another
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
