# Cross-track validation experiments

These scripts extend the four replication tracks without pooling their target
semantics.

## Qonductor mapping audit

```bash
python experiments/qonductor_mapping_audit.py --qonductor-root work/qonductor
```

The audit checks whether the 100-row public resource-estimator CSV has a stable
join to the archived circuit/job database. It does not guess a row mapping.

## Ma–Li / QCRE proxy

```bash
python experiments/mali_qcre_proxy.py --mali-root work/mali
```

The committed 340-row feature artifact can be re-scored without repeating the
transpilation (useful for checking report code):

```bash
python experiments/mali_qcre_proxy.py \
  --mali-root work/mali \
  --features-csv artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv
```

The script transpiles each Ma–Li logical QASM with the current FakeOsaka or
FakeKyoto target, reconstructs a backend-target weighted critical path, and
compares that proxy to the upstream observed `time_taken`. It reports
Pearson/Spearman association and five-fold one-dimensional log calibration for
logical depth, two-qubit depth, physical depth and the weighted path. This is
`OUR_PROXY`: the historical physical circuits and calibration snapshots are not
claimed to be recovered. Qiskit's ALAP scheduling pass is intentionally not
used because it is not robust for some large Ma–Li QASM inputs under Qiskit
1.4; the duration calculation is explicit and reproducible in the script.

## Final Ma–Li validation pass

Run the short validation pass on the committed feature CSV:

```bash
python experiments/mali_qcre_validation.py --mali-root work/mali
```

It groups repeated logical circuits by QASM SHA-256, applies one fixed grouped
five-fold split to all features and ablations, checks duration coverage, and
fits the optional `1024 * weighted_path` affine calibration. The seed-sensitivity
run re-transpiles the same 340 rows with three seeds:

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --seeds 1234,2025,31415 --optimization-level 1
```

The committed seed-row CSV can be re-scored without retranspiling:

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --rows-csv artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_rows.csv \
  --seeds 1234,2025,31415 --optimization-level 1
```

Both reports are diagnostic extensions of the proxy result; they do not turn a
FakeBackend reconstruction into historical QPU calibration data.

## Ma–Li direct source-specific estimator

The direct evaluation consumes the committed 340-row proxy feature table. It
does not retranspile or submit a QPU job. Create an isolated recent Python
environment, then install the pinned evaluation dependencies:

```bash
python3 -m venv .venv-mali-direct
.venv-mali-direct/bin/pip install -r experiments/requirements-mali-direct-estimator.txt
```

Run the evaluator with that interpreter:

```bash
.venv-mali-direct/bin/python experiments/run_mali_direct_estimator.py
```

It predicts `log1p(result.time_taken seconds)` from target-specific compiled
structure and the weighted-path proxy. Its primary split is five-fold by logical
QASM SHA-256. It also records a deliberately weaker paired-backend diagnostic,
a strict backend-plus-unseen-logical-circuit 2 × 5 split, and leave-one-family
out. The latter two must be read as ten correlated outer partitions / two
backend directions, not as broad device cross-validation.

The output directory contains a report, all model/split aggregate metrics,
per-fold metrics, strict per-backend metrics, all outer-fold predictions, and
input provenance:

```text
artifacts/validation/mali_direct_estimator/
```

No estimator is fitted and saved for deployment by this script. The output is
an evaluation artifact: a model/selection protocol must be frozen before
fitting an operational source-specific model.

## Public calibration snapshot variability

The temporal audit compares the frozen local fake-provider pair with a public
timestamped calibration series from the DAQEC-Benchmark (42 timestamps per
backend over 14 days, three replicate rows per timestamp):

```bash
python experiments/mali_snapshot_variability.py \
  --download \
  --properties-dir /path/to/Quantum-Execution-Time-Prediction/data/fake_backend_properties \
  --output-dir artifacts/validation/mali_snapshot_variability
```

For offline reruns, download the DAQEC `drift_characterization.csv` once and
replace `--download` with `--source-csv /path/to/drift_characterization.csv`.
The resulting report separates spatial variation inside one frozen snapshot
from temporal variation across public calibration timestamps. The DAQEC series
is a 2025 aggregate calibration source, not the missing per-qubit historical
Ma–Li tensor.
