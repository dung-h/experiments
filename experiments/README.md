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
