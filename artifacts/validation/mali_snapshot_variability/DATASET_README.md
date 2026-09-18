# Dataset and rerun instructions

**Finding recorded:** 2026-09-18

## Raw input

`drift_characterization.csv` is copied from the public DAQEC-Benchmark
Zenodo record:

- Record: <https://zenodo.org/records/17881116>
- Direct source: <https://zenodo.org/api/records/17881116/files/drift_characterization.csv/content>
- File: `drift_characterization.csv`
- Rows: 378 data rows plus header
- Backends: `ibm_brisbane`, `ibm_kyoto`, `ibm_osaka`
- Time span: 2025-01-15 through 2025-01-28 UTC
- License/provenance: the Zenodo record states CC-BY-4.0 for the dataset;
  preserve attribution when redistributing this raw file. It is an aggregate
  public calibration series, not Ma–Li's private or missing processed tensor.

The file contains three replicate observations at each of 42 timestamps per
backend. Its columns are:

```text
backend, day, timestamp_utc, avg_t1_us, avg_t2_us,
probe_t1_us, probe_t2_us
```

The committed raw-file SHA-256 is recorded in
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

## Re-download and verify

From the capsule root:

```bash
curl -L --retry 5 --retry-delay 2 \
  'https://zenodo.org/api/records/17881116/files/drift_characterization.csv/content?download=1' \
  -o artifacts/validation/mali_snapshot_variability/drift_characterization.csv

sha256sum artifacts/validation/mali_snapshot_variability/drift_characterization.csv
```

Expected SHA-256:

```text
7120288930e59311fd3e88974054a311373c509ce2d876fe513998a2530a71bc
```

## Reproduce the audit

The script uses only the Python standard library. The fake-provider JSON files
are already packaged in the capsule under `tracks/mali/overlay`.

```bash
python3 experiments/mali_snapshot_variability.py \
  --source-csv artifacts/validation/mali_snapshot_variability/drift_characterization.csv \
  --analysis-date 2026-09-18 \
  --properties-dir tracks/mali/overlay/data/fake_backend_properties \
  --output-dir artifacts/validation/mali_snapshot_variability
```

For a fresh download instead of the committed raw file, replace
`--source-csv ...` with `--download`.

Generated outputs:

- `SNAPSHOT_VARIABILITY_REPORT.md`: report-style findings;
- `snapshot_variability_summary.json`: machine-readable metrics and provenance;
- `public_timestamp_summary.csv`: one aggregate row per backend/timestamp.

## Interpretation boundary

The DAQEC series quantifies temporal variation of aggregate calibration
metrics. It does **not** provide the per-qubit historical T1/T2 and preprocessing
order needed to claim an exact Ma–Li tensor reconstruction. The audit therefore
supports a multi-snapshot sensitivity experiment, not a replacement for the
current `OUR_PROXY` label.
