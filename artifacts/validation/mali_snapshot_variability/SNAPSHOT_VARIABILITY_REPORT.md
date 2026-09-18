# Osaka/Kyoto calibration snapshot variability audit

## Provenance

Public source: `https://zenodo.org/api/records/17881116/files/drift_characterization.csv/content`

The Qiskit fake-provider JSON is a frozen per-backend snapshot. The DAQEC file contains timestamped aggregate calibration observations, not the historical Ma–Li per-qubit tensor.

## Frozen fake-provider comparison

| Backend | Version | Snapshot date | Qubits | Median T1 (µs) | Median T2 (µs) |
|---|---|---|---:|---:|---:|
| osaka | `1.0.16` | 2024-02-28 04:34:29-05:00 | 127 | 287.31 | 139.96 |
| kyoto | `1.2.29` | 2024-02-28 06:38:20-05:00 | 127 | 220.66 | 109.59 |

This is one static property JSON per backend, not a time series.

## Public timestamp-level variation

| Backend | Timestamp snapshots | Time span | T1 median (µs) | T1 range | T1 CV | T2 median (µs) | T2 range | T2 CV |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ibm_brisbane | 42 | 2025-01-15 08:00:00+00:00 → 2025-01-28 18:00:00+00:00 | 145.84 | 63.2% | 18.1% | 81.27 | 60.1% | 14.9% |
| ibm_kyoto | 42 | 2025-01-15 08:00:00+00:00 → 2025-01-28 18:00:00+00:00 | 171.06 | 53.0% | 16.8% | 101.30 | 48.6% | 12.8% |
| ibm_osaka | 42 | 2025-01-15 08:00:00+00:00 → 2025-01-28 18:00:00+00:00 | 151.93 | 63.6% | 17.3% | 90.68 | 50.3% | 12.9% |

The public series has 42 timestamps per backend over 14 days, with three replicate rows per timestamp. The observed min/max range is a sensitivity envelope, not a monotonic drift curve.

## Interpretation

- Osaka and Kyoto both vary materially over the 14-day series; the variation is large relative to the median.
- The fitted linear slopes are only descriptive. A near-zero slope does not mean the device is stable: fluctuations can be non-monotonic.
- These are aggregate mean T1/T2 observations from a different 2025 public benchmark, not the 2024 Ma–Li calibration state.
- They support using calibration time as a domain variable and running snapshot sensitivity, but they do not identify the exact snapshot used to construct Ma–Li's missing tensor.

## Consequence for the QWalk diagnosis

The QWalk pretrained failure remains a calibration-domain-shift diagnostic. A future exactness check should rebuild the graph with several matched per-qubit snapshots and report prediction ranges, rather than selecting one snapshot and treating it as historical truth.
