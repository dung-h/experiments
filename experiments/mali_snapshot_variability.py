#!/usr/bin/env python3
"""Measure temporal T1/T2 variation in public Osaka/Kyoto calibration data.

The script deliberately keeps three provenance classes separate:

* local Qiskit fake-provider properties (one frozen JSON per backend);
* public DAQEC-Benchmark timestamp-level calibration observations;
* derived summary statistics.

It uses only the Python standard library so that the audit can run without
the pandas/pyarrow stack. The DAQEC file is downloaded only when
``--download`` is supplied; otherwise pass a local CSV with ``--source-csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DAQEC_URL = (
    "https://zenodo.org/api/records/17881116/files/"
    "drift_characterization.csv/content"
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    minimum = min(values)
    maximum = max(values)
    return {
        "mean_us": mean,
        "median_us": median,
        "stdev_us": stdev,
        "min_us": minimum,
        "max_us": maximum,
        "range_us": maximum - minimum,
        "range_percent_of_median": (maximum - minimum) / median * 100
        if median
        else None,
        "cv_percent": stdev / mean * 100 if mean else None,
    }


def linear_slope(xs: list[float], ys: list[float]) -> float:
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if not denominator:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def fake_properties_summary(properties_dir: Path) -> dict[str, dict]:
    output = {}
    for backend in ("osaka", "kyoto"):
        path = properties_dir / f"{backend}.json"
        data = json.loads(path.read_text())
        t1 = []
        t2 = []
        for qubit in data["qubits"]:
            values = {item["name"]: item["value"] for item in qubit}
            t1.append(float(values["T1"]))
            t2.append(float(values["T2"]))
        output[backend] = {
            "provenance_class": "QISKIT_FAKE_PROVIDER_FROZEN",
            "path": str(path),
            "backend_name": data.get("backend_name"),
            "backend_version": data.get("backend_version"),
            "last_update_date": data.get("last_update_date"),
            "n_qubits": len(data["qubits"]),
            "t1": summary(t1),
            "t2": summary(t2),
        }
    return output


def read_drift_csv(path: Path) -> tuple[dict, list[dict]]:
    raw = list(csv.DictReader(path.open(newline="")))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw:
        grouped[(row["backend"], row["timestamp_utc"])].append(row)

    timestamp_rows = []
    for (backend, timestamp), rows in sorted(grouped.items()):
        dt = parse_timestamp(timestamp)
        timestamp_rows.append(
            {
                "backend": backend,
                "timestamp_utc": timestamp,
                "day_index": int(rows[0]["day"]),
                "replicate_rows": len(rows),
                "avg_t1_us": statistics.fmean(float(r["avg_t1_us"]) for r in rows),
                "avg_t2_us": statistics.fmean(float(r["avg_t2_us"]) for r in rows),
                "probe_t1_us": statistics.fmean(float(r["probe_t1_us"]) for r in rows),
                "probe_t2_us": statistics.fmean(float(r["probe_t2_us"]) for r in rows),
                "timestamp_epoch_days": dt.timestamp() / 86400,
            }
        )

    backends = sorted({row["backend"] for row in timestamp_rows})
    public = {}
    for backend in backends:
        rows = [row for row in timestamp_rows if row["backend"] == backend]
        times = [row["timestamp_epoch_days"] for row in rows]
        first = min(times)
        relative_days = [value - first for value in times]
        t1 = [row["avg_t1_us"] for row in rows]
        t2 = [row["avg_t2_us"] for row in rows]
        public[backend] = {
            "provenance_class": "PUBLIC_DAQEC_CALIBRATION_OBSERVATIONS",
            "n_timestamp_snapshots": len(rows),
            "replicate_rows_per_timestamp": sorted(
                {row["replicate_rows"] for row in rows}
            ),
            "first_timestamp_utc": min(row["timestamp_utc"] for row in rows),
            "last_timestamp_utc": max(row["timestamp_utc"] for row in rows),
            "t1": summary(t1),
            "t2": summary(t2),
            "t1_slope_us_per_day": linear_slope(relative_days, t1),
            "t2_slope_us_per_day": linear_slope(relative_days, t2),
        }
    return public, timestamp_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "backend",
        "timestamp_utc",
        "day_index",
        "replicate_rows",
        "avg_t1_us",
        "avg_t2_us",
        "probe_t1_us",
        "probe_t2_us",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_report(
    path: Path, fake: dict, public: dict, source: str, analysis_date: str
) -> None:
    lines = [
        "# Osaka/Kyoto calibration snapshot variability audit",
        "",
        f"**Finding recorded:** {analysis_date}",
        "",
        "## Provenance",
        "",
        f"Public source: `{source}`",
        "",
        "Dataset download and rerun instructions: "
        "[DATASET_README.md](DATASET_README.md)",
        "",
        "The Qiskit fake-provider JSON is a frozen per-backend snapshot. The "
        "DAQEC file contains timestamped aggregate calibration observations, "
        "not the historical Ma–Li per-qubit tensor.",
        "",
        "## Frozen fake-provider comparison",
        "",
        "| Backend | Version | Snapshot date | Qubits | Median T1 (µs) | Median T2 (µs) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for backend in ("osaka", "kyoto"):
        item = fake[backend]
        lines.append(
            f"| {backend} | `{item['backend_version']}` | "
            f"{item['last_update_date']} | {item['n_qubits']} | "
            f"{item['t1']['median_us']:.2f} | {item['t2']['median_us']:.2f} |"
        )
    lines.extend(
        [
            "",
            "This is one static property JSON per backend, not a time series.",
            "",
            "## Public timestamp-level variation",
            "",
            "| Backend | Timestamp snapshots | Time span | T1 median (µs) | T1 range | T1 CV | T2 median (µs) | T2 range | T2 CV |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for backend in sorted(public):
        item = public[backend]
        lines.append(
            f"| {backend} | {item['n_timestamp_snapshots']} | "
            f"{item['first_timestamp_utc']} → {item['last_timestamp_utc']} | "
            f"{item['t1']['median_us']:.2f} | "
            f"{item['t1']['range_percent_of_median']:.1f}% | "
            f"{item['t1']['cv_percent']:.1f}% | "
            f"{item['t2']['median_us']:.2f} | "
            f"{item['t2']['range_percent_of_median']:.1f}% | "
            f"{item['t2']['cv_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The public series has 42 timestamps per backend over 14 days, "
            "with three replicate rows per timestamp. The observed min/max "
            "range is a sensitivity envelope, not a monotonic drift curve.",
            "",
            "## Interpretation",
            "",
            "- Osaka and Kyoto both vary materially over the 14-day series; "
            "the variation is large relative to the median.",
            "- The fitted linear slopes are only descriptive. A near-zero slope "
            "does not mean the device is stable: fluctuations can be non-monotonic.",
            "- These are aggregate mean T1/T2 observations from a different "
            "2025 public benchmark, not the 2024 Ma–Li calibration state.",
            "- They support using calibration time as a domain variable and "
            "running snapshot sensitivity, but they do not identify the exact "
            "snapshot used to construct Ma–Li's missing tensor.",
            "",
            "## Consequence for the QWalk diagnosis",
            "",
            "The QWalk pretrained failure remains a calibration-domain-shift "
            "diagnostic. A future exactness check should rebuild the graph with "
            "several matched per-qubit snapshots and report prediction ranges, "
            "rather than selecting one snapshot and treating it as historical truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--properties-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Date recorded in the report (YYYY-MM-DD).",
    )
    args = parser.parse_args()
    if bool(args.source_csv) == args.download:
        parser.error("pass exactly one of --source-csv or --download")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    source_reference = None
    if args.download:
        temporary = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temporary.close()
        urllib.request.urlretrieve(DAQEC_URL, temporary.name)
        source_path = Path(temporary.name)
        source_reference = DAQEC_URL
    else:
        source_path = args.source_csv
        source_reference = str(source_path)

    fake = fake_properties_summary(args.properties_dir)
    public, timestamp_rows = read_drift_csv(source_path)
    summary_path = args.output_dir / "snapshot_variability_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "provenance_class": "OUR_AUDIT",
                "analysis_date": args.analysis_date,
                "source_url": DAQEC_URL,
                "source_reference": source_reference,
                "fake_provider": fake,
                "public_timestamp_series": public,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(args.output_dir / "public_timestamp_summary.csv", timestamp_rows)
    write_report(
        args.output_dir / "SNAPSHOT_VARIABILITY_REPORT.md",
        fake,
        public,
        source_reference,
        args.analysis_date,
    )
    if temporary is not None:
        Path(temporary.name).unlink(missing_ok=True)
    print(summary_path)


if __name__ == "__main__":
    main()
