#!/usr/bin/env python3
"""Audit whether Qonductor execution rows can be joined to circuit/job data.

The public resource-estimator CSV contains prediction columns but no stable
circuit/job key.  This script measures that limitation instead of guessing a
row order.  It is standard-library only and does not contact IBM Quantum.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def nearest_differences(values: list[float], candidates: list[float], scale: float) -> dict[str, float | int]:
    distances = [min(abs(value * scale - candidate) for candidate in candidates) for value in values]
    return {
        "scale_csv_to_db": scale,
        "n": len(distances),
        "nearest_median_absolute_difference": statistics.median(distances),
        "nearest_min_absolute_difference": min(distances),
        "exact_matches_below_1e-6": sum(distance < 1e-6 for distance in distances),
    }


def audit(qonductor_root: Path) -> dict[str, object]:
    csv_path = qonductor_root / "data" / "resource_estimator" / "execution_time_estimations.csv"
    db_path = qonductor_root / "data" / "database" / "quantum_scheduler.db"
    zip_path = qonductor_root / "data" / "database" / "circuits.zip"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_columns = list(rows[0]) if rows else []

    connection = sqlite3.connect(db_path)
    job_count = int(connection.execute("select count(*) from job").fetchone()[0])
    circuit_count = int(connection.execute("select count(*) from circuit").fetchone()[0])
    job_columns = [row[1] for row in connection.execute("pragma table_info(job)")]
    circuit_columns = [row[1] for row in connection.execute("pragma table_info(circuit)")]
    taken_times = [float(row[0]) for row in connection.execute("select taken_time from job")]
    backend_count = int(connection.execute("select count(distinct backend_name) from job").fetchone()[0])
    connection.close()
    with zipfile.ZipFile(zip_path) as archive:
        qasm_count = sum(name.endswith(".qasm") for name in archive.namelist())

    real_values = [float(row["real"]) for row in rows]
    return {
        "provenance_class": "DERIVED_RECOMPUTE",
        "csv": {
            "path": str(csv_path),
            "rows": len(rows),
            "columns": csv_columns,
            "has_stable_join_key": bool({"job_id", "circuit_id", "ibm_quantum_id"}.intersection(csv_columns)),
            "real_value_range": [min(real_values), max(real_values)] if real_values else [],
        },
        "database": {
            "path": str(db_path),
            "job_rows": job_count,
            "circuit_rows": circuit_count,
            "distinct_backends": backend_count,
            "job_columns": job_columns,
            "circuit_columns": circuit_columns,
            "taken_time_seconds_range": [min(taken_times), max(taken_times)] if taken_times else [],
        },
        "circuit_archive": {"path": str(zip_path), "qasm_files": qasm_count},
        "nearest_time_checks": [
            nearest_differences(real_values, taken_times, scale)
            for scale in (1.0, 0.001)
        ],
        "conclusion": (
            "No safe row-level join is established: the 100-row resource CSV has "
            "only predicted/real/dag columns, while circuit/job keys live in a "
            "separate database. Do not attach QCRE values by row order or nearest time."
        ),
    }


def report(summary: dict[str, object]) -> str:
    csv_info = summary["csv"]
    db_info = summary["database"]
    archive_info = summary["circuit_archive"]
    lines = [
        "# Qonductor circuit/job mapping audit",
        "",
        "This audit is credential-free and deliberately refuses to infer a join.",
        "",
        "| Object | Count/range |",
        "|---|---:|",
        f"| Resource-estimator rows | {csv_info['rows']} |",
        f"| Resource-estimator columns | `{', '.join(csv_info['columns'])}` |",
        f"| Database jobs | {db_info['job_rows']} |",
        f"| Database circuits | {db_info['circuit_rows']} |",
        f"| Distinct database backends | {db_info['distinct_backends']} |",
        f"| Archived QASM files | {archive_info['qasm_files']} |",
        "",
        f"Stable key in resource CSV: **{csv_info['has_stable_join_key']}**.",
        "",
        "The CSV has no `job_id`, `circuit_id` or `ibm_quantum_id`. Its `real`",
        "values cannot be safely matched to the database by row order or nearest",
        "runtime; the units and population also differ. Consequently, a QCRE",
        "comparison on the 100 headline rows is not currently identifiable.",
        "",
        f"Conclusion: {summary['conclusion']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qonductor-root", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation" / "qonductor_mapping"
    )
    args = parser.parse_args()
    root = Path(args.qonductor_root).expanduser().resolve() if args.qonductor_root else ROOT / "work" / "qonductor"
    summary = audit(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qonductor_mapping_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "QONDUCTOR_MAPPING_AUDIT.md").write_text(
        report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
